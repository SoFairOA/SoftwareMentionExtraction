import pandas as pd
import os
import re
import logging
from tqdm import tqdm
from datetime import datetime
import asyncio
from asyncio import Semaphore
from langchain_aws import ChatBedrock
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from wtpsplit import SaT
import time
import random
from botocore.exceptions import ClientError

SEMAPHORE_LIMIT = 1  
semaphore = Semaphore(SEMAPHORE_LIMIT)

#-----INITIALIZATION OF THE SaT MODEL FOR TEXT SEGMENTATION-----#
#-----"ud" STANDS FOR UNIVERSAL DEPENDENCIES AND "en" SPECIFIES ENGLISH AS THE LANGUAGE-----#
#-----THESE PARAMETERS ARE REQUIRED TO INCREASE SaT PRECISION DURING SEGMENTATION-----#
sat_model = SaT("sat-12l", style_or_domain="ud", language="en")
sat_model.half().to("cuda")  # Utilize GPU if available
logging.basicConfig(level=logging.INFO)

#-----BASING ON THE INPUT GIVEN TO THE LAUNCHER RETURNS A SPECIFIC KIND OF PROMPT-----#
def get_prompt(prompt_type):
    if prompt_type == "generic":
        return ("""
        <task_context>Extract all named software tools from the given text, including version numbers or specific editions mentioned, along with additional contextual information. ONLY report information explicitly stated in the text.</task_context>
        <tone_context>Ensure accuracy, be concise, and NEVER invent or assume information.</tone_context>
        <detailed_task_description>
        - List EACH software occurrence, even if repeated, but ONLY if explicitly mentioned in the text.
        - Extract ONLY explicit mentions of software. DO NOT invent or assume any software names.
        - For each software mention, determine:
          1. Action: Whether the software was created, used, or shared. Use ONLY if explicitly stated.
          2. Location: Where in the text it was mentioned (footnotes, bibliography, title, main body).
        - If the action or location is not explicitly stated, use "unknown" for that field.
        - If NO software is mentioned in the text, respond ONLY with "None".
        </detailed_task_description>
        <output_format>
        CRITICALLY IMPORTANT: Your response MUST ONLY contain the extracted information in the following format:
        Software_Name|Action|Location
        Separate multiple entries with commas (,).
        Example: "MATLAB|used|main body,Python|created|footnotes"
        Do not include any thinking process, explanations, or additional formatting.
        If no software is mentioned, respond ONLY with "None".
        </output_format>
        <post_processing>
        After composing your response, double-check that every piece of information comes directly from the text. Remove any entries that you're not 100% certain were explicitly mentioned.
        </post_processing>
        <final_check>
        Verify that your response contains ONLY entries for software explicitly mentioned in the text. If you have any doubt about a piece of information, remove it. It's better to provide less information than to include anything that might be incorrect or assumed.
        </final_check>
        <note>
        NEVER invent or assume information. If no software names are found or if you're unsure, respond ONLY with "None".
        </note>
        """)
    elif prompt_type == "detailed":
        return ("""
        <task_context>Extract all named software tools from the given text, including version numbers or specific editions mentioned, along with additional contextual information. ONLY report information explicitly stated in the text.</task_context>
        <tone_context>Ensure accuracy, be as detailed as possible while maintaining conciseness, and NEVER invent or assume information.</tone_context>
        <detailed_task_description>
            - List EACH software occurrence, even if repeated, but ONLY if explicitly mentioned in the text.
            - Identify and extract all explicitly mentioned software names, including applications, platforms, libraries, and tools.
            - Include version numbers, editions, or configurations ONLY when explicitly specified in the text.
            - Incldue URLs ONLY if they explicitly refer to a software.
            - For each software mention, determine:
              1. Action: Whether the software was created, used, or shared. Use ONLY if explicitly stated.
              2. Location: Where in the text it was mentioned (footnotes, bibliography, title, main body).
            - If the action or location is not explicitly stated, use "unknown" for that field.
            - Avoid generic terms unless they are part of the software's official name as mentioned in the text.
            - List each software occurrence separately, even if repeated with different versions or contexts.
            - DO NOT infer, assume, or generate any information not directly stated in the text.
            - If NO software is mentioned in the text, respond ONLY with "None".
        </detailed_task_description>
        <output_format>
        CRITICALLY IMPORTANT: Your response MUST ONLY contain the extracted information in the following format:
        Software_Name|Action|Location
        Separate multiple entries with commas (,).
        Example: "MATLAB R2021b|used|main body,Python 3.9|created|footnotes"
        Do not include any thinking process, explanations, or additional formatting.
        If no software is mentioned, respond ONLY with "None".
        </output_format>
        <examples>
        Incorrect: "SPSS|used|main body" (when the text only mentions SPSS without specifying its use)
        Correct: "SPSS|unknown|main body"

        Incorrect: "TensorFlow 2.4|created|bibliography,PyTorch|used|main body" (when PyTorch's use is not explicitly stated)
        Correct: "TensorFlow 2.4|created|bibliography,PyTorch|unknown|main body"
        </examples>
        <post_processing>
        After composing your response, rigorously review each entry. Ensure that every piece of information comes directly from the text. Remove any entries or parts of entries that are not explicitly stated in the text.
        </post_processing>
        <final_check>
        Verify that your response contains ONLY entries for software explicitly mentioned in the text, with actions and locations only included when directly stated. If you have any doubt about a piece of information, remove it or mark it as "unknown". It's better to provide less information than to include anything that might be incorrect or assumed.
        </final_check>
        <note>
        NEVER invent or assume information. If no software names are found or if you're unsure about any information, respond ONLY with "None".
        </note>
        """)
    elif prompt_type == "in-context":
        return ("""
        <task_context>Extract all explicitly named software tools from the given text, including version numbers or specific editions mentioned, along with additional contextual information. ONLY report information explicitly stated in the text.</task_context>
        <tone_context>Ensure accuracy, be as detailed as possible while maintaining conciseness, and NEVER invent or assume information.</tone_context>
        <detailed_task_description>
            - Identify and extract all explicitly mentioned software names, including applications, platforms, libraries, and tools.
            - Include version numbers, editions, or configurations ONLY when explicitly specified in the text.
            - For each software mention, determine:
              1. Action: Whether the software was created, used, or shared. Use ONLY if explicitly stated.
              2. Location: Where in the text it was mentioned (footnotes, bibliography, title, main body).
            - If the action or location is not explicitly stated, use "unknown" for that field.
            - Avoid generic terms unless they are part of the software's official name as mentioned in the text.
            - List each software occurrence separately, even if repeated with different versions or contexts.
            - DO NOT infer, assume, or generate any information not directly stated in the text.
            - If NO software is mentioned in the text, respond ONLY with "None".
        </detailed_task_description>
        <examples>
        Text: 'In our experiments, we utilized MATLAB R2021b for signal processing and R 4.1.2 for statistical analysis.'
        Correct response: MATLAB R2021b|used|main body,R 4.1.2|used|main body

        Text: 'The study was conducted using SPSS 27 for data analysis and Jupyter Notebook with Python 3.9 for scripting.'
        Correct response: SPSS 27|used|main body,Jupyter Notebook|used|main body,Python 3.9|unknown|main body

        Text: 'We implemented our neural network model using PyTorch 1.10 and CUDA 11.4 to accelerate computations.'
        Correct response: PyTorch 1.10|used|main body,CUDA 11.4|used|main body

        Text: 'Various software packages are available for this type of analysis.'
        Correct response: None
        </examples>
        <output_format>
        CRITICALLY IMPORTANT: Your response MUST ONLY contain the extracted information in the following format:
        Software_Name|Action|Location
        Separate multiple entries with a commas (,).
        Do not include any thinking process, explanations, or additional formatting.
        If no software is explicitly mentioned, respond ONLY with "None".
        </output_format>
        <post_processing>
        After composing your response, rigorously review each entry. Ensure that every piece of information comes directly from the text. Remove any entries or parts of entries that are not explicitly stated in the text.
        </post_processing>
        <final_check>
        Verify that your response contains ONLY entries for software explicitly mentioned in the text, with actions and locations only included when directly stated. If you have any doubt about a piece of information, remove it or mark it as "unknown". It's better to provide less information than to include anything that might be incorrect or assumed.
        </final_check>
        <note>
        NEVER invent or assume information. If no software names are found or if you're unsure about any information, respond ONLY with "None".
        </note>
        """)
    else:
        raise ValueError("Invalid prompt type provided.")


#-----UPLOAD AND MANAGE DATA FROM PARQUET FILES-----#
def process_text_from_parquet(parquet_file, model, split_type, window_size=None, overlap_sentences=None):
    #-----UPLOAD AND CONVERT THE PARQUET FILE INTO A LIST OF RECORDS TO MINIMIZE THE OVERHEAD ASSOCIATED WITH THE DATAFRAME STRUCTURE-----#
    df = pd.read_parquet(parquet_file)
    df_list = df.to_records(index=False)

    #-----CREATE AN EMPTY LIST TO STORE THE CHUNKS GENERATED FROM THE TEXT SEGMENTATION-----#
    chunks = []

    #-----ITERATE OVER EACH ROW IN THE PARQUET FILE, EXTRACTING THE DOCUMENT ID AND TEXT FOR PROCESSING-----#
    for row in df_list:
        document_id = row['id'].split('_')[0]  
        document_text = row['text']  

        logging.info(f"Processando il documento ID: {document_id} con split_type: {split_type}")

        #-----SENTENCE SPLITTING LOGIC USING SaT MODEL-----#
        #-----THIS USES A THRESHOLD OF 0.025 AS SPECIFIED IN SaT DOCUMENTATION-----#
        if split_type == "sentence":
            segments = model.split(document_text, threshold=0.025)
            start = 0

            #-----SLIDING WINDOW LOGIC TO MAINTAIN CONTEXT BETWEEN CHUNKS WITH SENTENCE OVERLAP-----#
            while start < len(segments):
                chunk_list = segments[start:start + window_size]
                chunk_text = ' '.join(chunk_list)
                chunks.append((document_id, chunk_text))
                start = min(start + window_size - overlap_sentences, len(segments))

        #-----FLATTEN PARAGRAPH SEGMENTS: WHEN USING SaT WITH `do_paragraph_segmentation=True`, THE MODEL RETURNS A LIST OF LISTS-----#
        #-----EACH INNER LIST REPRESENTS A PARAGRAPH, WHERE EACH ELEMENT IS A SENTENCE WITHIN THAT PARAGRAPH-----#
        #-----THIS LOGIC ENSURES THAT EACH INNER LIST (PARAGRAPH) IS JOINED INTO A SINGLE STRING BY COMBINING THE SENTENCES WITHIN-----#
        #-----HOWEVER, EACH PARAGRAPH REMAINS AS A SEPARATE CHUNK, MEANING THE MODEL WILL ANALYZE EACH PARAGRAPH INDIVIDUALLY-----#
        #-----THE CODE DOES NOT COMBINE ALL PARAGRAPHS INTO ONE STRING, BUT PRESERVES THEIR SEPARATE STRUCTURE DURING PROCESSING-----#
        elif split_type == "paragraph":
            segments = model.split(document_text, do_paragraph_segmentation=True, paragraph_threshold=0.99995)
            flattened_segments = [' '.join(segment) if isinstance(segment, list) else segment for segment in segments]
            for chunk in flattened_segments:
                chunks.append((document_id, chunk))

        #-----COMPLETE DOCUMENT HANDLING: NO SPLITTING, KEEP THE WHOLE DOCUMENT AS A SINGLE CHUNK-----#
        elif split_type == "complete":
            chunks.append((document_id, document_text))

    #-----LOG THE NUMBER OF CHUNKS CREATED AND SHOW A PREVIEW OF THE FIRST 5 CHUNKS-----#
    logging.info(f"Numero di chunk creati: {len(chunks)}")
    for i, (doc_id, chunk) in enumerate(chunks[:5]):
        logging.info(f"Chunk {i} (Document ID {doc_id}): {chunk[:100]}...")

    #-----RETURN THE LIST OF CHUNKS ALONG WITH WINDOW SIZE AND OVERLAP SENTENCES USED FOR SPLITTING-----#
    return chunks, window_size, overlap_sentences






async def process_chunk_with_bedrock_async(llm, chunk, prompt, index, semaphore):
    
    #-----USE A SEMAPHORE TO LIMIT THE NUMBER OF CONCURRENT REQUESTS SENT TO THE LLM-----#
    async with semaphore:
        retries = 0
        #-----SET THE MAXIMUM NUMBER OF RETRIES TO HANDLE THROTTLING OR TEMPORARY ERRORS-----#
        max_retries = 8  

        while retries < max_retries:
            try:
                #-----LOGGING TO INDICATE WHICH CHUNK IS BEING PROCESSED AND RETRY ATTEMPT-----#
                logging.info(f"Processing chunk {index}: {chunk[:100]}...")
                logging.info(f"Sending request for chunk {index} to Bedrock LLM. Retry attempt: {retries}")
                
                #-----CREATE SYSTEM MESSAGE AND USER MESSAGE TO DEFINE THE CONTEXT AND THE TASK-----#
                system_message = SystemMessage(content="""
                    <task_context>You are an advanced language model specialized in extracting software mentions from text. Act as a comprehensive software extraction assistant, focusing on identifying a wide range of software tools, systems, and platforms.</task_context>

                    <tone_context>Maintain a professional and thorough approach, balancing precision with comprehensive identification.</tone_context>

                    <rules>
                    - Extract ALL potential software names, including applications, platforms, systems, libraries, tools, and frameworks.
                    - Include version numbers, editions, or configurations when specified.
                    - For each software mention, determine:
                    1. Action: Whether the software was created, used, shared, or mentioned. Use "mentioned" if the action is not clear.
                    2. Location: Where in the text it was found (footnotes, bibliography, title, main body).
                    - If the action or location is not explicitly stated, use "unknown" for that field.
                    - Be inclusive: If an item could reasonably be considered software, include it.
                    - Use "possible" in the Action field for items that might be software but are not definitively identified as such.
                    - Always think through your response carefully, considering both explicit and implicit software mentions.
                    - If no software is mentioned in the text, respond ONLY with "None".
                    - Extract ONLY software explicitly mentioned in the text.
                    - DO NOT include software from the examples if they are NOT strictly present in the text. This means that you MUST NEVER include software names from the examples unless they are directly and clearly mentioned in the text you are analyzing. 
                    - Ensure that you NEVER assume or infer the presence of software based on previous examples.
                    - Always verify that each software mention is strictly found in the current text and not inferred from examples.
                    </rules>

                    <output_format>
                    Prefix your response with "Assistant: →" followed by a newline.
                    Then, respond with entries in the following format:
                    Software_Name|Action|Location
                    Separate multiple entries with a commas (,).
                    Example:
                    Assistant: →
                    MATLAB R2021b|used|main body,Python 3.9|created|footnotes,z-Tree|mentioned|main body,ORSEE|possible|bibliography

                    If no software is found, respond ONLY with:
                    Assistant: →
                    None
                    </output_format>

                    <examples>
                    - "The experiment was conducted using z-Tree." → z-Tree|used|main body
                    - "Participants were recruited via ORSEE (Online Recruitment System for Economic Experiments)." → ORSEE|used|main body
                    - "We utilized a custom-built system for data analysis." → custom-built system|used|main body
                    - "We have developed MUMMALS, a program to construct multiple protein sequence alignment using probabilistic consistency. MUMMALS improves alignment quality by using pairwise alignment" → MUMMALS|used|main body, → MUMMALS|used|main body
                    </examples>

                    <final_check>
                    Before submitting your response:
                    1. Review the text for any mentions of tools, systems, or platforms that could be considered software.
                    2. Ensure you've included all potential software, even if you're not 100% certain.
                    3. Verify that the response is in the specified format, including the prefix.
                    4. Double-check that you have ONLY included software names that are directly present in the text. Examples provided are NEVER to be included unless strictly mentioned in the text.
                    5. Ensure to include every istance of a software you find, even if repeated more than once
                    </final_check>

                    <note>
                    It's better to include a potential software and mark it as "possible" than to omit it entirely.
                    It is ABSOLUTELY essential that no examples provided are included unless they are strictly present in the text you are analyzing. Do not assume, infer, or guess. 
                    </note>
                    """)            
                user_message = HumanMessage(content = prompt + chunk)
                                
                #-----SEND THE REQUEST TO THE LLM USING THE ADAPTER TO STRUCTURE THE MESSAGES-----#
                response = await llm.abatch(inputs=[[system_message, user_message]])
                
                #-----LOG THE SUCCESSFUL RESPONSE FROM BEDROCK AND WAIT TO AVOID THROTTLING-----#
                logging.info(f"Received response for chunk {index}.")
                
                #-----ADD A MANUAL WAIT BETWEEN REQUESTS TO AVOID THROTTLING-----#
                await asyncio.sleep(5) 
                
                #-----RETURN THE INDEX AND THE RESPONSE CONTENT-----#
                return index, response[0].content  
            
            except ClientError as e:
                #-----HANDLE THROTTLING EXCEPTION BY INCREASING WAIT TIME EXPONENTIALLY-----#
                if e.response['Error']['Code'] == 'ThrottlingException':
                    retries += 1
                    #-----INCREASE WAIT TIME WITH EXPONENTIAL BACKOFF-----#
                    wait_time = (3 ** retries) + random.uniform(10, 15)  
                    logging.error(f"Error raised by bedrock service: {e}. Retrying for chunk {index}. Retry attempt: {retries}. Waiting for {wait_time} seconds.")
                    #-----WAIT BEFORE RETRYING TO AVOID THROTTLING-----#
                    await asyncio.sleep(wait_time)  
                else:
                    #-----RAISE THE ERROR IF IT'S NOT A THROTTLING EXCEPTION-----#
                    raise
        
        #-----RAISE AN EXCEPTION IF THE MAXIMUM NUMBER OF RETRIES IS EXCEEDED-----#
        raise Exception(f"Max retries exceeded for chunk {index}.")

async def run_llm_with_parquet_bedrock(parquet_file, model_name, temperature, split_type, prompt_type, top_p, top_k, max_tokens):
    extracted_softwares_total = []
    start_time = datetime.now()
    #-----INITIALIZE THE BEDROCK LLM WITH SPECIFIED PARAMETERS LIKE TEMPERATURE, TOP_P, TOP_K, AND MAX_TOKENS-----#
    logging.info(f"Initializing Bedrock LLM: {model_name} with temperature: {temperature}, top_p: {top_p}, top_k: {top_k}, max_tokens: {max_tokens}")

    #-----INITIALIZATION OF THE LLM MODEL WITH THE SPECIFIED PARAMETERS-----#
    llm = ChatBedrock(
        model_id=model_name,
        region_name="eu-central-1",
        model_kwargs=dict(
            temperature=temperature, 
            max_tokens=max_tokens, 
            top_k=top_k, 
            top_p=top_p)
    )

    #-----LOAD DATA FROM THE PARQUET FILE AND SPLIT THE TEXT BASED ON THE SPECIFIED SPLIT_TYPE-----#
    chunks, window_size_used, overlap_used = process_text_from_parquet(parquet_file, sat_model, split_type)
 
    #-----GROUP THE CHUNKS BY DOCUMENT_ID TO KEEP RESULTS AGGREGATED BY DOCUMENT-----#    
    grouped_data = {}
    for document_id, chunk_text in chunks:
        if document_id not in grouped_data:
            grouped_data[document_id] = []
        grouped_data[document_id].append(chunk_text)
    #-----PROCESS EACH DOCUMENT (GROUP OF CHUNKS) USING TQDM TO TRACK PROGRESS-----#
    for document_id, chunk_list in tqdm(grouped_data.items(), total=len(grouped_data), desc="Processing documents"):
        extracted_softwares = []
        question = get_prompt(prompt_type)

        if split_type in ["paragraph", "sentence"]:
            #-----PROCESS EACH CHUNK WITHIN A DOCUMENT AND TRACK PROGRESS WITH TQDM-----#
            tasks = [
                process_chunk_with_bedrock_async(llm, chunk, question, i, semaphore)
                for i, chunk in tqdm(enumerate(chunk_list), total=len(chunk_list), desc=f"Processing chunks for document {document_id}", leave=False)
            ]
            #-----USE ASYNCIO.GATHER TO PROCESS ALL CHUNKS CONCURRENTLY AND COLLECT RESULTS-----#
            results = await asyncio.gather(*tasks)
            #-----COLLECT AND PARSE THE RESULTS INTO SOFTWARE LISTS-----#
            for i, result in sorted(results):
                software_list = result.split(",")
                extracted_softwares.extend([software.strip() for software in software_list if software.strip()])
        #-----FOR COMPLETE SPLIT, PROCESS THE ENTIRE DOCUMENT TEXT AS A SINGLE CHUNK-----#
        elif split_type == "complete":
            document_text = "\n".join(chunk_list)  
            logging.info(f"Processing complete document {document_id}...")
            result = await process_chunk_with_bedrock_async(llm, document_text, question, 0, semaphore)

            software_list = result[1].split(",")
            extracted_softwares.extend([software.strip() for software in software_list if software.strip()])

        #-----ADD THE AGGREGATED RESULTS FOR THE DOCUMENT TO THE FINAL LIST-----#
        extracted_softwares_total.append({
            "document_id": document_id,
            "softwares": extracted_softwares
        })

    #-----AFTER PROCESSING ALL DOCUMENTS, LOG THE TOTAL TIME TAKEN-----#
    end_time = datetime.now()
    logging.info(f"Finished processing all documents. Total time taken: {end_time - start_time}")
    
    #-----RETURN THE AGGREGATED SOFTWARE MENTIONS AND PROCESS PARAMETERS-----#
    return extracted_softwares_total, start_time, end_time, question, "PARQUET", top_p, top_k, max_tokens, window_size_used, overlap_used
