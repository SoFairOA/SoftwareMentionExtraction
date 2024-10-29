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
from langchain_openai import ChatOpenAI

#-----LOGGING SETUP-----#
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('software_extraction.log'),
        logging.StreamHandler()
    ]
)

SEMAPHORE_LIMIT = 10  
semaphore = Semaphore(SEMAPHORE_LIMIT)

#-----INITIALIZATION OF THE SaT MODEL FOR TEXT SEGMENTATION-----#
#-----"ud" STANDS FOR UNIVERSAL DEPENDENCIES AND "en" SPECIFIES ENGLISH AS THE LANGUAGE-----#
#-----THESE PARAMETERS ARE REQUIRED TO INCREASE SaT PRECISION DURING SEGMENTATION-----#
def initialize_sat_model():
    with tqdm(total=1, desc="Loading SaT model") as pbar:
        sat_model = SaT("sat-12l", style_or_domain="ud", language="en")
        sat_model.half().to("cuda")
        pbar.update(1)
    return sat_model

sat_model = initialize_sat_model()

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
            - Include URLs ONLY if they explicitly refer to a software.
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

#-----PROCESS TEXT FROM PARQUET FILES WITH PROGRESS TRACKING-----#
def process_text_from_parquet(parquet_file, model, split_type, window_size=None, overlap_sentences=None):
    df = pd.read_parquet(parquet_file)
    
    df_list = df.to_records(index=False)
    chunks = []
    total_documents = len(df_list)
    
    for row in tqdm(df_list, desc="Processing documents", total=total_documents):
        document_id = row['id'].split('_')[0]
        document_text = row['text']
        
        if split_type == "sentence":
            new_chunks = split_text_by_sentences(document_text, document_id, model, window_size, overlap_sentences)
        elif split_type == "paragraph":
            new_chunks = split_text_by_paragraphs(document_text, document_id, model)
        else:  # complete
            new_chunks = [(document_id, document_text)]
                
        chunks.extend(new_chunks)
    
    return chunks, window_size, overlap_sentences

#-----TEXT SPLITTING FUNCTIONS WITH PROGRESS TRACKING-----#
def split_text_by_sentences(document_text, document_id, model, window_size, overlap_sentences):
    segments = model.split(document_text, threshold=0.025)
    
    chunks = []
    total_iterations = (len(segments) - overlap_sentences) // (window_size - overlap_sentences) + 1
    
    for start in tqdm(range(0, len(segments), window_size - overlap_sentences), desc=f"Creating sentence chunks for document {document_id}", leave=False, total=total_iterations):
        chunk_list = segments[start:start + window_size]
        chunk_text = ' '.join(chunk_list)
        chunks.append((document_id, chunk_text))
    
    return chunks

def split_text_by_paragraphs(document_text, document_id, model):
    segments = model.split(document_text, do_paragraph_segmentation=True, paragraph_threshold=0.99995)
    
    chunks = []
    for segment in tqdm(segments, desc=f"Processing paragraph segments for document {document_id}", leave=False, total=len(segments)):
        chunk_text = ' '.join(segment) if isinstance(segment, list) else segment
        chunks.append((document_id, chunk_text))
    
    return chunks

#-----EXECUTE ASYNCHRONOUSLY WITH RETRIES AND PROGRESS TRACKING-----#
async def execute_with_retries(func, *args, max_retries=8, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return await func(*args, **kwargs)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                retries += 1
                wait_time = (1 ** retries) + random.uniform(0, 1)
                logging.warning(f"ThrottlingException: Retry {retries}/{max_retries}, waiting {wait_time} seconds.")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"ClientError encountered: {e}")
                raise
    logging.error(f"Max retries exceeded for function {func.__name__}")
    raise Exception(f"Max retries exceeded after {max_retries} attempts.")

#-----PROCESS CHUNK WITH BEDROCK ASYNCHRONOUSLY-----#
async def process_chunk_with_bedrock_async(llm, chunk, prompt, index):
    async with semaphore:
        logging.info(f"Processing chunk {index} with chosen model...")
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
                5. Ensure to include every instance of a software you find, even if repeated more than once
                </final_check>

                <note>
                It's better to include a potential software and mark it as "possible" than to omit it entirely.
                It is ABSOLUTELY essential that no examples provided are included unless they are strictly present in the text you are analyzing. Do not assume, infer, or guess. 
                </note>
                """)
        user_message = HumanMessage(content=prompt + chunk)
        response = await execute_with_retries(llm.abatch, inputs=[[system_message, user_message]])
        logging.info(f"Chunk {index} processed successfully.")
        return index, response[0].content

#-----RUN LLM WITH PARQUET FILE-----#
async def run_llm_with_parquet_bedrock(parquet_file, model_name, temperature, split_type, prompt_type, top_p, top_k, max_tokens, window_size=12, overlap_sentences=2):
    start_time = datetime.now()
    logging.info(f"Starting processing at {start_time}")
    logging.info(f"Initializing LLM: {model_name}")
    if "claude" in model_name:
        with tqdm(total=1, desc="Initializing LLM") as pbar:
            llm = ChatBedrock(
                model_id=model_name,
                region_name="eu-central-1",
                model_kwargs=dict(
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    top_k=top_k, 
                    top_p=top_p)
            )
            pbar.update(1)
    elif "llama" in model_name:
        with tqdm(total=1, desc="Initializing LLM") as pbar:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_base=os.getenv("OPENAI_BASE_URL"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )
        pbar.update(1)
        
    chunks, window_size_used, overlap_used = process_text_from_parquet(parquet_file, sat_model, split_type, window_size, overlap_sentences)
    
    #-----GROUP CHUNKS BY THEIR DOCUMENT ID WITH PROGRESS LOGGING-----#
    grouped_data = {}
    total_chunks = len(chunks)
    for document_id, chunk_text in tqdm(chunks, desc="Grouping chunks by document", total=total_chunks):
        if document_id not in grouped_data:
            grouped_data[document_id] = []
        grouped_data[document_id].append(chunk_text)
    logging.info(f"Total documents grouped: {len(grouped_data)}")
    
    extracted_softwares_total = []
    question = get_prompt(prompt_type)
    
    #-----PROCESS DOCUMENTS WITH THEIR PROGRESS BAR-----#
    total_documents = len(grouped_data)
    for document_id, chunk_list in tqdm(grouped_data.items(), desc="Processing documents", total=total_documents):
        extracted_softwares = []
        
        if split_type in ["paragraph", "sentence"]:
            tasks = [process_chunk_with_bedrock_async(llm, chunk, question, i) for i, chunk in enumerate(chunk_list)]
            results = await asyncio.gather(*tasks)
            
            for i, result in sorted(results):
                software_list = result.split(",")
                extracted_softwares.extend([software.strip() for software in software_list if software.strip()])
        elif split_type == "complete":
            document_text = "\n".join(chunk_list)
            result = await process_chunk_with_bedrock_async(llm, document_text, question, 0)
            software_list = result[1].split(",")
            extracted_softwares.extend([software.strip() for software in software_list if software.strip()])
        
        extracted_softwares_total.append({
            "document_id": document_id,
            "softwares": extracted_softwares
        })
        logging.info(f"Document {document_id} processed. Softwares found: {len(extracted_softwares)}")
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    logging.info(f"Finished processing all documents. Total time taken: {processing_time}")
    logging.info(f"Documents processed: {len(grouped_data)}")
    logging.info(f"Total software mentions extracted: {sum(len(doc['softwares']) for doc in extracted_softwares_total)}")
    
    return extracted_softwares_total, start_time, end_time, question, "PARQUET", top_p, top_k, max_tokens, window_size_used, overlap_used