import pandas as pd
import os
import logging
from datetime import datetime
import asyncio
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from utils import semaphore, KEYWORD_PATTERNS, keyword_based_filtering, semantic_similarity_filter_torch, segment_text_with_overlap, process_text_from_parquet, process_text_from_hf_parquet, extract_json_from_response, get_chunk_embedding, get_query_embedding, convert_keywords_to_patterns, normalize_keyword, programming_language_description, url_description, software_description, url_sentences, software_sentences, programming_language_sentences, save_extracted_entities, predefined_queries, query_embedding_cache, execute_with_retries
from jinja2 import Environment, FileSystemLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = Environment(loader=FileSystemLoader("templates"))

async def process_chunk_with_llm_async(llm, chunk, index):
    async with semaphore:
        logging.info(f"Processing chunk {index} with chosen model...")
        # system_extraction_template = env.get_template("system_extraction.j2")
        # human_extraction_template = env.get_template("human_extraction.j2")

        # system_message = SystemMessage(content=system_extraction_template.render(
        #     software_description=software_description,
        #     url_description=url_description,
        #     programming_language_description=programming_language_description,
        #     software_sentences="\n".join(software_sentences),
        #     url_sentences="\n".join(url_sentences),
        #     programming_language_sentences="\n".join(programming_language_sentences),
        #     # examples_section=examples_section
        # ))

        # human_message = HumanMessage(content=human_extraction_template.render(chunk=chunk))
        system_message = SystemMessage(content=f"""
        You are Professor John, an expert in entity extraction, tasked with identifying mentions of software, URLs, and programming languages in academic research text. Follow the descriptions and examples below for guidance on each category:\n

        **Software**: 
        {software_description}\n

        **URL**: 
        {url_description}. URLs should only be extracted if they are directly related to a software mentioned in the text. Ensure that each URL is associated with the software it belongs to.\n

        **Programming Language**: 
        {programming_language_description}. Programming languages should only be extracted if they are explicitly mentioned in relation to a software. If a language is associated with a software, include it under the software's `"language"` field.\n

        Use the following examples for context. These sentences illustrate typical mentions of software, URLs, and programming languages in research contexts. Pay attention to patterns in language that indicate each type of entity explicitly:\n

        - **Software Examples**: 
        {software_sentences}\n

        - **URL Examples**: 
        {url_sentences}\n

        - **Programming Language Examples**: 
        {programming_language_sentences}\n
        
        **To extract entities correctly, follow these steps**:\n
        1. **Identify and extract the software name**: Look for specific software tools, platforms, or programs explicitly mentioned in the text. Make sure the software name is stated clearly and explicitly, not as part of a generic term like 'tool' or 'platform'.\n
        2. **Check if a version is mentioned**: After identifying the software, check if a version number or specific edition is mentioned. If no version is mentioned, leave the field empty.\n
        3. **Check if a URL is associated with the software**: Look for any URL that is directly associated with the software mentioned in the text. If a URL is explicitly given, include it in the output.\n
        4. **Identify any programming languages mentioned**: Check if any programming language is mentioned, especially if it is related to the software or research method. If found, include the language and its version if available.\n
        5. **Identify the Publisher**: For each software entity extracted check if there is a publisher associated to it.\n
        
        **Rules**:\n
        1. Extract only explicitly named software, URLs, and programming languages.\n
        2. URLs should only be extracted if they are directly related to a software. If a software is mentioned, include its associated URL if available.\n
        3. Programming languages should only be included if they are related to the software. For example, if `SciPy` is mentioned with `Python`, include Python under the `"language"` field for SciPy.\n
        4. Publishers should be extracted only if directly stated in the text, not desumed from your knowledge.
        4. Respond strictly in JSON array format and **nothing else**.\n
        5. If you don't find informations to fill the fields in the JSON array you must reproduce the structure as it is leaving the fields blank: []
        
        **Examples**:\n
        Format each identified entity as follows:\n
        Correct output:\n

        [
        {{
            "software": "<software_name>",
            "version": ["<software_version>"], 
            "publisher": ["<software_publisher>"],
            "url": ["<software_url>"], 
            "language": ["<software_language>"]
        }},
        ...
        ]

        Incorrect output:\n

        Any introductory text, explanations, or lists outside the JSON array.\n
        Any response not formatted as JSON.\n

        When there are no entities to extract, respond only with: [].\n

        Return only the JSON array of identified entities and no other text. Follow the format precisely, or the response will be invalid.\n
        DO NOT USE EXAMPLES PROVIDED AS OUTPUT. STICK TO OUTPUT FORMAT. DO NOT ASSUME INFORMATION THAT AREN'T PRESENT IN THE TEXT.
        """)

        human_message = HumanMessage(content=f"""Hello Professor John! Please read very carefully the text i will provide to you and extract only software, URLs, and programming languages and return them in strict JSON array format. As you usually do think step by step.\n

        Text:\n
        {chunk}

        Return your response in the following format:\n

        [
        {{
            "software": "<software_name>", 
            "version": ["<software_version>],             
            "publisher": ["<software_publisher>"], 
            "url": ["<software_url>"], 
            "language": ["<software_language>"]
        }}
        ]

        If there are no entities, respond with [].\n
        Extract only what you read in the text, don't generate data, it's ok to not extract anything if it's not in the source text. STICK TO OUTPUT FORMAT""")

        
        try:
            response = await execute_with_retries(llm.abatch, inputs=[[system_message, human_message]])
            raw_content = response[0].content
            logging.info(f"Raw model response for chunk {index}: {raw_content}")

            extracted_json = extract_json_from_response(raw_content)
            if extracted_json is not None:
                logging.info(f"Parsed JSON for chunk {index}: {extracted_json}")
                return index, extracted_json
            else:
                logging.error(f"JSON decoding error in chunk {index}: No valid JSON was found.")
                return index, []

        except Exception as e:
            logging.error(f"Error processing chunk {index}: {e}")
            return index, []

async def run_llm_with_parquet(
    parquet_file,
    repo_id,
    config_name,
    split_name,
    model_name,
    temperature,
    split_type,
    top_p,
    top_k,
    max_tokens,
    window_size,
    overlap_sentences,
    batch_processing,
    limit=None  
):    
    if repo_id and config_name:
        logger.info("Loading dataset from Hugging Face.")
        grouped_data, window_size_used, overlap_used = process_text_from_hf_parquet(
            repo_id=repo_id,
            config_name=config_name,
            split_name=split_name,
            split_type=split_type,
            window_size=window_size,
            overlap_sentences=overlap_sentences,
            batch_processing=batch_processing,
            limit=limit  
        )
    elif parquet_file:
        logger.info("Loading dataset from local Parquet file.")
        grouped_data, window_size_used, overlap_used = process_text_from_parquet(
            parquet_file,
            split_type,
            window_size,
            overlap_sentences,
            batch_processing,
            limit=limit  #
        )
    else:
        logger.error("Invalid parameters for loading dataset.")
        raise ValueError("Invalid parameters for loading dataset.")


    if limit is not None:

        limited_grouped_data = {}
        doc_keys = list(grouped_data.keys())[:limit]
        for k in doc_keys:
            limited_grouped_data[k] = grouped_data[k]
        grouped_data = limited_grouped_data
     
    start_time = datetime.now()
    logger.info(f"Starting processing at {start_time}")
    logger.info(f"Initializing LLM: {model_name}")

    if "claude" in model_name:
        llm = ChatBedrock(
            model_id=model_name,
            verbose=False,
            region_name="eu-central-1",
            model_kwargs=dict(
                temperature=temperature, 
                max_tokens=max_tokens, 
                top_k=top_k, 
                top_p=top_p
            )
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            verbose=False,
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens
        )

    extracted_softwares_total = []

    for document_id, chunk_list in grouped_data.items():
        extracted_softwares = []
        tasks = [process_chunk_with_llm_async(llm, chunk, i) for i, chunk in enumerate(chunk_list)]
        results = await asyncio.gather(*tasks)
        for _, result in sorted(results):
            extracted_softwares.extend(result)

        extracted_softwares_total.append({
            "document_id": document_id,
            "softwares": extracted_softwares
        })
        logger.info(f"Document {document_id} processed. Model found {len(extracted_softwares)} possible entities before cleaning.")

    end_time = datetime.now()
    logger.info(f"Finished processing all documents. Total time taken: {end_time - start_time}")
    
    return extracted_softwares_total, start_time, end_time, "PARQUET", top_p, top_k, max_tokens, split_type, window_size_used, overlap_used