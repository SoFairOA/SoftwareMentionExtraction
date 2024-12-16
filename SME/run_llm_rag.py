import os
import re
import logging
from datetime import datetime
import asyncio
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json
from utils import semaphore, KEYWORD_PATTERNS, keyword_based_filtering, semantic_similarity_filter_torch, segment_text_with_overlap, process_text_from_parquet, process_text_from_hf_parquet, extract_json_from_response, get_chunk_embedding, get_query_embedding, convert_keywords_to_patterns, normalize_keyword, programming_language_description, url_description, software_description, url_sentences, software_sentences, programming_language_sentences, save_extracted_entities, predefined_queries, query_embedding_cache, execute_with_retries
from jinja2 import Environment, FileSystemLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = Environment(loader=FileSystemLoader("templates"))


async def verify_extracted_entities(llm, extracted_entities, batch_size=5):
    verified_entities = []
    # system_verify_template = env.get_template("system_verify.j2")
    # human_verify_template = env.get_template("human_verify.j2")

    for i in range(0, len(extracted_entities), batch_size):
        batch = extracted_entities[i : i + batch_size]
        entities_text = json.dumps(batch, indent=4)
        # system_message = SystemMessage(content=system_verify_template.render())
        # human_message = HumanMessage(content=human_verify_template.render(entities_text=entities_text))

        system_message = SystemMessage(content=
            """You are Professor John, a highly skilled expert in software identification.\n
            Your role is to verify and correct software entities provided in JSON format.\n 
            You must determine whether the entity is a valid software. If it is not valid, you must delete it and return an empty JSON array: `[]`.\n
            Provide only JSON output without any explanations.""")
            
        human_message = HumanMessage(content=f"""
            Hello Professor John! Please verify the correctness of the following extracted software entity. Follow these instructions carefully:\n
            **Read carefully the entity name and then**:\n
            1. If the entity is not a software, respond with an empty JSON array: `[]`.\n
            2. If the entity is a valid software, return it in the same JSON format.\n
            3. Provide no additional text or explanations; output only valid JSON.\n

            **Entity to verify**:\n
            {entities_text}
            """)

        async with semaphore:
            try:
                response = await execute_with_retries(
                    llm.abatch, inputs=[[system_message, human_message]]
                )
                raw_content = response[0].content
                logger.info(f"Verification response: {raw_content}")

                verified_batch = extract_json_from_response(raw_content)
                if verified_batch is not None:
                    for entity, original_entity in zip(verified_batch, batch):
                        entity["document_id"] = original_entity["document_id"]
                    verified_entities.extend(verified_batch)
                else:
                    logger.info("All entities were removed during verification.")
            except Exception as e:
                logger.error(f"Error during entity verification: {e}")
                continue

    return verified_entities




async def process_chunk_with_llm_async(llm, chunk, index, limited_examples, document_id):
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
        #     examples_list=limited_examples
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
        
        - **Examples Section**:
        {limited_examples}\n
        
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
            "version": ["<software_version>"],             
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
                for entity in extracted_json:
                    entity["document_id"] = document_id
                return index, extracted_json
            else:
                logging.error(f"JSON decoding error in chunk {index}: No valid JSON found.")
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
    extracted_softwares_total = []
    
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
            limit=limit  
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
                top_p=top_p,
            ),
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            verbose=False,
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    seen = set()
    prev_count = len(seen)
    extracted_software_examples = []  
    max_iterations = 10 
    new_keywords = set()  
    new_queries = set()
    for iteration in range(max_iterations):
        if iteration == 0:
            compiled_combined_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in KEYWORD_PATTERNS]
            logger.info(f"This is iteration 1 and the keywords are: {KEYWORD_PATTERNS}")
            logger.info(f"This is iteration 1 and the queries are: {predefined_queries}")
        else:
            compiled_combined_patterns = convert_keywords_to_patterns(new_keywords)
            logger.info(f"This is iteration {iteration + 1} and the keywords are: {new_keywords}")
            logger.info(f"This is iteration {iteration + 1} and the queries are: {new_queries}")
        logger.info(f"Starting iteration {iteration + 1}")

        iteration_results = []


        MAX_EXAMPLES = 5
        if extracted_software_examples:
            limited_examples = extracted_software_examples[-MAX_EXAMPLES:]
        else:
            limited_examples = []                
        # MAX_EXAMPLES = 5  # Limit the number of examples to avoid overly long prompts
        # if extracted_software_examples:
        #     limited_examples = extracted_software_examples[-MAX_EXAMPLES:]  # Take the last MAX_EXAMPLES
        #     examples_text = "\n".join(f"- {software}" for software in limited_examples)
        #     examples_section = f"\n\nExamples of software extracted during previous iterations:\n{examples_text}\n"
        # else:
        #     examples_section = ""
        for document_id, chunk_list in grouped_data.items():
            processed_chunks = set()
            new_chunks = [chunk for chunk in chunk_list if chunk not in processed_chunks]
            logger.info(f"Processing document {document_id} in iteration {iteration + 1}")

            # Combine the patterns
            # compiled_combined_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in KEYWORD_PATTERNS]

            # Step 1: Keyword-based filtering
            filtered_chunks = keyword_based_filtering(new_chunks, compiled_combined_patterns)
            logger.info(f"Number of chunks after keyword-based filtering: {len(filtered_chunks)}")

            if not filtered_chunks:
                logger.warning(f"No chunks found after keyword filtering for document {document_id}.")
                continue

            # Step 2: Semantic similarity filtering
            if iteration == 0:
                relevant_chunks = semantic_similarity_filter_torch(filtered_chunks, predefined_queries)
                logger.info(f"Number of chunks after semantic similarity filtering: {len(relevant_chunks)}")
            else:
                threshold = 0.3 + 0.1 * iteration
                relevant_chunks = semantic_similarity_filter_torch(filtered_chunks, list(new_queries), threshold=threshold)
                logger.info(f"Number of chunks after semantic similarity filtering: {len(relevant_chunks)}")
            
            if not relevant_chunks:
                logger.warning(f"No relevant chunks found after semantic similarity filtering for document {document_id}.")
                continue

            tasks = [
                process_chunk_with_llm_async(
                    llm,
                    chunk,
                    i,
                    limited_examples,
                    document_id,
                )
                for i, chunk in enumerate(relevant_chunks)
            ]
            results = await asyncio.gather(*tasks)
            processed_chunks.update(new_chunks)

            # Step 3: Collect extracted entities avoiding duplicates
            extracted_entities = []
            for _, result in sorted(results):
                unique_entities = save_extracted_entities(result, extracted_entities)
                # Ensure that each entity has the correct document_id
                for entity in unique_entities:
                    entity["document_id"] = document_id
                extracted_entities.extend(unique_entities)

            # Verify the extracted entities for this document
            if extracted_entities:
                verified_entities = await verify_extracted_entities(
                    llm, extracted_entities, batch_size=5
                )
            else:
                verified_entities = []

            # Update examples for the prompt with verified entities
            extracted_software_examples.extend(
                [entity["software"] for entity in verified_entities if "software" in entity and entity["software"]]
            )

            # Update iteration_results
            iteration_results.append(
                {
                    "document_id": document_id,
                    "softwares": verified_entities,
                }
            )
            
            # Update new_keywords with names of verified software
            for entity in verified_entities:
                if "software" in entity and entity["software"]:
                    software = entity["software"]
                    if isinstance(software, list):
                        for s in software:
                            if s not in seen:
                                # predefined_queries.append(s)
                                new_queries.add(s)
                                new_keywords.add(s)  
                                seen.add(s)
                                logger.info(f"seen: {seen}")
                                logger.info("Added new keyword and query: %s", s)
                            else:
                                pass
                    elif isinstance(software, str):
                        if software not in seen:
                            # predefined_queries.append(software)
                            new_queries.add(software)
                            new_keywords.add(software)  
                            seen.add(software)
                            logger.info(f"seen: {seen}")
                            logger.info("Added new keyword and query: %s", software)
                        else:
                            pass
            # After updating keywords and queries, clear the query embedding cache
            # query_embedding_cache.clear()
            # logger.info("Cleared query embedding cache after processing document %s", document_id)


        # Add the results of the current iteration
        extracted_softwares_total.extend(iteration_results)

        # Break if no new keywords are found
        current_count = len(seen)
        logger.info(f"prev count = {prev_count}")
        logger.info(f"current count = {current_count}")
        if current_count == prev_count:
            logger.info("No new keywords found in this iteration, stopping.")
            break
        prev_count = current_count


    # After all iterations
    if not extracted_softwares_total:
        logger.warning("No results were extracted. Check the input data or model behavior.")

    # Ensure all documents are included in extracted_softwares_total
    all_document_ids = set(grouped_data.keys())
    documents_with_results = set(result["document_id"] for result in extracted_softwares_total)
    documents_without_results = all_document_ids - documents_with_results

    for document_id in documents_without_results:
        extracted_softwares_total.append(
            {
                "document_id": document_id,
                "softwares": [],  
            }
        )

    end_time = datetime.now()
    logger.info(f"Finished processing all documents. Total time taken: {end_time - start_time}")

    return (
        extracted_softwares_total,
        start_time,
        end_time,
        "PARQUET",
        top_p,
        top_k,
        max_tokens,
        split_type,
        window_size_used,
        overlap_used,
    )