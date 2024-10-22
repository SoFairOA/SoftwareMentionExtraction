import argparse
import os
import asyncio
from run_llm_SaT import run_llm_with_parquet_bedrock
from output_generator_dir import generate_output

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the LLM to extract software mentions from Parquet files.")
    parser.add_argument("--model", type=str,
                        help="The name of the LLM model to use.")
    parser.add_argument("--parquet_file", type=str, required=True, 
                        help="Path to the Parquet file to process.") 
    parser.add_argument("--temperature", type=float, default=0.06,
                        help="Temperature setting for the LLM model (default: 0.06).")
    parser.add_argument("--split_type", type=str, choices=["sentence", "paragraph", "complete"], default="paragraph",
                        help="Specify how to split the text: by sentences ('sentence'), paragraphs ('paragraph'), or as a whole ('complete').")    
    parser.add_argument("--prompt_type", type=str, choices=["generic", "detailed", "in-context"], default="generic",
                        help="The type of prompt to use (default: generic).")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to the output directory where the result files will be saved.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p parameter for the LLM model (default: 0.9).")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k parameter for the LLM model (default: 50).")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate (default: 1024).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    #-----CREATE THE FOLDER IF IT DOESN'T EXIST-----#
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Directory di output {args.output_dir} creata.")
    
    #-----EXECUTE THE LLM WITH SPECIFIED PARAMETERS-----#
    results, start_time, end_time, prompt_used, document_type, top_p, top_k, max_tokens, window_size_used, overlap_used = asyncio.run(
        run_llm_with_parquet_bedrock(
            parquet_file=args.parquet_file,
            model_name=args.model,
            temperature=args.temperature,
            split_type=args.split_type,
            prompt_type=args.prompt_type,
            top_p=args.top_p, 
            top_k=args.top_k, 
            max_tokens=args.max_tokens
        )
    )

    #-----PROCESS RESULTS-----#
    for result in results:
        document_id = result["document_id"]
        extracted_softwares = result["softwares"]

        #-----ITERATE ON THE SOFTWARE ENTITIES EXTRACTED FROM TEXTS TO SAVE THEM-----#
        software_entries = [software for software in extracted_softwares]
        
        #-----GENERATION OF THE OUTPUT IN CSV FORMAT WITH INFORMATION OF THE SPECIFIC RUN-----#
        generate_output(
            software_entries, 
            start_time, 
            end_time, 
            prompt_used, 
            document_type, 
            args, 
            args.output_dir, 
            top_p, 
            top_k, 
            max_tokens, 
            window_size_used, 
            overlap_used,
            document_id
        )

if __name__ == "__main__":
    main()
