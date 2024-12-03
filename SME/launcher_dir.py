import argparse
from run_llm_SaT import run_llm_with_parquet as run_llm_SaT
from run_llm_rag import run_llm_with_parquet as run_llm_rag
from output_generator_dir import generate_output
import asyncio
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run LLM processing on a Parquet file with optional caching.")
    parser.add_argument("--pipeline", type=str, choices=["SaT", "RAG"], required=True, help="Choose the pipeline to run: 'SaT' for the old pipeline or 'RAG' for the new pipeline.")
    parser.add_argument("--parquet_file", type=str, required=True, help="Path to the Parquet file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output results.")
    parser.add_argument("--model", type=str, required=True, help="Model name for processing.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p sampling.")
    parser.add_argument("--top_k", type=int, default=10, help="Top k sampling.")
    parser.add_argument("--split_type", type=str, choices=["sentence", "paragraph", "complete"], default="paragraph", help="Split type.")
    parser.add_argument("--window_size", type=int, default=20, help="Number of sentences per chunk.")
    parser.add_argument("--overlap_sentences", type=int, default=2, help="Number of overlapping sentences.")
    parser.add_argument("--batch_processing", action="store_true", help="Process documents in batches if set.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="The maximum number of tokens the model can generate")
    args = parser.parse_args()
    start_time = datetime.now()
    
    if args.pipeline == "SaT":
        run_llm_function = run_llm_SaT
    elif args.pipeline == "RAG":
        run_llm_function = run_llm_rag
    else:
        raise ValueError("Invalid pipeline selected. Choose either 'SaT' or 'RAG'.")
    
    results, start_time, end_time, document_type, top_p, top_k, max_tokens, split_type, window_size, overlap_sentences = asyncio.run(
        run_llm_function(
            args.parquet_file, args.model, args.temperature, args.split_type, args.top_p,
            args.top_k, args.max_tokens, window_size=args.window_size, overlap_sentences=args.overlap_sentences,
            batch_processing=args.batch_processing
        )
    )
    
    generate_output(results, args.output_dir, start_time, end_time, args)

if __name__ == "__main__":
    main()
