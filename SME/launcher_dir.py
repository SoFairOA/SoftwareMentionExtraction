import argparse
from run_llm_SaT import run_llm_with_parquet as run_llm_SaT
from run_llm_rag import run_llm_with_parquet as run_llm_rag
from output_generator_dir import generate_output
import asyncio
from datetime import datetime
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser(description="Run LLM processing on a Parquet file or a Hugging Face dataset with optional caching.")
    parser.add_argument("--pipeline", type=str, choices=["SaT", "RAG"], required=True, help="Choose the pipeline to run: 'SaT' or 'RAG'.")
    parser.add_argument("--parquet_file", type=str, help="Path to the local Parquet file.")
    parser.add_argument("--repo_id", type=str, help="Repository ID on Hugging Face (e.g., 'SoFairOA/softcite_dataset').")
    parser.add_argument("--config_name", type=str, help="Configuration name on Hugging Face (e.g., 'documents').")
    parser.add_argument("--split_name", type=str, default="test", help="Name of the split to load (e.g., 'train', 'test')")
    parser.add_argument("--limit", type=int, help="Limit the number of documents to process.")
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

    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        logger.info("HUGGINGFACE_HUB_TOKEN is set.")
    else:
        logger.warning("HUGGINGFACE_HUB_TOKEN is not set. Make sure you are logged in with `huggingface-cli login` if needed.")

    # Validazione degli argomenti: o parquet_file o repo_id e config_name
    if not args.parquet_file and not (args.repo_id and args.config_name):
        raise ValueError("You must specify either --parquet_file for a local file or both --repo_id and --config_name for a Hugging Face dataset.")
    if args.parquet_file and (args.repo_id or args.config_name):
        raise ValueError("Specify only one of --parquet_file or the dataset parameters, not both.")

    # Seleziona la pipeline
    if args.pipeline == "SaT":
        run_llm_function = run_llm_SaT
    elif args.pipeline == "RAG":
        run_llm_function = run_llm_rag
    else:
        raise ValueError("Invalid pipeline selected.")

    results, start_time, end_time, document_type, top_p, top_k, max_tokens, split_type, window_size, overlap_sentences = asyncio.run(
        run_llm_function(
            parquet_file=args.parquet_file,
            repo_id=args.repo_id,
            config_name=args.config_name,
            split_name=args.split_name,
            model_name=args.model,
            temperature=args.temperature,
            split_type=args.split_type,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            window_size=args.window_size,
            overlap_sentences=args.overlap_sentences,
            batch_processing=args.batch_processing,
            limit=args.limit  # Passa il limite
        )
    )

    generate_output(results, args.output_dir, start_time, end_time, args)

if __name__ == "__main__":
    main()
