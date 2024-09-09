import os
import sys
import pandas as pd
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Tuple

# Update imports
from sentence_chunker import process_data_sentences
from character_chunker import process_data_tokens
from semantic_chunker import *
from shared_utils import save_documents_to_json
from token_chunker import process_data_tokens_tiktoken
from recursive_chunker import process_data_recursive_langchain

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the assets directory (two levels up from the script)
assets_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'assets'))

# Add the assets directory to the Python path
sys.path.insert(0, assets_dir)

def log_chunk_stats(chunk_type: str, stats: List[Tuple[int, int, int, float, float]]):
    logger.info(f"{chunk_type} chunking statistics:")
    for stat in stats:
        chunk_size, min_length, max_length, mean_length, median_length = stat
        logger.info(f"  Chunk size {chunk_size}:")
        logger.info(f"    Min length = {min_length}")
        logger.info(f"    Max length = {max_length}")
        logger.info(f"    Mean length = {mean_length:.2f}")
        logger.info(f"    Median length = {median_length:.2f}")

from shared_utils import get_embeddings, embed_dataframe
#options for encode model = "all-MiniLM-L6-v2" or "hf"
def main(chunk_types: List[str] = ["recursive", "sentence", "char", "token", "semantic"], encode_model: str = "all-MiniLM-L6-v2", embed_model_name: str = 'all-mini'):
    try:
        load_dotenv()
        
        df_path = os.path.join(assets_dir, 'csv', 'data_subset.csv')
        chunk_sizes = [128, 256, 512]  # You can adjust this as needed
        if embed_model_name == None:
            embed_model_name = embed_model

        docs_path = os.path.join(assets_dir, 'docs')
        
        if not os.path.exists(df_path):
            logger.error(f"CSV file not found: {df_path}")
            return

        logger.info(f"Loading DataFrame from {df_path}")
        df = pd.read_csv(df_path)
        
        # Drop rows with NA values in the 'text' column
        initial_row_count = len(df)
        df = df.dropna(subset=['text'])
        rows_dropped = initial_row_count - len(df)
        logger.info(f"Dropped {rows_dropped} rows with NA values in the 'text' column")
        logger.info(f"Proceeding with {len(df)} rows for chunking")

        if encode_model == "hf":
            logger.info("Initializing HuggingFaceEmbeddings model")
            embed_model = HuggingFaceEmbeddings()
        else:
            logger.info(f"Initializing SentenceTransformer model: {encode_model}")
            embed_model = SentenceTransformer(f'sentence-transformers/{encode_model}')
        
        # Embed the dataframe
        df_embedded = embed_dataframe(df, embed_model)
        
        # Process chunks based on the specified types
        for chunk_type in chunk_types:
            if chunk_type == "sentence":
                logger.info("Processing sentence-based chunks")
                sentence_stats = process_data_sentences(df_embedded, chunk_sizes, embed_model, embed_model_name, docs_path)
                log_chunk_stats("Sentence-based", sentence_stats)
            elif chunk_type == "char":
                logger.info("Processing character-based chunks")
                char_stats = process_data_tokens(df_embedded, chunk_sizes, embed_model, embed_model_name, docs_path)
                log_chunk_stats("Character-based", char_stats)
            elif chunk_type == "semantic":
                logger.info("Processing semantic chunks using TextTiling")
                semantic_stats = process_data_semantic(df_embedded, chunk_sizes, embed_model, embed_model_name, docs_path)
                log_chunk_stats("Semantic (TextTiling)", semantic_stats)
            elif chunk_type == "recursive":
                logger.info("Processing recursive chunks using LangChain")
                recursive_stats = process_data_recursive_langchain(df_embedded, chunk_sizes, embed_model, embed_model_name, docs_path)
                log_chunk_stats("Recursive (LangChain)", recursive_stats)
            elif chunk_type == "token":
                logger.info("Processing token-based chunks using tiktoken")
                token_stats = process_data_tokens_tiktoken(df_embedded, chunk_sizes, embed_model, embed_model_name, docs_path)
                log_chunk_stats("Token-based (tiktoken)", token_stats)
        
        logger.info("Chunk creation process completed successfully")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()