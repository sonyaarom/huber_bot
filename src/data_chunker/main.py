import os
import sys
import pandas as pd
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Tuple

# Update imports
from sentence_chunker import process_data_sentences
from token_chunker import process_data_tokens
from sentence_chunker import process_data_sentences
from token_chunker import process_data_tokens
from semantic_chunker import process_data_semantic_texttiling
from shared_utils import save_documents_to_json
import sys
import pandas as pd
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Tuple


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the assets directory (two levels up from the script)
assets_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'assets'))

# Add the assets directory to the Python path
sys.path.insert(0, assets_dir)

def log_chunk_stats(chunk_type: str, stats: List[Tuple[int, int, int]]):
    logger.info(f"{chunk_type} chunking statistics:")
    for chunk_size, min_length, max_length in stats:
        logger.info(f"  Chunk size {chunk_size}: Min length = {min_length}, Max length = {max_length}")

def main(chunk_types: List[str] = ["semantic", "sentence", "token"]):
    try:
        load_dotenv()
        
        df_path = os.path.join(assets_dir, 'csv', 'data_subset.csv')
        chunk_sizes = [128, 256, 512, 1024]  # You can adjust this as needed
        chunk_overlap = 200
        embed_model_name = "hf"
        docs_path = os.path.join(assets_dir, 'docs')
        
        if not os.path.exists(df_path):
            logger.error(f"CSV file not found: {df_path}")
            return

        logger.info(f"Loading DataFrame from {df_path}")
        df = pd.read_csv(df_path)
        
        logger.info("Initializing HuggingFaceEmbeddings model")
        embed_model = HuggingFaceEmbeddings()
        
        if "sentence" in chunk_types:
            logger.info("Processing sentence-based chunks")
            sentence_stats = process_data_sentences(df, chunk_sizes, chunk_overlap, embed_model, embed_model_name, docs_path)
            log_chunk_stats("Sentence-based", sentence_stats)
        
        if "token" in chunk_types:
            logger.info("Processing token-based chunks")
            token_stats = process_data_tokens(df, chunk_sizes, embed_model, embed_model_name, docs_path)
            log_chunk_stats("Token-based", token_stats)
        
        if "semantic" in chunk_types:
            logger.info("Processing semantic chunks using TextTiling")
            semantic_stats = process_data_semantic_texttiling(df, chunk_sizes[0], embed_model, embed_model_name, docs_path)
            log_chunk_stats("Semantic (TextTiling)", semantic_stats)
        
        logger.info("Chunk creation process completed successfully")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()

#CHANGE NAMING TO KOW THE EMBEDDING MODEL