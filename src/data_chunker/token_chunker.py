import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from logging import getLogger
import gc
from shared_utils import * 

logger = getLogger(__name__)

def chunk_text(text: str, chunk_length: int, overlap: int, min_length: int = 50) -> List[str]:
    """
    Splits the input text into chunks of specified length with overlap and minimum length.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_length
        chunk = text[start:end]
        if len(chunk) >= min_length:
            chunks.append(chunk)
        start += (chunk_length - overlap)
    return chunks

def get_overlap(chunk_length: int) -> int:
    """
    Returns the appropriate overlap based on chunk length.
    """
    if chunk_length in [128, 256]:
        return 50
    else:
        return 200

def process_data_tokens(df: pd.DataFrame, chunk_lengths: List[int], embed_model: Any, embed_model_name: str, base_path: str) -> List[Tuple[int, int, int, float, float]]:
    chunk_stats = []

    for chunk_length in chunk_lengths:
        logger.info(f"Processing character-based chunks with length: {chunk_length}")
        
        # Calculate overlap
        overlap = get_overlap(chunk_length)
        logger.info(f"Using overlap of {overlap} for chunk length {chunk_length}")
        
        # Process all rows at once
        df['chunks'] = df['text'].apply(lambda x: chunk_text(x, chunk_length, overlap) if isinstance(x, str) else [])
        
        # Explode the DataFrame
        exploded_df = df.explode('chunks').reset_index(drop=True)
        exploded_df['len'] = exploded_df['chunks'].str.len()
        exploded_df = exploded_df[exploded_df['len'] >= 50]  # Remove chunks shorter than 50 characters
        
        # Generate unique IDs
        exploded_df['unique_id'] = exploded_df.groupby('id').cumcount().add(1).astype(str)
        exploded_df['unique_id'] = exploded_df['id'] + '_' + exploded_df['unique_id']
        
        # Prepare the final DataFrame
        chunked_df = pd.DataFrame({
            'unique_id': exploded_df['unique_id'],
            'url': exploded_df['url'],
            'last_updated': exploded_df['last_updated'],
            'html_content': exploded_df['html_content'],
            'text': exploded_df['chunks'],
            'len': exploded_df['len'],
            'general_id': exploded_df['id']
        })
        
        logger.info(f"Created {len(chunked_df)} chunks")
        
        # Compute statistics
        min_length = chunked_df['len'].min()
        max_length = chunked_df['len'].max()
        mean_length = chunked_df['len'].mean()
        median_length = chunked_df['len'].median()
        chunk_stats.append((chunk_length, min_length, max_length, mean_length, median_length))
        
        # Print length statistics
        logger.info(f"Chunk length statistics for target length {chunk_length}:")
        logger.info(f"  Minimum length: {min_length}")
        logger.info(f"  Maximum length: {max_length}")
        logger.info(f"  Mean length: {mean_length:.2f}")
        logger.info(f"  Median length: {median_length:.2f}")
        
        logger.info("Embedding chunked texts")
        embedded_df = embed_dataframe(chunked_df, embed_model)
        
        logger.info("Generating document dictionaries")
        documents = generate_documents(embedded_df, chunk_length, 'token')
        
        logger.info("Saving documents to JSON")
        save_documents_to_json(documents, chunk_length, embed_model_name, 'token', base_path)
        
        # Clear memory
        del chunked_df, embedded_df, documents
        gc.collect()
        
        logger.info(f"Finished processing chunks of length {chunk_length}.")
    
    return chunk_stats
