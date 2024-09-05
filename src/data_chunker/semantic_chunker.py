import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.texttiling import TextTilingTokenizer
import nltk
import logging
from typing import Tuple
from shared_utils import *

nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

def semantic_split_texttiling(df: pd.DataFrame, min_chunk_size: int = 100, max_chunk_size: int = 500) -> Tuple[pd.DataFrame, int, int]:
    """
    Perform semantic splitting on a dataframe containing text from URLs using TextTiling.
    
    Args:
    - df: Input DataFrame with at least 'id', 'url', 'text' columns
    - min_chunk_size: Minimum size for each chunk (in words)
    - max_chunk_size: Maximum size for each chunk (in words)
    
    Returns:
    - A tuple containing:
      1. A new DataFrame with semantically split text
      2. Minimum chunk length
      3. Maximum chunk length
    """
    tt = TextTilingTokenizer(w=20, k=10)
    
    chunked_rows = []
    min_length = float('inf')
    max_length = 0
    
    for _, row in df.iterrows():
        text = row['text']
        
        if text is None or not isinstance(text, str) or not text.strip():
            logger.warning(f"Invalid or empty text for document {row['id']}. Skipping this document.")
            continue
        
        try:
            segments = tt.tokenize(text)
        except ValueError as e:
            logger.warning(f"TextTiling failed for document {row['id']}: {str(e)}. Falling back to sentence tokenization.")
            segments = [' '.join(sent) for sent in [sent_tokenize(text)[i:i+5] for i in range(0, len(sent_tokenize(text)), 5)]]
        except Exception as e:
            logger.error(f"Unexpected error in TextTiling for document {row['id']}: {str(e)}. Skipping this document.")
            continue
        
        refined_segments = []
        current_segment = []
        current_size = 0
        
        for segment in segments:
            segment_words = word_tokenize(segment)
            segment_size = len(segment_words)
            
            if current_size + segment_size <= max_chunk_size:
                current_segment.extend(segment_words)
                current_size += segment_size
            else:
                if current_size >= min_chunk_size:
                    refined_segments.append(' '.join(current_segment))
                    current_segment = segment_words
                    current_size = segment_size
                else:
                    while len(segment_words) > 0:
                        space_left = max_chunk_size - current_size
                        if space_left >= min_chunk_size or len(current_segment) == 0:
                            current_segment.extend(segment_words[:space_left])
                            segment_words = segment_words[space_left:]
                            refined_segments.append(' '.join(current_segment))
                            current_segment = []
                            current_size = 0
                        else:
                            refined_segments.append(' '.join(current_segment))
                            current_segment = segment_words[:max_chunk_size]
                            segment_words = segment_words[max_chunk_size:]
                            current_size = len(current_segment)
        
        if current_segment:
            refined_segments.append(' '.join(current_segment))
        
        for i, chunk in enumerate(refined_segments):
            chunk_length = len(word_tokenize(chunk))
            min_length = min(min_length, chunk_length)
            max_length = max(max_length, chunk_length)
            chunked_rows.append({
                'unique_id': f"{row['id']}_{i+1}",
                'url': row['url'],
                'text': chunk,
                'len': chunk_length,
                'original_id': row['id']
            })
    
    chunked_df = pd.DataFrame(chunked_rows)
    logger.info(f"Created {len(chunked_df)} semantic chunks from {len(df)} original rows.")
    logger.info(f"Min chunk length: {min_length}, Max chunk length: {max_length}")
    
    return chunked_df, min_length, max_length

from typing import Any, List, Tuple

def process_data_semantic_texttiling(df: pd.DataFrame, chunk_size: int, embed_model: Any, embed_model_name: str, base_path: str) -> List[Tuple[int, int, int]]:
    logger.info(f"Processing semantic chunks with TextTiling (target size: {chunk_size})")
    
    chunked_df, min_length, max_length = semantic_split_texttiling(df, min_chunk_size=chunk_size//2, max_chunk_size=chunk_size)
    
    if chunked_df.empty:
        logger.error("No chunks were created. Semantic splitting failed.")
        return [(chunk_size, 0, 0)]
    
    logger.info(f"Chunking complete. Created {len(chunked_df)} chunks.")
    logger.info(f"Chunk length statistics: Min = {min_length}, Max = {max_length}")
    
    logger.info("Embedding chunked texts")
    embedded_df = embed_dataframe(chunked_df, embed_model)
    
    logger.info("Generating document dictionaries")
    documents = generate_documents(embedded_df, chunk_size, 'semantic')
    
    logger.info("Saving documents to JSON")
    save_documents_to_json(docs=documents, chunk_size=chunk_size, doc_type='semantic', base_path=base_path, model_name=embed_model_name)
    
    
    return [(chunk_size, min_length, max_length)]