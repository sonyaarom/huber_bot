import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.texttiling import TextTilingTokenizer
import nltk
import logging
from typing import Tuple, List, Any, Generator
from shared_utils import embed_dataframe, generate_documents, save_documents_to_json

nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

def get_overlap(chunk_size: int) -> int:
    """
    Returns the appropriate overlap based on chunk size.
    """
    if chunk_size <= 256:
        return 50
    else:
        return 200

def adaptive_semantic_chunk(text: str, min_chunk_size: int = 100, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Adaptively chunks text using TextTiling with fallback mechanisms.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []

    tt = TextTilingTokenizer(w=20, k=10)
    
    try:
        segments = tt.tokenize(text)
    except ValueError as e:
        logger.warning(f"TextTiling failed: {str(e)}. Falling back to sentence tokenization.")
        segments = [' '.join(sent) for sent in [sent_tokenize(text)[i:i+5] for i in range(0, len(sent_tokenize(text)), 5)]]
    except Exception as e:
        logger.error(f"Unexpected error in TextTiling: {str(e)}. Falling back to simple chunking.")
        words = word_tokenize(text)
        segments = [' '.join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size-overlap)]
        return segments

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
                overlap_words = current_segment[-overlap:] if overlap > 0 else []
                current_segment = overlap_words + segment_words
                current_size = len(current_segment)
            else:
                while segment_words:
                    space_left = max_chunk_size - current_size
                    if space_left >= min_chunk_size or not current_segment:
                        current_segment.extend(segment_words[:space_left])
                        segment_words = segment_words[space_left:]
                        refined_segments.append(' '.join(current_segment))
                        overlap_words = current_segment[-overlap:] if overlap > 0 else []
                        current_segment = overlap_words
                        current_size = len(current_segment)
                    else:
                        refined_segments.append(' '.join(current_segment))
                        overlap_words = current_segment[-overlap:] if overlap > 0 else []
                        current_segment = overlap_words + segment_words[:max_chunk_size]
                        segment_words = segment_words[max_chunk_size:]
                        current_size = len(current_segment)
    
    if current_segment:
        refined_segments.append(' '.join(current_segment))

    return refined_segments

def chunk_dataframe(data: pd.DataFrame, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
    logger.info(f"Starting to chunk dataframe with chunk size: {chunk_size}")
    
    initial_row_count = len(data)
    data = data.dropna(subset=['text'])
    rows_dropped = initial_row_count - len(data)
    logger.info(f"Dropped {rows_dropped} rows with NA values in the 'text' column")

    overlap = get_overlap(chunk_size)
    logger.info(f"Using overlap of {overlap} words for chunk size {chunk_size}")

    batch_size = 1000  # Process in batches to reduce memory usage
    for start in range(0, len(data), batch_size):
        chunked_rows = []
        end = min(start + batch_size, len(data))
        
        for _, row in data.iloc[start:end].iterrows():
            try:
                chunks = adaptive_semantic_chunk(row['text'], min_chunk_size=chunk_size//2, max_chunk_size=chunk_size, overlap=overlap)
                for i, chunk in enumerate(chunks):
                    chunk_length = len(word_tokenize(chunk))
                    chunked_rows.append({
                        'unique_id': f"{row['id']}_{i+1}",
                        'url': row.get('url', ''),
                        'last_updated': row.get('last_updated', ''),
                        'html_content': row.get('html_content', ''),
                        'text': chunk,
                        'len': chunk_length,
                        'general_id': row['id']
                    })
            except Exception as e:
                logger.error(f"Error processing row {row['id']}: {str(e)}")

        result = pd.DataFrame(chunked_rows)
        logger.info(f"Processed batch. Created {len(result)} chunks from {end-start} rows.")
        yield result

def process_data_semantic(df: pd.DataFrame, chunk_sizes, embed_model: Any, embed_model_name: str, base_path: str) -> List[Tuple[int, int, int, float, float]]:
    chunk_stats = []
    for chunk_size in chunk_sizes:
        logger.info(f"Processing semantic chunks with target size: {chunk_size}")
        
        all_chunks = pd.DataFrame()
        for chunked_batch in chunk_dataframe(df, chunk_size):
            all_chunks = pd.concat([all_chunks, chunked_batch], ignore_index=True)
        
        if all_chunks.empty:
            logger.error("No chunks were created. Semantic splitting failed.")
            chunk_stats.append((chunk_size, 0, 0, 0, 0))
            continue
        
        chunk_lengths = all_chunks['len'].tolist()
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        mean_length = np.mean(chunk_lengths)
        median_length = np.median(chunk_lengths)
        
        logger.info(f"Chunking complete. Created {len(all_chunks)} chunks.")
        logger.info(f"Chunk length statistics: Min = {min_length}, Max = {max_length}, Mean = {mean_length:.2f}, Median = {median_length:.2f}")
        
        logger.info("Embedding chunked texts")
        embedded_df = embed_dataframe(all_chunks, embed_model)
        
        logger.info("Generating document dictionaries")
        documents = generate_documents(embedded_df, chunk_size, 'semantic')
        
        logger.info("Saving documents to JSON")
        save_documents_to_json(docs=documents, chunk_size=chunk_size, doc_type='semantic', base_path=base_path, model_name=embed_model_name)
        
        chunk_stats.append((chunk_size, min_length, max_length, mean_length, median_length))
    
    return chunk_stats

