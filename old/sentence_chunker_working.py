# sentence_chunker.py
from shared_utils import *
from llama_index.core.node_parser import SentenceSplitter
import re

def get_overlap(chunk_size: int) -> int:
    """
    Returns the appropriate overlap based on chunk size.
    """
    if chunk_size <= 256:
        return 50
    else:
        return 200

def chunk_text_by_sentences(text, chunk_size=2000, chunk_overlap=200, min_chunk_size=50):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    
    # Process chunks to ensure they meet size requirements
    processed_chunks = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) <= chunk_size:
            current_chunk += chunk + " "
        else:
            if len(current_chunk) >= min_chunk_size:
                processed_chunks.append(current_chunk.strip())
            current_chunk = chunk + " "
        
        # If current_chunk exceeds chunk_size, split it
        while len(current_chunk) > chunk_size:
            split_point = chunk_size
            # Try to split at a sentence boundary
            sentence_end = re.search(r'[.!?]\s+', current_chunk[:chunk_size][::-1])
            if sentence_end:
                split_point = chunk_size - sentence_end.start()
            processed_chunks.append(current_chunk[:split_point].strip())
            current_chunk = current_chunk[split_point:].strip()
    
    # Add any remaining text as a chunk
    if len(current_chunk) >= min_chunk_size:
        processed_chunks.append(current_chunk.strip())
    
    logger.debug(f"Created {len(processed_chunks)} sentence-based chunks.")
    return processed_chunks

def chunk_dataframe(data: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> Tuple[pd.DataFrame, int, int, float, float]:
    logger.info(f"Starting to chunk dataframe with chunk size: {chunk_size} and overlap: {chunk_overlap}")
    new_rows = []
    chunk_lengths = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Chunking (chunk size: {chunk_size})"):
        try:
            chunks = chunk_text_by_sentences(row['text'], chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_length = len(chunk)
                chunk_lengths.append(chunk_length)
                new_rows.append({
                    'unique_id': f"{row['id']}_{i+1}",
                    'url': row['url'],
                    'last_updated': row['last_updated'],
                    'html_content': row['html_content'],
                    'text': chunk,
                    'len': chunk_length,
                    'general_id': row['id']
                })
        except Exception as e:
            logger.error(f"Error processing row {row['id']}: {str(e)}")
    
    result = pd.DataFrame(new_rows)
    logger.info(f"Finished chunking. Created {len(result)} chunks from {len(data)} original rows.")
    
    min_length = min(chunk_lengths) if chunk_lengths else 0
    max_length = max(chunk_lengths) if chunk_lengths else 0
    mean_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    median_length = sorted(chunk_lengths)[len(chunk_lengths) // 2] if chunk_lengths else 0
    
    return result, min_length, max_length, mean_length, median_length

def process_data_sentences(df: pd.DataFrame, chunk_sizes: List[int], embed_model: Any, embed_model_name: str, base_path: str) -> List[Tuple[int, int, int, float, float]]:
    chunk_stats = []
    for chunk_size in chunk_sizes:
        logger.info(f"Processing sentence-based chunks with size: {chunk_size}")
        
        chunk_overlap = get_overlap(chunk_size)
        logger.info(f"Using overlap of {chunk_overlap} characters for chunk size {chunk_size}")
        
        chunked_df, min_length, max_length, mean_length, median_length = chunk_dataframe(df, chunk_size, chunk_overlap)
        
        logger.info("Embedding chunked texts")
        embedded_df = embed_dataframe(chunked_df, embed_model)
        
        logger.info("Generating document dictionaries")
        documents = generate_documents(embedded_df, chunk_size, 'sentence')
        
        logger.info("Saving documents to JSON")
        save_documents_to_json(documents, chunk_size, embed_model_name, 'sentence', base_path)
        
        chunk_stats.append((chunk_size, min_length, max_length, mean_length, median_length))
        
        logger.info(f"Chunk length statistics for size {chunk_size}:")
        logger.info(f"  Minimum length: {min_length}")
        logger.info(f"  Maximum length: {max_length}")
        logger.info(f"  Mean length: {mean_length:.2f}")
        logger.info(f"  Median length: {median_length:.2f}")
    
    return chunk_stats