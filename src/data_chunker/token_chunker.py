# token_chunker.py

from shared_utils import *


def chunk_text(text, max_length=512, min_length=100, overlap=50):
    """
    Splits the input text into chunks of specified length range with optional overlap,
    ensuring that words are not split and the maximum length is strictly enforced.

    Args:
        text (str): The input text to be chunked.
        max_length (int): The maximum length of each chunk. Default is 512 characters.
        min_length (int): The minimum length of each chunk. Default is 100 characters.
        overlap (int): The number of characters that overlap between chunks. Default is 50 characters.
    
    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        # If adding this word would exceed max_length, finalize the current chunk
        if current_length + len(word) + 1 > max_length:
            if current_length >= min_length:
                chunks.append(' '.join(current_chunk))
                # Start a new chunk with overlap
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    if overlap_length + len(w) + 1 > overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_length += len(w) + 1
                current_chunk = overlap_words
                current_length = overlap_length
            else:
                # If current chunk is too short, continue adding words
                pass
        
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space

        # Check again if we've exceeded max_length
        while current_length > max_length:
            # Remove words from the beginning until we're under max_length
            removed_word = current_chunk.pop(0)
            current_length -= len(removed_word) + 1

    # Add the last chunk if it meets the minimum length
    if current_length >= min_length:
        chunks.append(' '.join(current_chunk))
    elif chunks:
        # If the last chunk is too short, append it to the previous chunk
        chunks[-1] = f"{chunks[-1]} {' '.join(current_chunk)}"
    
    logger.debug(f"Created {len(chunks)} chunks.")
    return chunks



def chunk_dataframe(data: pd.DataFrame, chunk_length: int) -> Tuple[pd.DataFrame, int, int]:
    logger.info(f"Starting to chunk dataframe with chunk length: {chunk_length}")
    new_rows = []
    min_length = float('inf')
    max_length = 0
    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Chunking (length: {chunk_length})"):
        try:
            chunks = chunk_text(row['text'], max_length=chunk_length, overlap=chunk_length // 4)
            for i, chunk in enumerate(chunks):
                chunk_length = len(chunk)
                min_length = min(min_length, chunk_length)
                max_length = max(max_length, chunk_length)
                new_rows.append({
                    'unique_id': f"{row['id']}_{i+1}",
                    'url': row['url'],
                    'last_updated': row['last_updated'],
                    'html_content': row['html_content'],
                    'text': chunk,
                    'len': chunk_length,
                    'id': row['id']
                })
        except Exception as e:
            logger.error(f"Error processing row {row['id']}: {str(e)}")
    result = pd.DataFrame(new_rows)
    logger.info(f"Finished chunking. Created {len(result)} chunks from {len(data)} original rows.")
    return result, min_length, max_length

def process_data_tokens(df: pd.DataFrame, chunk_lengths: List[int], embed_model: Any, embed_model_name: str, base_path: str) -> List[Tuple[int, int, int]]:
    chunk_stats = []
    for chunk_length in chunk_lengths:
        logger.info(f"Processing token-based chunks with length: {chunk_length}")
        chunked_df, min_length, max_length = chunk_dataframe(df, chunk_length)
        
        logger.info("Embedding chunked texts")
        embedded_df = embed_dataframe(chunked_df, embed_model)
        
        logger.info("Generating document dictionaries")
        documents = generate_documents(embedded_df, chunk_length, 'token')
        
        logger.info("Saving documents to JSON")
        save_documents_to_json(documents, chunk_length, embed_model_name, 'token', base_path)
        
        chunk_stats.append((chunk_length, min_length, max_length))
    return chunk_stats
