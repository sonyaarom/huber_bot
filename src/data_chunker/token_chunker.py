import pandas as pd
from typing import List, Tuple, Any
from tqdm import tqdm
from logging import getLogger
import gc
from shared_utils import *
import tiktoken
from gliner import GLiNER
import numpy as np
from collections import defaultdict

ner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
labels = ["person", "course", "date",  "research_project", "address", "organisation", "phone_number", "url", "other"]
logger = getLogger(__name__)

def get_overlap(chunk_length: int) -> int:
    """
    Returns the appropriate overlap based on chunk length.
    """
    if chunk_length in [128, 256]:
        return 50
    else:
        return 200
    

def convert_entities_to_label_name_dict(entities):
    label_name_dict = defaultdict(set)  # Use a set instead of a list

    for entity in entities:
        # Strip whitespace and convert to lowercase for case-insensitive deduplication
        text = entity['text'].strip().lower()
        label_name_dict[entity['label']].add(text)
    
    # Convert sets back to lists
    return {k: list(v) for k, v in label_name_dict.items()}


def chunk_text(text: str, chunk_size: int, overlap: int, encoding) -> List[str]:
    """
    Splits the input text into chunks of specified token length with overlap.
    """
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def process_data_tokens_tiktoken(df: pd.DataFrame, chunk_lengths: List[int], embed_model: Any, embed_model_name: str, base_path: str,  bm25_values: dict) -> List[Tuple[int, int, int, float, float]]:
    chunk_stats = []
    encoding = tiktoken.get_encoding("cl100k_base")

    for chunk_length in chunk_lengths:
        logger.info(f"Processing token-based chunks with length: {chunk_length}")

        # Calculate overlap
        overlap = get_overlap(chunk_length)
        logger.info(f"Using overlap of {overlap} for chunk length {chunk_length}")

        # Process all rows
        all_chunks = []
        unique_ids = []
        general_ids = []
        urls = []
        last_updateds = []
        html_contents = []
        entities = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking texts"):
            if isinstance(row['text'], str):
                chunks = chunk_text(row['text'], chunk_length, overlap, encoding)
                all_chunks.extend(chunks)
                unique_ids.extend([f"{row['id']}_{i+1}" for i in range(len(chunks))])
                general_ids.extend([row['id']] * len(chunks))
                urls.extend([row['url']] * len(chunks))
                last_updateds.extend([row.get('last_updated', '')] * len(chunks))
                html_contents.extend([row.get('html_content', '')] * len(chunks))
                entities.extend([convert_entities_to_label_name_dict(ner_model.predict_entities(chunk, labels)) for chunk in chunks])

        # Create the chunked DataFrame
        chunked_df = pd.DataFrame({
            'unique_id': unique_ids,
            'url': urls,
            'last_updated': last_updateds,
            'html_content': html_contents,
            'text': all_chunks,
            'len': [len(encoding.encode(chunk)) for chunk in all_chunks],
            'general_id': general_ids,
            'entities': entities
        })

        # Remove chunks shorter than 50 tokens
        chunked_df = chunked_df[chunked_df['len'] >= 50]

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

        logger.info("Applying BM25 sparse vectorization")
        chunked_df = apply_bm25_sparse_vectors(chunked_df, 'text', bm25_values)

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
