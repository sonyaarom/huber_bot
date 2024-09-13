import pandas as pd
from typing import List, Tuple, Any, Dict
from tqdm import tqdm
from logging import getLogger
import gc
from shared_utils import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gliner import GLiNER
from collections import defaultdict

logger = getLogger(__name__)

# Initialize GLiNER model
ner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
labels = ["person", "course", "date", "research_paper", "research_project", "teams", "city", "address", "organisation", "phone_number", "url", "other"]

def get_overlap(chunk_size: int) -> int:
    return 50 if chunk_size <= 256 else 200

def convert_entities_to_label_name_dict(entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Converts a list of entity dictionaries into a dictionary where each label has a list of entity texts.
    """
    label_name_dict = defaultdict(set)
    for entity in entities:
        text = entity['text'].strip().lower()
        label_name_dict[entity['label']].add(text)
    return {k: list(v) for k, v in label_name_dict.items()}

def process_data_recursive_langchain(df: pd.DataFrame, chunk_lengths: List[int], embed_model: Any, embed_model_name: str, base_path: str) -> List[Tuple[int, int, int, float, float]]:
    chunk_stats = []

    for chunk_length in chunk_lengths:
        logger.info(f"Processing recursive chunks with max length: {chunk_length}")
        
        # Initialize the RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_length,
            chunk_overlap=get_overlap(chunk_length),
            length_function=len,
            is_separator_regex=False,
        )

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
                chunks = text_splitter.split_text(row['text'])
                all_chunks.extend(chunks)
                unique_ids.extend([f"{row['id']}_{i+1}" for i in range(len(chunks))])
                general_ids.extend([row['id']] * len(chunks))
                urls.extend([row['url']] * len(chunks))
                last_updateds.extend([row.get('last_updated', '')] * len(chunks))
                html_contents.extend([row.get('html_content', '')] * len(chunks))
                
                # Perform Named Entity Recognition on each chunk
                chunk_entities = [convert_entities_to_label_name_dict(ner_model.predict_entities(chunk, labels)) for chunk in chunks]
                entities.extend(chunk_entities)

        # Create the chunked DataFrame
        chunked_df = pd.DataFrame({
            'unique_id': unique_ids,
            'url': urls,
            'last_updated': last_updateds,
            'html_content': html_contents,
            'text': all_chunks,
            'len': [len(chunk) for chunk in all_chunks],
            'general_id': general_ids,
            'entities': entities
        })

        # Remove chunks shorter than 50 characters
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
        chunked_df = apply_bm25_sparse_vectors(chunked_df, 'text')

        logger.info("Embedding chunked texts")
        embedded_df = embed_dataframe(chunked_df, embed_model)

        logger.info("Generating document dictionaries")
        documents = generate_documents(embedded_df, chunk_length, 'recursive')

        logger.info("Saving documents to JSON")
        save_documents_to_json(documents, chunk_length, embed_model_name, 'recursive', base_path)

        # Clear memory
        del chunked_df, embedded_df, documents
        gc.collect()

        logger.info(f"Finished processing chunks of max length {chunk_length}.")

    return chunk_stats