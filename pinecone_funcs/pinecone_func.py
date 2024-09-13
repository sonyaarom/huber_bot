import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from pinecone import Pinecone, ServerlessSpec, PineconeException
from tqdm import tqdm
        
import numpy as np
from typing import Dict, List, Any
from pinecone import Pinecone, PineconeException
from tqdm import tqdm


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_pinecone(api_key: str = None, environment: str = None) -> Pinecone:
    """
    Initializes a connection to the Pinecone vector database.
    If api_key or environment are not provided, it will attempt to use environment variables.
    """
    logger.info("Initializing Pinecone connection")
    
    api_key = api_key or os.getenv('PINECONE_API_KEY')
    environment = environment or os.getenv('PINECONE_ENVIRONMENT')
    
    if not api_key:
        logger.error("Pinecone API key not provided and PINECONE_API_KEY environment variable not set")
        raise ValueError("Pinecone API key not set")
    
    if not environment:
        logger.error("Pinecone environment not provided and PINECONE_ENVIRONMENT environment variable not set")
        raise ValueError("Pinecone environment not set")
    
    try:
        pc = Pinecone(api_key=api_key, environment=environment)
        logger.info("Pinecone connection initialized successfully")
        return pc
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int, project_name: str, 
                          cloud: str = "aws", region: str = "us-east-1", metric: str = 'cosine') -> Tuple[Optional[Pinecone.Index], bool]:
    """
    Creates or accesses a Pinecone index with the specified parameters.
    """
    full_index_name = f"{index_name}"
    logger.info(f"Attempting to create or access index: {full_index_name} with metric: {metric}")
    try:
        if full_index_name not in pc.list_indexes():
            pc.create_index(
                full_index_name, 
                dimension=dimension, 
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
                deletion_protection='disabled'
            )
            logger.info(f"Created new index: {full_index_name} in {cloud} {region} with metric: {metric}")
            return pc.Index(full_index_name), True
        else:
            logger.info(f"Index {full_index_name} already exists. Proceeding with existing index.")
            return pc.Index(full_index_name), False
    except PineconeException as e:
        if "ALREADY_EXISTS" in str(e):
            logger.warning(f"Index {full_index_name} already exists. Proceeding with existing index.")
            return pc.Index(full_index_name), False
        else:
            logger.error(f"Error creating/accessing index {full_index_name}: {str(e)}")
            return None, False
        


def validate_vector(vector: List[float]) -> List[float]:
    """
    Validate the vector, replacing NaN values with 0.
    """
    return np.nan_to_num(vector, nan=0.1).tolist()
from typing import Dict, List, Any
from tqdm import tqdm
from pinecone import Pinecone, PineconeException
import logging

logger = logging.getLogger(__name__)

def format_sparse_vector(sparse_dict: Dict[str, float]) -> Dict[str, List]:
    """
    Formats the sparse vector to match Pinecone's expected format.
    
    :param sparse_dict: Dictionary with string keys and float values
    :return: Dictionary with 'indices' and 'values' lists
    """
    indices = [int(key) for key in sparse_dict.keys()]
    values = list(sparse_dict.values())
    return {"indices": indices, "values": values}

def upload_to_pinecone(documents: Dict[int, List[Dict[str, Any]]], pc: Pinecone, 
                       embedding_model_name: str, doc_type: str, project_name: str, metric: str = 'cosine') -> None:
    """
    Uploads document embeddings and sparse vectors to Pinecone indexes, configured for hybrid search.
    """
    logger.info(f"Starting upload process to Pinecone for {doc_type}-based documents in project {project_name} with metric: {metric}")
    
    for chunk_size, docs in documents.items():
        if not docs:
            logger.warning(f"No documents found for {doc_type}-based, chunk size {chunk_size}. Skipping.")
            continue
        
        # Validate and clean vectors
        cleaned_docs = []
        for doc in docs:
            try:
                cleaned_vector = validate_vector(doc['values'])
                cleaned_doc = doc.copy()
                cleaned_doc['values'] = cleaned_vector
                cleaned_docs.append(cleaned_doc)
            except Exception as e:
                logger.error(f"Error cleaning vector for document {doc.get('unique_id', 'unknown')}: {str(e)}")
        
        if not cleaned_docs:
            logger.warning(f"No valid documents after cleaning for {doc_type}-based, chunk size {chunk_size}. Skipping.")
            continue
        
        dimension = len(cleaned_docs[0]['values'])
        index_name = f"{embedding_model_name.replace('_', '-')}-{doc_type}-dim{dimension}-chunk{chunk_size}"
        
        index, is_new = create_pinecone_index(pc, index_name, dimension, project_name, metric=metric)
        
        if index is None:
            logger.error(f"Skipping processing for index {index_name} due to error.")
            continue
        
        # Check for existing documents
        existing_ids = set()
        batch_size = 100
        for i in range(0, len(cleaned_docs), batch_size):
            batch = cleaned_docs[i:i+batch_size]
            ids_to_check = [doc['unique_id'] for doc in batch]
            try:
                fetch_response = index.fetch(ids_to_check)
                existing_ids.update(fetch_response.vectors.keys())
            except PineconeException as e:
                logger.error(f"Error checking existing documents in index {index_name}: {str(e)}")
                break
        
        # Filter out existing documents
        new_docs = [doc for doc in cleaned_docs if doc['unique_id'] not in existing_ids]
        logger.info(f"Found {len(existing_ids)} existing documents. {len(new_docs)} new documents to upload.")
        
        if not new_docs:
            logger.info(f"No new documents to upload for {doc_type}-based, chunk size {chunk_size}. Skipping.")
            continue
        
        vectors_to_upsert = []
        for doc in new_docs:
            vector = {
                'id': doc['unique_id'],
                'values': doc['values'],
                'metadata': {
                    **doc['metadata'],
                    "chunk_size": chunk_size,
                    "doc_type": doc_type,
                    "project": project_name
                }
            }
            # Format and add sparse values if they exist
            if 'sparse_values' in doc and doc['sparse_values']:
                vector['sparse_values'] = format_sparse_vector(doc['sparse_values'])
            vectors_to_upsert.append(vector)
        
        for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc=f"Uploading to {index_name}"):
            batch = vectors_to_upsert[i:i+batch_size]
            try:
                index.upsert(vectors=batch)
            except PineconeException as e:
                logger.error(f"Error upserting to index {index_name}: {str(e)}")
                break
        else:
            logger.info(f"Successfully uploaded {len(new_docs)} new documents to index '{index_name}'")
    
    logger.info("Upload process completed.")

def delete_index(pc: Pinecone, index_name: str) -> bool:
    """
    Deletes a Pinecone index.
    """
    try:
        pc.delete_index(index_name)
        logger.info(f"Successfully deleted index: {index_name}")
        return True
    except PineconeException as e:
        logger.error(f"Error deleting index {index_name}: {str(e)}")
        return False

def list_indexes(pc: Pinecone) -> List[str]:
    """
    Lists all Pinecone indexes.
    """
    try:
        indexes = pc.list_indexes()
        logger.info(f"Found indexes: {indexes}")
        return indexes
    except PineconeException as e:
        logger.error(f"Error listing indexes: {str(e)}")
        return []

# Additional utility functions can be added here as needed