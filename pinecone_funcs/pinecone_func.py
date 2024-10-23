import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from pinecone import Pinecone, ServerlessSpec, PineconeException
from tqdm import tqdm
        
import numpy as np
from typing import Dict, List, Any
from pinecone import Pinecone, PineconeException
from tqdm import tqdm

from typing import Dict, List, Any
from tqdm import tqdm
from pinecone import Pinecone, PineconeException
import logging


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
                      index_name: str,  # Now accepting index_name directly
                      embedding_model_name: str, doc_type: str, project_name: str, 
                      metric: str = 'cosine', host = 'us-east-1', cloud = 'aws') -> None:
    """
    Uploads document embeddings and sparse vectors to a Pinecone index.
    Now accepts index_name directly to ensure consistency.
    """
    logger.info(f"Starting upload process to Pinecone for project {project_name} with metric: {metric}")
    logger.info(f"Using provided index name: {index_name}")
    
    try:
        # Access the index
        index = pc.Index(index_name)
        
        # Process documents for each chunk size
        for chunk_size, docs in documents.items():
            if not docs:
                logger.warning(f"No documents found for chunk size {chunk_size}. Skipping.")
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
                logger.warning(f"No valid documents after cleaning for chunk size {chunk_size}. Skipping.")
                continue

            # Prepare vectors to upsert
            vectors_to_upsert = []
            for doc in cleaned_docs:
                vector = {
                    'id': doc['unique_id'],
                    'values': doc['values'],
                    'metadata': {
                        **doc.get('metadata', {}),
                        "chunk_size": chunk_size,
                        "project": project_name
                    }
                }
                if 'sparse_values' in doc and doc['sparse_values']:
                    vector['sparse_values'] = format_sparse_vector(doc['sparse_values'])
                vectors_to_upsert.append(vector)

            # Upsert vectors in batches
            batch_size = 100
            total_uploaded = 0
            for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc=f"Uploading to {index_name}"):
                batch = vectors_to_upsert[i:i+batch_size]
                try:
                    index.upsert(vectors=batch)
                    total_uploaded += len(batch)
                except Exception as e:
                    logger.error(f"Error upserting batch to index {index_name}: {str(e)}")
                    raise

            logger.info(f"Successfully uploaded {total_uploaded} documents to index '{index_name}'")
            
    except Exception as e:
        logger.error(f"Error during upload process: {str(e)}")
        raise



def prepare_documents_for_upload(documents: List[Dict[str, Any]], chunk_size: int, doc_type: str, project_name: str) -> List[Dict[str, Any]]:
    """
    Prepares documents for upload by creating a structure similar to what's used in upload_to_pinecone.
    
    Args:
    documents (List[Dict[str, Any]]): List of document dictionaries.
    chunk_size (int): Size of the chunk for these documents.
    doc_type (str): Type of the document (e.g., 'text', 'recursive').
    project_name (str): Name of the project.

    Returns:
    List[Dict[str, Any]]: List of prepared documents ready for upload.
    """
    logger.info(f"Preparing {len(documents)} documents for upload. Chunk size: {chunk_size}, Doc type: {doc_type}, Project: {project_name}")

    prepared_docs = []

    for doc in tqdm(documents, desc="Preparing documents"):
        try:
            # Validate and clean vector
            cleaned_vector = validate_vector(doc.get('values', []))

            # Prepare the document structure
            prepared_doc = {
                'id': doc.get('unique_id', ''),
                'values': cleaned_vector,
                'metadata': {
                    'url': doc.get('url', ''),
                    'last_updated': doc.get('last_updated', ''),
                    'text': doc.get('text', ''),
                    'chunk_size': chunk_size,
                    'doc_type': doc_type,
                    'project': project_name
                }
            }

            # Add any additional metadata fields
            for key, value in doc.items():
                if key not in ['unique_id', 'values', 'url', 'last_updated', 'text']:
                    prepared_doc['metadata'][key] = value

            # Format and add sparse values if they exist
            if 'sparse_values' in doc and doc['sparse_values']:
                prepared_doc['sparse_values'] = format_sparse_vector(doc['sparse_values'])

            prepared_docs.append(prepared_doc)

        except Exception as e:
            logger.error(f"Error preparing document {doc.get('unique_id', 'unknown')}: {str(e)}")

    logger.info(f"Successfully prepared {len(prepared_docs)} documents for upload")

    return prepared_docs

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