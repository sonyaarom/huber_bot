import os
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

logger = logging.getLogger(__name__)

def flatten_values(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten nested 'values' in documents."""
    return [{**doc, 'values': doc['values'][0] if isinstance(doc['values'][0], list) else doc['values']}
            for doc in documents]

def pinecone_upsert(index: Pinecone, documents: List[Dict[str, Any]], batch_size: int = 1000) -> None:
    """Upsert documents to Pinecone index in batches."""
    flattened_docs = flatten_values(documents)
    for i in range(0, len(flattened_docs), batch_size):
        batch = flattened_docs[i:i + batch_size]
        index.upsert(vectors=batch)

def initialize_pinecone() -> Pinecone:
    """Initialize a connection to the Pinecone vector database."""
    logger.info("Initializing Pinecone connection")
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        raise ValueError("PINECONE_API_KEY not set")
    try:
        pc = Pinecone(api_key=api_key)
        logger.info("Pinecone connection initialized successfully")
        return pc
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int, cloud: str = "aws", region: str = "us-east-1") -> Tuple[Pinecone, bool]:
    """Create or access a Pinecone index with the specified parameters."""
    logger.info(f"Attempting to create or access index: {index_name}")
    try:
        if index_name not in pc.list_indexes():
            pc.create_index(
                index_name, 
                dimension=dimension, 
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
                deletion_protection='disabled'
            )
            logger.info(f"Created new index: {index_name} in {cloud} {region}")
            return pc.Index(index_name), True
        else:
            logger.info(f"Index {index_name} already exists.")
            return pc.Index(index_name), False
    except Exception as e:
        logger.error(f"Error creating/accessing index {index_name}: {str(e)}")
        return None, False

def upload_to_pinecone(documents: Dict[int, List[Dict[str, Any]]], pc: Pinecone, embedding_model_name: str, doc_type: str) -> None:
    """Upload document embeddings to Pinecone indexes, organized by document type and chunk sizes."""
    logger.info(f"Starting upload process to Pinecone for {doc_type}-based documents")
    for chunk_size, docs in documents.items():
        if not docs:
            logger.warning(f"No documents found for {doc_type}-based, chunk size {chunk_size}. Skipping.")
            continue

        dimension = len(docs[0]['values'])
        index_name = f"{embedding_model_name.replace('_', '-')}-{doc_type}-dim{dimension}-chunk{chunk_size}"

        index, is_new = create_pinecone_index(pc, index_name, dimension)

        if index is None:
            logger.error(f"Skipping processing for index {index_name} due to error.")
            continue

        new_docs = filter_existing_documents(index, docs)
        
        if not new_docs:
            logger.info(f"No new documents to upload for {doc_type}-based, chunk size {chunk_size}. Skipping.")
            continue

        vectors_to_upsert = prepare_vectors_for_upsert(new_docs, chunk_size, doc_type)
        upsert_vectors(index, vectors_to_upsert)

def filter_existing_documents(index: Pinecone, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out existing documents from the list."""
    existing_ids = set()
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        ids_to_check = [doc['unique_id'] for doc in batch]
        try:
            fetch_response = index.fetch(ids_to_check)
            existing_ids.update(fetch_response.vectors.keys())
        except Exception as e:
            logger.error(f"Error checking existing documents: {str(e)}")
            break
    new_docs = [doc for doc in docs if doc['unique_id'] not in existing_ids]
    logger.info(f"Found {len(existing_ids)} existing documents. {len(new_docs)} new documents to upload.")
    return new_docs

def prepare_vectors_for_upsert(docs: List[Dict[str, Any]], chunk_size: int, doc_type: str) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    """Prepare vectors for upserting to Pinecone."""
    return [
        (doc['unique_id'], doc['values'], {
            "text": doc['metadata']['text'], 
            "general_id": doc['metadata']['general_id'],
            "chunk_size": chunk_size,
            "doc_type": doc_type
        })
        for doc in docs
    ]

def upsert_vectors(index: Pinecone, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
    """Upsert vectors to Pinecone index in batches."""
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size), desc=f"Uploading to {index.describe_index_stats().index_fullname}"):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            logger.error(f"Error upserting to index: {str(e)}")
            break
    else:
        logger.info(f"Successfully uploaded {len(vectors)} new documents to index '{index.describe_index_stats().index_fullname}'")