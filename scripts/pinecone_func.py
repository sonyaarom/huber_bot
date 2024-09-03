import os
import logging
from tqdm import tqdm
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

logger = logging.getLogger(__name__)


def flatten_values(documents):
    for doc in documents:
        if isinstance(doc['values'][0], list):
            doc['values'] = doc['values'][0]
            
    return documents

def pinecone_upsert(index, documents, batch_size=1000):
    documents = flatten_values(documents)
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        index.upsert(vectors=batch)


def initialize_pinecone():
    """
    Initializes a connection to the Pinecone vector database.

    Returns:
        Pinecone: An initialized Pinecone client object.
    """
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

def create_pinecone_index(pc, index_name, dimension, cloud="aws", region="us-east-1"):
    """
    Creates or accesses a Pinecone index with the specified parameters.
    """
    logger.info(f"Attempting to create or access index: {index_name}")
    try:
        if index_name not in pc.list_indexes():
            pc.create_index(
                index_name, 
                dimension=dimension, 
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                ),
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

def upload_to_pinecone(documents, pc, embedding_model_name, api_key, environment, doc_type):
    """
    Uploads document embeddings to Pinecone indexes, organized by document type and chunk sizes.
    """
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

        # Check for existing documents
        existing_ids = set()
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            ids_to_check = [doc['unique_id'] for doc in batch]
            try:
                fetch_response = index.fetch(ids_to_check)
                existing_ids.update(fetch_response.vectors.keys())
            except Exception as e:
                logger.error(f"Error checking existing documents in index {index_name}: {str(e)}")
                break

        # Filter out existing documents
        new_docs = [doc for doc in docs if doc['unique_id'] not in existing_ids]
        logger.info(f"Found {len(existing_ids)} existing documents. {len(new_docs)} new documents to upload.")

        if not new_docs:
            logger.info(f"No new documents to upload for {doc_type}-based, chunk size {chunk_size}. Skipping.")
            continue

        vectors_to_upsert = [
            (doc['unique_id'], doc['values'], {
                "text": doc['metadata']['text'], 
                "general_id": doc['metadata']['general_id'],
                "chunk_size": chunk_size,
                "doc_type": doc_type
            })
            for doc in new_docs
        ]

        for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc=f"Uploading to {index_name}"):
            batch = vectors_to_upsert[i:i+batch_size]
            try:
                index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Error upserting to index {index_name}: {str(e)}")
                break
        else:
            logger.info(f"Successfully uploaded {len(new_docs)} new documents to index '{index_name}'")