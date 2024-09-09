import os
import logging
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_pinecone():
    """
    Initializes a connection to the Pinecone vector database.
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

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int, project_name: str, cloud: str = "aws", region: str = "us-east-1"):
    """
    Creates or accesses a Pinecone index with the specified parameters.
    """
    full_index_name = f"{index_name}"
    logger.info(f"Attempting to create or access index: {full_index_name}")
    try:
        if full_index_name not in pc.list_indexes():
            pc.create_index(
                full_index_name, 
                dimension=dimension, 
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                ),
                deletion_protection='disabled'
            )
            logger.info(f"Created new index: {full_index_name} in {cloud} {region}")
            return pc.Index(full_index_name), True
        else:
            logger.info(f"Index {full_index_name} already exists. Proceeding with existing index.")
            return pc.Index(full_index_name), False
    except Exception as e:
        if "ALREADY_EXISTS" in str(e):
            logger.warning(f"Index {full_index_name} already exists. Proceeding with existing index.")
            return pc.Index(full_index_name), False
        else:
            logger.error(f"Error creating/accessing index {full_index_name}: {str(e)}")
            return None, False

def upload_to_pinecone(documents, pc, embedding_model_name, doc_type, project_name):
    """
    Uploads document embeddings to Pinecone indexes, organized by document type and chunk sizes.
    """
    logger.info(f"Starting upload process to Pinecone for {doc_type}-based documents in project {project_name}")
    for chunk_size, docs in documents.items():
        if not docs:
            logger.warning(f"No documents found for {doc_type}-based, chunk size {chunk_size}. Skipping.")
            continue

        dimension = len(docs[0]['values'])
        index_name = f"{embedding_model_name.replace('_', '-')}-{doc_type}-dim{dimension}-chunk{chunk_size}"

        index, is_new = create_pinecone_index(pc, index_name, dimension, project_name)

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
                "doc_type": doc_type,
                "project": project_name
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

# Additional utility functions can be added here as needed