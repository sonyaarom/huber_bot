#shared utils
import pandas as pd
import os
import sys
from tqdm import tqdm
from typing import Any, List, Dict, Tuple
import logging
import json
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
import re
from llama_index.core.node_parser import SentenceSplitter
import uuid

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the assets directory
assets_dir = os.path.abspath(os.path.join(current_dir, '../..', 'assets'))

# Add the assets directory to the Python path
sys.path.insert(0, assets_dir)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_pinecone():
    """
    Initializes a connection to the Pinecone vector database.

    This function attempts to create a connection to Pinecone using the API key
    stored in the PINECONE_API_KEY environment variable. It handles potential
    errors and provides logging for the initialization process.

    Returns:
        Pinecone: An initialized Pinecone client object.

    Raises:
        ValueError: If the PINECONE_API_KEY environment variable is not set.
        Exception: If there's any other error during the Pinecone initialization process.

    Side Effects:
        - Logs information about the initialization process.
        - Accesses the PINECONE_API_KEY environment variable.

    Notes:
        - The function expects the PINECONE_API_KEY to be set in the environment variables.
        - It's recommended to call this function once and reuse the returned Pinecone
          client object throughout your application.
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


def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int, cloud: str = "aws", region: str = "us-east-1"):
    """
    Creates or accesses a Pinecone index with the specified parameters.

    Args:
        pc (Pinecone): An initialized Pinecone client object.
        index_name (str): The name of the index to create or access.
        dimension (int): The dimensionality of the vectors to be stored in the index.
        cloud (str, optional): The cloud provider to use. Defaults to "aws".
        region (str, optional): The region to create the index in. Defaults to "us-east-1".

    Returns:
        Tuple[Optional[Pinecone.Index], bool]: A tuple containing the Pinecone Index object 
                                               and a boolean indicating if a new index was created.
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
            logger.info(f"Index {index_name} already exists. Proceeding with existing index.")
            return pc.Index(index_name), False
    except Exception as e:
        if "ALREADY_EXISTS" in str(e):
            logger.warning(f"Index {index_name} already exists. Proceeding with existing index.")
            return pc.Index(index_name), False
        else:
            logger.error(f"Error creating/accessing index {index_name}: {str(e)}")
            return None, False
        

def get_embeddings(embed_model: Any, texts: List[str]) -> List[List[float]]:
    """
    A wrapper function to get embeddings from different types of models.

    Args:
        embed_model (Any): The embedding model (either HuggingFaceEmbeddings or a model with encode_documents method).
        texts (List[str]): A list of texts to embed.

    Returns:
        List[List[float]]: A list of embeddings.
    """
    if hasattr(embed_model, 'embed_documents'):
        return embed_model.embed_documents(texts)
    elif hasattr(embed_model, 'encode'):
        return embed_model.encode(texts)
    else:
        raise AttributeError("The provided model doesn't have 'embed_documents' or 'encode' method.")

# Update the embed_dataframe function to use the new wrapper
def embed_dataframe(data: pd.DataFrame, embed_model: Any) -> pd.DataFrame:
    """
    Generates embeddings for the text content of each row in the input DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing at least a 'text' column and a
                             'unique_id' column for error reporting.
        embed_model (Any): An embedding model object that has either 'embed_documents' or 'encode' method.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'embedding' column containing
                      the generated embeddings. Rows where embedding failed are removed.
    """
    embeddings = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Embedding texts"):
        try:
            embedding = get_embeddings(embed_model, [row['text']])[0]
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error embedding row {row['unique_id']}: {str(e)}")
            embeddings.append(None)
    
    data['embedding'] = embeddings
    result_df = data.dropna(subset=['embedding'])
    logger.info(f"Finished embedding. {len(result_df)} rows successfully embedded")
    return result_df

from typing import List, Dict, Any

def generate_documents(df: pd.DataFrame, chunk_size: int, doc_type: str) -> List[Dict[str, Any]]:
    """
    Generates a list of document dictionaries from the input DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing embedded text chunks.
        chunk_size (int): The chunk size used in chunking.
        doc_type (str): The type of document ('sentence', 'token', or 'semantic').
    
    Returns:
        List[Dict[str, Any]]: A list of document dictionaries.
    """
    documents = []
    for _, row in df.iterrows():
        # Debugging: Print row contents
        #print(f"Processing row: {row}")
        
        # Use get() method with a default value to avoid KeyError
        unique_id = row.get('unique_id', str(uuid.uuid4()))
        
        # Prioritize 'id' over 'general_id' for consistency
        general_id = row.get('id', row.get('general_id', 'unknown'))
        
        # Debugging: Print ID fields
        #print(f"unique_id: {unique_id}, general_id: {general_id}")
        
        document = {
            "unique_id": unique_id,
            'values': row.get("embedding", []),
            "metadata": {
                "url": row.get("url", ""),
                "text": row.get("text", ""),
                'general_id': general_id,
                'chunk_size': chunk_size,
                'doc_type': doc_type
            }
        }
        
        # Add 'last_updated' to metadata if it exists in the DataFrame
        if 'last_updated' in row:
            document['metadata']['date'] = row['last_updated']
        
        documents.append(document)
    
    return documents




import numpy as np

def save_documents_to_json(docs: List[dict], chunk_size: int, model_name: str, doc_type: str, base_path: str) -> None:
    """
    Saves a list of document dictionaries to a JSON file, converting NumPy arrays to lists.

    Args:
        docs (List[dict]): A list of document dictionaries to be saved.
        chunk_size (int): The chunk size used in chunking.
        model_name (str): The name of the model used for processing.
        doc_type (str): The type of document ('sentence', 'token', or 'semantic_texttiling').
        base_path (str): The base path where to save the JSON file.
    """
    filename = f"{doc_type}-vectors-{chunk_size}chunksize-{model_name}.json"
    filepath = os.path.join(base_path, filename)
    logger.info(f"Saving {len(docs)} documents to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_list(item) for item in obj]
        return obj

    # Convert NumPy arrays to lists
    docs_converted = numpy_to_list(docs)

    with open(filepath, 'w') as f:
        json.dump(docs_converted, f)
    logger.info(f"Documents successfully saved to {filepath}")



def load_documents(directory):
    """
    Loads document data from JSON files in the specified directory.

    Args:
        directory (str): The path to the directory containing JSON files with document data.

    Returns:
        Dict[str, Dict[int, List[Dict]]]: A dictionary where keys are document types ('token' or 'sentence'),
                                          values are dictionaries with chunk sizes as keys and 
                                          lists of document dictionaries as values.
    """
    logger.info(f"Loading documents from directory: {directory}")
    documents = {'token': {}, 'sentence': {}}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            logger.debug(f"Processing file: {filename}")
            try:
                # Updated regex pattern to match both old and new filename formats and extract document type
                match = re.search(r'(sentsplit_)?documents-(\d+)(chunksize|sentences|-)', filename)
                if match:
                    is_sentence = bool(match.group(1))  # True if 'sentsplit_' prefix is present
                    chunk_size = int(match.group(2))
                    doc_type = 'sentence' if is_sentence else 'token'
                    
                    with open(os.path.join(directory, filename), 'r') as f:
                        data = json.load(f)
                        documents[doc_type][chunk_size] = data
                        logger.info(f"Loaded {len(data)} {doc_type}-separated documents for chunk size {chunk_size}")
                else:
                    logger.warning(f"Could not extract information from filename: {filename}")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
    
    logger.info(f"Total configurations loaded: Token-separated: {len(documents['token'])}, Sentence-separated: {len(documents['sentence'])}")
    return documents


def upload_to_pinecone(documents, pc, embedding_model_name, api_key, environment, doc_type):
    """
    Uploads document embeddings to Pinecone indexes, organized by document type and chunk sizes.

    Args:
        documents (Dict[int, List[Dict]]): A dictionary where keys are chunk sizes and values are
                                           lists of document dictionaries containing embeddings and metadata.
        pc (Pinecone): An initialized Pinecone client object.
        embedding_model_name (str): The name of the embedding model used, for index naming.
        api_key (str): The Pinecone API key.
        environment (str): The Pinecone environment.
        doc_type (str): The type of documents being uploaded ('sentence' or 'token').
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