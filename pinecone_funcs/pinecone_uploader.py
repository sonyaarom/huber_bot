import os
import json
import time
import argparse
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pinecone_func import initialize_pinecone, create_pinecone_index, upload_to_pinecone

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("pinecone_uploader.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


class VectorFileHandler(FileSystemEventHandler):
    def __init__(self, pinecone_client, embedding_model_name, project_name, metric):
        self.pc = pinecone_client
        self.embedding_model_name = embedding_model_name
        self.project_name = project_name
        self.metric = metric
        logger.info(f"VectorFileHandler initialized with embedding model: {embedding_model_name}, project: {project_name}, and metric: {metric}")

    def process_file(self, file_path):
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                documents = json.load(file)
            
            filename = os.path.basename(file_path)
            filename_parts = filename.split('-')
            
            # Extract doc_type and chunk_size
            doc_type = filename_parts[0]
            chunk_size = None
            for part in filename_parts:
                if 'chunksize' in part:
                    chunk_size = int(part.replace('chunksize', ''))
                    break
            
            if chunk_size is None:
                raise ValueError(f"Couldn't extract chunk size from filename: {filename}")
            
            logger.info(f"Extracted doc_type: {doc_type}, chunk_size: {chunk_size}")
            
            # Create a dictionary with chunk_size as key and documents as value
            documents_dict = {chunk_size: documents}
            
            # Call upload_to_pinecone with the dictionary, project name, and metric
            upload_to_pinecone(documents_dict, self.pc, self.embedding_model_name, doc_type, self.project_name, self.metric)
            
            logger.info(f"Successfully uploaded vectors from {file_path}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {file_path}")
        except Exception as e:
            logger.exception(f"Error processing file {file_path}: {str(e)}")

def process_existing_files(folder_path, handler):
    logger.info(f"Processing existing files in {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and handler.embedding_model_name in filename:
            file_path = os.path.join(folder_path, filename)
            handler.process_file(file_path)
    logger.info("Finished processing existing files")

def delete_all_vectors(pinecone_client):
    logger.info("Starting deletion of all vectors from all indexes")
    try:
        indexes = pinecone_client.list_indexes()
        logger.info(f"Found indexes: {indexes}")
        
        for index_info in indexes:
            try:
                index_name = index_info.name
                logger.info(f"Attempting to access index: {index_name}")
                index = pinecone_client.Index(index_name)
                stats = index.describe_index_stats()
                if stats.total_vector_count > 0:
                    logger.info(f"Deleting all vectors from index: {index_name}")
                    index.delete(delete_all=True)
                    logger.info(f"Deleted all vectors from index: {index_name}")
                else:
                    logger.info(f"Index {index_name} is already empty")
            except Exception as e:
                logger.error(f"Error processing index {index_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing indexes: {str(e)}")
    
    logger.info("Finished deletion process")

def delete_index_interactive(pinecone_client, index_name):
    """
    Deletes a single index after user confirmation.
    """
    logger.warning(f"Are you sure you want to delete the index '{index_name}'? (y/n)")
    confirmation = input().lower()
    if confirmation == 'y':
        try:
            pinecone_client.delete_index(index_name)
            logger.info(f"Successfully deleted index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting index {index_name}: {str(e)}")
            return False
    else:
        logger.info(f"Skipping deletion of index: {index_name}")
        return False

def delete_all_indexes(pinecone_client, interactive=True):
    logger.info("Starting deletion of indexes")
    try:
        indexes = pinecone_client.list_indexes()
        logger.info(f"Found indexes: {indexes}")
        
        if not interactive:
            logger.warning("Deleting all indexes without confirmation.")
            for index_info in indexes:
                index_name = index_info.name
                try:
                    pinecone_client.delete_index(index_name)
                    logger.info(f"Successfully deleted index: {index_name}")
                except Exception as e:
                    logger.error(f"Error deleting index {index_name}: {str(e)}")
        else:
            for index_info in indexes:
                index_name = index_info.name
                deleted = delete_index_interactive(pinecone_client, index_name)
                if not deleted:
                    logger.info(f"Index {index_name} was not deleted.")
    except Exception as e:
        logger.error(f"Error listing indexes: {str(e)}")
    
    logger.info("Finished deletion process")


def upload_files(folder_path, pinecone_client, embedding_model_name, project_name, metric):
    logger.info(f"Processing files in folder: {folder_path} for project: {project_name} with metric: {metric}")
    event_handler = VectorFileHandler(pinecone_client, embedding_model_name, project_name, metric)
    
    # Process existing files
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and embedding_model_name in filename:
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Processing file: {file_path}")
            event_handler.process_file(file_path)
    
    logger.info("Finished processing all files in the folder")
import os
import argparse
import logging
from pinecone_func import initialize_pinecone, create_pinecone_index, upload_to_pinecone

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("pinecone_uploader.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Pinecone Vector Upload and Delete Tool")
    parser.add_argument("--folder", type=str, help="Folder containing vector files to upload")
    parser.add_argument("--delete", choices=['interactive', 'all'], help="Delete indexes: 'interactive' for selective deletion, 'all' to delete all indexes")
    parser.add_argument("--embedding_model", type=str, required=True, help="Name of the embedding model")
    parser.add_argument("--project", type=str, default="default", help="Project name for the indexes")
    parser.add_argument("--api_key", type=str, help="Pinecone API key")
    parser.add_argument("--metric", choices=['dotproduct', 'cosine', 'euclidean'], default='cosine', help="Distance metric for the index")
    
    args = parser.parse_args()
    
    # Prioritize command-line argument, then environment variable, then default value
    api_key = args.api_key or os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    
    if not api_key or not environment:
        logger.error("Pinecone API key and environment must be set")
        raise ValueError("Pinecone API key and environment must be set")

    try:
        logger.info("Initializing Pinecone")
        pc = initialize_pinecone(api_key, environment)
        logger.info("Pinecone initialized successfully")
        logger.info(f"Pinecone client initialized with API key: {api_key[:5]}... and environment: {environment}")
        
        if args.delete:
            if args.delete == 'all':
                logger.warning("WARNING: This operation will delete all indexes without confirmation.")
                logger.warning("Are you sure you want to proceed? (y/n)")
                confirmation = input().lower()
                if confirmation == 'y':
                    logger.info("Proceeding with deletion of all indexes.")
                    delete_all_indexes(pc, interactive=False)
                else:
                    logger.info("Deletion cancelled.")
            elif args.delete == 'interactive':
                logger.warning("WARNING: This operation will allow you to delete indexes interactively.")
                logger.warning("Do you want to proceed with the interactive deletion process? (y/n)")
                confirmation = input().lower()
                if confirmation == 'y':
                    logger.info("Interactive deletion process confirmed. Proceeding with index review.")
                    delete_all_indexes(pc, interactive=True)
                else:
                    logger.info("Interactive deletion process cancelled.")
        elif args.folder:
            logger.info(f"Upload option selected for folder: {args.folder}")
            try:
                upload_files(args.folder, pc, args.embedding_model, args.project, args.metric)
            except Exception as e:
                logger.error(f"Error during file upload: {str(e)}")
        else:
            logger.warning("No valid option selected. Use --delete 'interactive' for selective index deletion, --delete 'all' to delete all indexes, or --folder to upload files from a folder.")
            print("Use --delete 'interactive' for selective index deletion, --delete 'all' to delete all indexes, or --folder to upload files from a folder")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    
    finally:
        logger.info("Script execution completed.")

if __name__ == "__main__":
    main()


#python pinecone_uploader.py --folder FOLDER --embedding_model EMBED_NAME  --project PROJECT_INTERNAL --metric dotproduct/euclidean/cosine  --api_key API_KEY
#python pinecone_uploader.py --delete all --embedding_model EMBED_NAME --api_key API_KEY
