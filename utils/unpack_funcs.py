import logging
import boto3
import io
import gzip
from download_funcs import download_file
logging.basicConfig(level=logging.INFO)


def unzip_file(file_path):
    """
    Unzips a gzip file from a given path (S3 or local) and reads its content.
    
    Args:
    file_path (str): The path to the gzip file.
    
    Returns:
    bytes: The unzipped content of the file.
    """
    logging.info(f"Unzipping file: {file_path}")
    try:
        if file_path.startswith('s3://'):
            bucket_name, key = file_path[5:].split('/', 1)
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket_name, Key=key)
            with gzip.GzipFile(fileobj=io.BytesIO(response['Body'].read())) as gzipfile:
                content = gzipfile.read()
        else:
            with gzip.open(file_path, 'rb') as gzipfile:
                content = gzipfile.read()
        logging.info(f"Successfully unzipped file: {file_path}")
        return content
    except Exception as e:
        logging.error(f"Error unzipping file: {e}")
        raise
