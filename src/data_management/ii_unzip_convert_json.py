import boto3
import gzip
import io
import logging
import re
import hashlib
import json
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_file(bucket_name, s3_key_prefix):
    """
    Retrieves the key of the most recently modified file in an S3 bucket with a given prefix.
    
    Args:
    bucket_name (str): The name of the S3 bucket.
    s3_key_prefix (str): The prefix to filter objects in the bucket.
    
    Returns:
    str: The key of the most recently modified file.
    
    Raises:
    FileNotFoundError: If no files are found with the given prefix.
    """
    
    s3 = boto3.client('s3')
    logging.info(f"Listing objects in bucket {bucket_name} with prefix {s3_key_prefix}")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_key_prefix)
    if 'Contents' not in response:
        raise FileNotFoundError(f"No files found with prefix {s3_key_prefix} in bucket {bucket_name}")
    files = response['Contents']
    latest_file = max(files, key=lambda x: x['LastModified'])
    logging.info(f"Latest file found: {latest_file['Key']}")
    return latest_file['Key']

def process_and_upload_file(source_bucket, source_key, dest_bucket, dest_key):
    """
    Downloads a gzip file from S3, unzips it in memory, and uploads the unzipped content to S3.
    
    Args:
    source_bucket (str): The name of the source S3 bucket.
    source_key (str): The key of the source gzip file in S3.
    dest_bucket (str): The name of the destination S3 bucket.
    dest_key (str): The key for the destination file in S3.
    
    Raises:
    ClientError: If there's an error uploading the file to S3.
    """
    s3 = boto3.client('s3')
    
    # Download and unzip in memory
    logging.info(f"Downloading and unzipping file from S3: {source_key}")
    response = s3.get_object(Bucket=source_bucket, Key=source_key)
    with gzip.GzipFile(fileobj=io.BytesIO(response['Body'].read())) as gzipfile:
        content = gzipfile.read()

    # Upload unzipped content to S3 with versioning
    logging.info(f"Uploading unzipped content to S3: {dest_key}")
    try:
        s3.put_object(Bucket=dest_bucket, Key=dest_key, Body=content)
        logging.info(f"Successfully uploaded {dest_key} to {dest_bucket}")
    except ClientError as e:
        logging.error(f"Error uploading file: {e}")
        raise


def unzip_local_file(file_path):
    """
    Unzips a gzip file from a local path and reads its content into memory.
    
    Args:
    file_path (str): The path to the local gzip file.
    
    Returns:
    bytes: The unzipped content of the file.
    """
    logging.info(f"Unzipping local file: {file_path}")
    try:
        with gzip.open(file_path, 'rb') as gzipfile:
            content = gzipfile.read()
        logging.info(f"Successfully unzipped local file: {file_path}")
        return content
    except Exception as e:
        logging.error(f"Error unzipping local file: {e}")
        raise


def filter_file_content(file_content: str) -> list:
    """
    Filters the file content to extract URLs and lastmod dates using regex.

    Args:
    file_content (str): The content of the XML file.

    Returns:
    list: A list of tuples containing URLs and lastmod dates.
    """
    pattern = r'''<loc>(https://www\.wiwi\.hu-berlin\.de/en/(?!.*\.jpeg|.*\.pdf|.*\.png|.*\.jpg).*?)(?<!/view)</loc>\s*<lastmod>([^<]+)</lastmod>'''
    matches = re.findall(pattern, file_content)
    return matches

def create_matches_dict(matches: list) -> dict:
    """
    Creates a dictionary from the matches list, using MD5 hash of URLs as keys.

    Args:
    matches (list): A list of tuples containing URLs and lastmod dates.

    Returns:
    dict: A dictionary with MD5 hashes as keys and URL info as values.
    """
    data_dict = {}
    for link, date in matches:
        hash_object = hashlib.md5()
        hash_object.update(link.encode('utf-8'))
        url_hash = hash_object.hexdigest()
        data_dict[url_hash] = {
            'url': link,
            'last_updated': date
        }
    return data_dict

def process_s3_files(bucket_name, xml_prefix='', json_prefix='json_files/'):
    """
    Processes XML files in an S3 bucket, extracts URL information, and saves the result as a JSON file.

    Args:
    bucket_name (str): The name of the S3 bucket.
    xml_prefix (str): The prefix for XML files in the bucket.
    json_prefix (str): The prefix for the output JSON file.
    """
    s3 = boto3.client('s3')
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=xml_prefix)
    
    all_matches = {}
    
    for page in pages:
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.xml'):
                response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                file_content = response['Body'].read().decode('utf-8')
                
                matches = filter_file_content(file_content)
                matches_dict = create_matches_dict(matches)
                
                all_matches.update(matches_dict)
                
                print(f"Processed {obj['Key']}")
                print(f"Number of matches: {len(matches_dict)}")
                
            
    
    # Save all matches to a JSON file in S3
    json_data = json.dumps(all_matches, indent=2)
    json_file_key = f"{json_prefix}all_matches.json"
    s3.put_object(Bucket=bucket_name, Key=json_file_key, Body=json_data)
    print(f"All matches saved to {json_file_key} in bucket {bucket_name}")

if __name__ == "__main__":
    source_bucket = 'hu-chatbot-schema'
    source_prefix = 'gz_files/'
    dest_bucket = 'hu-chatbot-schema'  # You can change this if you want to use a different bucket
    dest_prefix = 'xml_files/'  # Prefix for the XML files
    
    try:
        # Get the latest gzip file
        latest_file_key = get_latest_file(source_bucket, source_prefix)
        
        # Generate the destination key (changing extension from .gz to .xml)
        dest_key = dest_prefix + latest_file_key.split('/')[-1].replace('.gz', '.xml')
        
        # Process and upload the file
        process_and_upload_file(source_bucket, latest_file_key, dest_bucket, dest_key)
        
        logging.info(f"Process completed. Check S3 bucket {dest_bucket} for the file {dest_key}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

#if __name__ == "__main__":
#    bucket_name = 'hu-chatbot-schema'
#    xml_prefix = 'xml_files/'
#    json_prefix = 'json_files/'
#    process_s3_files(bucket_name, xml_prefix, json_prefix)