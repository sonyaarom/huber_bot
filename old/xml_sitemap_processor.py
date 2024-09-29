import boto3
import gzip
import io
import logging
import re
import hashlib
import json
import os
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_file(location, prefix):
    """
    Retrieves the path of the most recently modified file in a given location (S3 or local).
    
    Args:
    location (str): The S3 bucket name or local directory path.
    prefix (str): The prefix to filter objects.
    
    Returns:
    str: The path of the most recently modified file.
    """
    if location.startswith('s3://'):
        bucket_name = location[5:]
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' not in response:
            raise FileNotFoundError(f"No files found with prefix {prefix} in bucket {bucket_name}")
        files = response['Contents']
        latest_file = max(files, key=lambda x: x['LastModified'])
        return f"s3://{bucket_name}/{latest_file['Key']}"
    else:
        full_path = os.path.join(location, prefix)
        files = [os.path.join(full_path, f) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
        if not files:
            raise FileNotFoundError(f"No files found in {full_path}")
        return max(files, key=os.path.getmtime)

def process_and_upload_file(source_path, dest_path):
    """
    Processes a gzip file from source and uploads the unzipped content to destination.
    
    Args:
    source_path (str): The path of the source gzip file (S3 or local).
    dest_path (str): The path for the destination file (S3 or local).
    """
    content = unzip_file(source_path)
    
    if dest_path.startswith('s3://'):
        bucket_name, key = dest_path[5:].split('/', 1)
        s3 = boto3.client('s3')
        try:
            s3.put_object(Bucket=bucket_name, Key=key, Body=content)
            logging.info(f"Successfully uploaded to S3: {dest_path}")
        except ClientError as e:
            logging.error(f"Error uploading file to S3: {e}")
            raise
    else:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            f.write(content)
        logging.info(f"Successfully saved to local path: {dest_path}")

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

def process_files(source_location, dest_location, xml_prefix='', json_prefix=''):
    """
    Processes XML files in a given location, extracts URL information, and saves the result as a JSON file.

    Args:
    source_location (str): The source location (S3 bucket or local directory).
    dest_location (str): The destination location (S3 bucket or local directory).
    xml_prefix (str): The prefix for XML files.
    json_prefix (str): The prefix for the output JSON file.
    """
    all_matches = {}
    
    if source_location.startswith('s3://'):
        bucket_name = source_location[5:]
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=xml_prefix)
        
        for page in pages:
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.xml'):
                    response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                    file_content = response['Body'].read().decode('utf-8')
                    process_file_content(file_content, all_matches)
                    print(f"Processed {obj['Key']}")
    else:
        for root, _, files in os.walk(os.path.join(source_location, xml_prefix)):
            for file in files:
                if file.endswith('.xml'):
                    with open(os.path.join(root, file), 'r') as f:
                        file_content = f.read()
                    process_file_content(file_content, all_matches)
                    print(f"Processed {os.path.join(root, file)}")
    
    # Save all matches to a JSON file
    json_data = json.dumps(all_matches, indent=2)
    if dest_location.startswith('s3://'):
        bucket_name = dest_location[5:]
        json_file_key = f"{json_prefix}all_matches.json"
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket_name, Key=json_file_key, Body=json_data)
        print(f"All matches saved to s3://{bucket_name}/{json_file_key}")
    else:
        json_file_path = os.path.join(dest_location, json_prefix, 'all_matches.json')
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        with open(json_file_path, 'w') as f:
            f.write(json_data)
        print(f"All matches saved to {json_file_path}")

def process_file_content(file_content, all_matches):
    matches = filter_file_content(file_content)
    matches_dict = create_matches_dict(matches)
    all_matches.update(matches_dict)
    print(f"Number of matches: {len(matches_dict)}")

def filter_file_content(file_content: str, pattern: str) -> list:
    """
    Filters the file content to extract URLs and lastmod dates using regex.

    Args:
    file_content (str): The content of the XML file.

    Returns:
    list: A list of tuples containing URLs and lastmod dates.
    """
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

def process_files(source_location, dest_location, xml_prefix='', json_prefix=''):
    """
    Processes XML files in a given location, extracts URL information, and saves the result as a JSON file.

    Args:
    source_location (str): The source location (S3 bucket or local directory).
    dest_location (str): The destination location (S3 bucket or local directory).
    xml_prefix (str): The prefix for XML files.
    json_prefix (str): The prefix for the output JSON file.
    """
    all_matches = {}
    
    if source_location.startswith('s3://'):
        bucket_name = source_location[5:]
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=xml_prefix)
        
        for page in pages:
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.xml'):
                    response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                    file_content = response['Body'].read().decode('utf-8')
                    process_file_content(file_content, all_matches)
                    print(f"Processed {obj['Key']}")
    else:
        for root, _, files in os.walk(os.path.join(source_location, xml_prefix)):
            for file in files:
                if file.endswith('.xml'):
                    with open(os.path.join(root, file), 'r') as f:
                        file_content = f.read()
                    process_file_content(file_content, all_matches)
                    print(f"Processed {os.path.join(root, file)}")
    
    # Save all matches to a JSON file
    json_data = json.dumps(all_matches, indent=2)
    if dest_location.startswith('s3://'):
        bucket_name = dest_location[5:]
        json_file_key = f"{json_prefix}all_matches.json"
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket_name, Key=json_file_key, Body=json_data)
        print(f"All matches saved to s3://{bucket_name}/{json_file_key}")
    else:
        json_file_path = os.path.join(dest_location, json_prefix, 'all_matches.json')
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        with open(json_file_path, 'w') as f:
            f.write(json_data)
        print(f"All matches saved to {json_file_path}")

def process_file_content(file_content, all_matches):
    matches = filter_file_content(file_content)
    matches_dict = create_matches_dict(matches)
    all_matches.update(matches_dict)
    print(f"Number of matches: {len(matches_dict)}")

# if __name__ == "__main__":

#     #source_location = 's3://hu-chatbot-schema'  # Can be 's3://bucket-name' or local path
#     #dest_location = 's3://hu-chatbot-schema'  # Can be 's3://bucket-name' or local path
#     source_location = "assets/gz_files/"  # Directory containing XML files
#     dest_location = 'assets/'
#     xml_prefix = 'xml_files/'
#     json_prefix = 'json_files/'
    
#     try:
#         latest_file = get_latest_file(source_location, '')  # Empty string to get all files in the directory
#         dest_file = os.path.join(dest_location, xml_prefix, os.path.basename(latest_file))
#         process_and_upload_file(latest_file, dest_file)
#         process_files(dest_location, dest_location, xml_prefix, json_prefix)
#         logging.info("Process completed.")
#     except Exception as e:
#         logging.error(f"An error occurred: {str(e)}")