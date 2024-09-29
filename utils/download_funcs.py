import boto3
import json
import sys
import io
import pandas as pd
from urllib.parse import urlparse
import requests
import os


def download_from_s3(bucket_name, file_key):
    """
    Downloads a file from an S3 bucket and returns its content.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - file_key (str): The key of the file to download.

    Returns:
    - dict: The content of the downloaded file.
    """
    s3 = boto3.client('s3')  # Create an S3 client
    response = s3.get_object(Bucket=bucket_name, Key=file_key)  # Get the object from S3
    #file_content = response['Body'].read().decode('utf-8')  # Read and decode the content
    return response  # Parse and return the content as a JSON object

def download_csv_from_s3(bucket_name, file_key):
    """
    Downloads a CSV file from an S3 bucket and returns its content as a pandas DataFrame.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - file_key (str): The key of the file to download.

    Returns:
    - pandas.DataFrame: The content of the downloaded CSV file.
    """
    s3 = boto3.client('s3')  # Create an S3 client
    response = s3.get_object(Bucket=bucket_name, Key=file_key)  # Get the object from S3
    file_content = response['Body'].read().decode('utf-8')  # Read and decode the content
    data = pd.read_csv(io.StringIO(file_content))
    return data

def upload_to_s3(bucket_name, file_key, data):
    """
    Uploads the given data to an S3 bucket.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - file_key (str): The key of the file to upload.
    - data (dict): The data to upload.
    """
    s3 = boto3.client('s3')  # Create an S3 client
    #s3.put_object(Bucket=bucket_name, Key=file_key, Body=json.dumps(data))  # Upload the JSON-encoded data to S3
    s3.put_object(Bucket=bucket_name, Key=file_key, Body = data)

def upload_json_to_s3(bucket_name, file_key, data):
    """
    Uploads the given data to an S3 bucket.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - file_key (str): The key of the file to upload.
    - data (dict): The data to upload.
    """
    s3 = boto3.client('s3')  # Create an S3 client
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=json.dumps(data))  # Upload the JSON-encoded data to S3

def upload_csv_to_s3(bucket_name, file_key, data):
    """
    Uploads the given data to an S3 bucket.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - file_key (str): The key of the file to upload.
    - data (dict): The data to upload.
    """
    s3 = boto3.client('s3')  # Create an S3 client
    file = data.to_csv(index=False)
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=file)  # Upload the JSON-encoded data to S3



def download_file(url):
    """
    Downloads a file from the given URL and saves it to a temporary location.

    Args:
        url (str): The URL of the file to download.

    Returns:
        str: The path to the downloaded file.
    """
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)

    response = requests.get(url)
    if response.status_code == 200:
        temp_file_path = f'/tmp/{file_name}'
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(response.content)
        return temp_file_path
    else:
        raise Exception(f"Failed to download file from URL. Status code: {response.status_code}")


def store_file(temp_file_path, local_directory=None, bucket_name=None, s3_key_prefix=None):
    """
    Stores a file locally and/or in S3 bucket, based on provided parameters.

    Args:
        temp_file_path (str): The path to the temporary file.
        local_directory (str, optional): The local directory to store the file. If None, file is not stored locally.
        bucket_name (str, optional): The name of the S3 bucket. If None, file is not uploaded to S3.
        s3_key_prefix (str, optional): The S3 key prefix (path) to store the file under. Required if bucket_name is provided.

    Returns:
        dict: Information about the stored file.
    """
    file_name = os.path.basename(temp_file_path)
    result = {}

    # Store locally if local_directory is provided
    if local_directory is not None:
        local_path = os.path.join(local_directory, file_name)
        os.makedirs(local_directory, exist_ok=True)
        with open(temp_file_path, 'rb') as src_file, open(local_path, 'wb') as dst_file:
            dst_file.write(src_file.read())
        result['LocalPath'] = local_path

    # Upload to S3 if bucket_name is provided
    if bucket_name is not None:
        s3_key = os.path.join(s3_key_prefix, file_name)
        with open(temp_file_path, 'rb') as temp_file:
            upload_to_s3(bucket_name, s3_key, temp_file.read())

        # Get the version ID of the uploaded file
        s3_client = boto3.client('s3')
        head_response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        version_id = head_response.get('VersionId', 'null')

        result.update({
            'Bucket': bucket_name,
            'Key': s3_key,
            'VersionId': version_id
        })

    # Clean up the temporary file
    os.remove(temp_file_path)

    return result

def download_file_from_s3(bucket_name, object_key, local_file_path=None):
    # Create an S3 client
    s3 = boto3.client('s3')
    
    try:
        # If local_file_path is not provided, use a temporary directory
        if local_file_path is None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            local_file_path = os.path.join(temp_dir, os.path.basename(object_key))
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the file
        s3.download_file(bucket_name, object_key, local_file_path)
        print(f"File downloaded successfully to {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return None

def download_and_store_file(url, local_directory=None, bucket_name=None, s3_key_prefix=None):
    """
    Downloads a file from a URL and stores it locally and/or in S3, based on provided parameters.

    Args:
        url (str): The URL of the file to download.
        local_directory (str, optional): The local directory to store the file. If None, file is not stored locally.
        bucket_name (str, optional): The name of the S3 bucket. If None, file is not uploaded to S3.
        s3_key_prefix (str, optional): The S3 key prefix (path) to store the file under. Required if bucket_name is provided.

    Returns:
        dict: Information about the stored file.
    """
    if url is None:
        raise ValueError("URL must be provided.")
    
    if local_directory is None and bucket_name is None:
        raise ValueError("At least one of local_directory or bucket_name must be provided.")
    
    if bucket_name is not None and s3_key_prefix is None:
        raise ValueError("s3_key_prefix must be provided when bucket_name is specified.")

    temp_file_path = download_file(url)
    result = store_file(temp_file_path, local_directory, bucket_name, s3_key_prefix)
    return result
