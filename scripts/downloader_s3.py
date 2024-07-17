import boto3
import json
import sys

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

