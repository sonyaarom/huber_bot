import sys
import os
import requests
import boto3
from urllib.parse import urlparse

# Ensure the script directory is in the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from downloader_s3 import upload_to_s3

def download_file(url):
    """
    Downloads a file from the given URL and saves it to a temporary location.

    Args:
        url (str): The URL of the file to download.

    Returns:
        str: The path to the downloaded file.
    """
    # Parse the URL to get the file name
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        # Save the file to a temporary location
        temp_file_path = f'/tmp/{file_name}'
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(response.content)
        return temp_file_path
    else:
        raise Exception(f"Failed to download file from URL. Status code: {response.status_code}")

def store_in_s3(temp_file_path, bucket_name, s3_key_prefix=''):
    """
    Uploads a file from a temporary location to an S3 bucket with versioning.

    Args:
        temp_file_path (str): The path to the temporary file.
        bucket_name (str): The name of the S3 bucket.
        s3_key_prefix (str, optional): The S3 key prefix (path) to store the file under. Defaults to ''.

    Returns:
        dict: Information about the uploaded file including version ID.
    """
    # Parse the file name
    file_name = os.path.basename(temp_file_path)

    # Upload the file to S3 using the new upload_to_s3 function
    s3_key = os.path.join(s3_key_prefix, file_name)
    with open(temp_file_path, 'rb') as temp_file:
        upload_to_s3(bucket_name, s3_key, temp_file.read())

    # Get the version ID of the uploaded file
    s3_client = boto3.client('s3')
    head_response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
    version_id = head_response.get('VersionId', 'null')

    # Clean up the temporary file
    os.remove(temp_file_path)

    return {
        'Bucket': bucket_name,
        'Key': s3_key,
        'VersionId': version_id
    }

# Example usage
#if __name__ == "__main__":
#    url = 'https://www.wiwi.hu-berlin.de/sitemap.xml.gz'
#    bucket_name = 'hu-chatbot-schema'
#    s3_key_prefix = 'gz_files'
#
#    # Step 1: Download the file
#    temp_file_path = download_file(url)
#    
#    # Step 2: Upload the file to S3
#    result = store_in_s3(temp_file_path, bucket_name, s3_key_prefix)
#    
#    print(f"File uploaded to S3 with version ID: {result['VersionId']}")