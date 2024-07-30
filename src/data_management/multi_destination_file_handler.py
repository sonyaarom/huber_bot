import sys
import os
import requests
import boto3
from urllib.parse import urlparse


# Ensure the script directory is in the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
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

def store_file(temp_file_path, local_directory=None, bucket_name=None, s3_key_prefix=''):
    """
    Stores a file in a local directory and/or S3 bucket.

    Args:
        temp_file_path (str): The path to the temporary file.
        local_directory (str, optional): The local directory to store the file. If None, file is not stored locally.
        bucket_name (str, optional): The name of the S3 bucket. If None, file is not uploaded to S3.
        s3_key_prefix (str, optional): The S3 key prefix (path) to store the file under. Defaults to ''.

    Returns:
        dict: Information about the stored file.
    """
    file_name = os.path.basename(temp_file_path)
    result = {}

    # Store locally if a directory is provided
    if local_directory:
        local_path = os.path.join(local_directory, file_name)
        os.makedirs(local_directory, exist_ok=True)
        with open(temp_file_path, 'rb') as src_file, open(local_path, 'wb') as dst_file:
            dst_file.write(src_file.read())
        result['LocalPath'] = local_path

    # Upload to S3 if a bucket name is provided
    if bucket_name:
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

# Example usage
# if __name__ == "__main__":
#     url = 'https://www.wiwi.hu-berlin.de/sitemap.xml.gz'
#     local_directory = 'assets/gz_files'
#     bucket_name = 'hu-chatbot-schema'
#     s3_key_prefix = 'gz_files'

#     # Step 1: Download the file
#     temp_file_path = download_file(url)
    
#     # Step 2: Store the file locally and/or upload to S3
#     result = store_file(temp_file_path, local_directory, bucket_name, s3_key_prefix)
    
#     if 'LocalPath' in result:
#         print(f"File stored locally at: {result['LocalPath']}")
#     if 'VersionId' in result:
#         print(f"File uploaded to S3 with version ID: {result['VersionId']}")