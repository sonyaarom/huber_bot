import boto3
import requests
import os
from urllib.parse import urlparse

def download_and_store_in_s3(url, bucket_name, s3_key_prefix=''):
    """
    Downloads a file from the given URL and stores it in an S3 bucket with versioning.

    Args:
        url (str): The URL of the file to download.
        bucket_name (str): The name of the S3 bucket.
        s3_key_prefix (str, optional): The S3 key prefix (path) to store the file under. Defaults to ''.

    Returns:
        dict: Information about the uploaded file including version ID.
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
        
        # Upload the file to S3
        s3_client = boto3.client('s3')
        s3_key = os.path.join(s3_key_prefix, file_name)
        with open(temp_file_path, 'rb') as temp_file:
            response = s3_client.upload_fileobj(temp_file, bucket_name, s3_key)
        
        # Get the version ID of the uploaded file
        head_response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        version_id = head_response.get('VersionId', 'null')

        # Clean up the temporary file
        os.remove(temp_file_path)

        return {
            'Bucket': bucket_name,
            'Key': s3_key,
            'VersionId': version_id
        }
    else:
        raise Exception(f"Failed to download file from URL. Status code: {response.status_code}")

# Example usage
if __name__ == "__main__":
    url = 'https://www.wiwi.hu-berlin.de/sitemap.xml.gz'
    bucket_name = 'hu-chatbot-schema'
    s3_key_prefix = 'gz_files'

    result = download_and_store_in_s3(url, bucket_name, s3_key_prefix)
    print(f"File uploaded to S3 with version ID: {result['VersionId']}")