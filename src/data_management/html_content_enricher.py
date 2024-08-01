import boto3
import json
import requests
import os
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_json_from_s3(bucket_name: str, json_key: str) -> Dict:
    """Download a JSON file from an S3 bucket."""
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=json_key)
    file_content = response['Body'].read().decode('utf-8')
    return json.loads(file_content)

def read_local_json(file_path: str) -> Dict:
    """Read a local JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def upload_json_to_s3(bucket_name: str, json_key: str, data: Dict) -> None:
    """Upload a JSON file to an S3 bucket."""
    s3 = boto3.client('s3')
    json_data = json.dumps(data, indent=2)
    s3.put_object(Bucket=bucket_name, Key=json_key, Body=json_data)
    logging.info(f"Updated JSON uploaded to S3: {bucket_name}/{json_key}")

def save_local_json(file_path: str, data: Dict) -> None:
    """Save JSON data to a local file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)
    logging.info(f"JSON saved locally to {file_path}")

def get_html_content(url: str) -> Optional[str]:
    """Retrieve HTML content from a given URL."""
    try:
        response = requests.get(url)
        return response.text
    except requests.RequestException:
        logging.error(f"Failed to retrieve HTML content from {url}")
        return None

def add_html_content_to_dict(data_dict: Dict) -> Dict:
    """Add HTML content for each URL in the existing dictionary."""
    updated_dict = {}
    total_urls = len(data_dict)
    for i, (url_hash, data) in enumerate(data_dict.items(), 1):
        url = data['url']
        date = data['last_updated']
        logging.info(f"Processing URL {i}/{total_urls}: {url}")
        html_content = get_html_content(url)
        updated_dict[url_hash] = {
            'url': url,
            'last_updated': date,
            'html_content': html_content
        }
    return updated_dict

def process_json(input_source: Optional[str] = None, 
                 bucket_name: Optional[str] = None, 
                 json_key: Optional[str] = None, 
                 updated_json_key: Optional[str] = None) -> Dict:
    """Process the JSON file, enriching it with HTML content for each URL, and save it back."""
    logging.info("Starting JSON processing")
    
    if bucket_name and json_key:
        logging.info("Downloading JSON from S3")
        data = download_json_from_s3(bucket_name, json_key)
    elif input_source:
        logging.info("Reading local JSON file")
        data = read_local_json(input_source)
    else:
        raise ValueError("Either S3 details or local file path must be provided")

    logging.info("Enriching dictionary with HTML content")
    updated_data = add_html_content_to_dict(data)

    # Save enriched data
    if bucket_name and updated_json_key:
        upload_json_to_s3(bucket_name, updated_json_key, updated_data)
    if input_source:
        save_local_json(input_source, updated_data)
    
    logging.info("JSON processing completed")
    return updated_data

# if __name__ == "__main__":
#     # S3 configuration
#     bucket_name = 'hu-chatbot-schema'
#     json_key = 'json_files/all_matches.json'
#     updated_json_key = 'json_files/updated_all_matches.json'
    
#     # Local configuration
#     local_input_path = 'assets/json_files/all_matches.json'
#     local_output_path = 'assets/json_files/enriched_all_matches.json'
    
#     enriched_dict = process_json(input_source=local_input_path, 
#                                  bucket_name=bucket_name, 
#                                  json_key=json_key, 
#                                  updated_json_key=updated_json_key)
    
#     # Save the enriched dictionary locally
#     save_local_json(local_output_path, enriched_dict)
    
#     logging.info("Processing completed. Enriched dictionary is available.")
#     logging.info(f"Total number of processed items: {len(enriched_dict)}")