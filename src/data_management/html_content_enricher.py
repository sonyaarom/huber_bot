import boto3
import json
import requests
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from downloader_s3 import download_from_s3, upload_to_s3

import logging

# Set up logging at the beginning of the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContentEnricher:
    """
    A class to enrich dictionaries with HTML content from URLs and handle JSON files locally or on S3.
    """

    def __init__(self, input_source=None, bucket_name=None, json_key=None, updated_json_key=None):
        """
        Initialize the ContentEnricher with optional S3 or local file parameters.

        Args:
            input_source (str): Path to the local JSON file.
            bucket_name (str): Name of the S3 bucket.
            json_key (str): Key for the JSON file in the S3 bucket.
            updated_json_key (str): Key for the updated JSON file in the S3 bucket.
        """
        self.input_source = input_source
        self.bucket_name = bucket_name
        self.json_key = json_key
        self.updated_json_key = updated_json_key
        self.s3 = boto3.client('s3') if bucket_name else None
        
    def download_json_from_s3(self):
        """
        Download a JSON file from an S3 bucket.

        Returns:
            dict: JSON data from the S3 bucket.
        """
        response = download_from_s3(self.bucket_name, self.json_key)
        file_content = response['Body'].read().decode('utf-8')
        return json.loads(file_content)

    def read_local_json(self):
        """
        Read a local JSON file.

        Returns:
            dict: JSON data from the local file.
        """
        with open(self.input_source, 'r', encoding='utf-8') as file:
            return json.load(file)

    def upload_json_to_s3(self, data):
        """
        Upload a JSON file to an S3 bucket.

        Args:
            data (dict): JSON data to upload.
        """
        json_data = json.dumps(data, indent=2)
        upload_to_s3(self.bucket_name, self.updated_json_key, json_data)
        print(f"Updated JSON saved to {self.updated_json_key} in bucket {self.bucket_name}")

    def save_local_json(self, data):
        """
        Save JSON data to a local file.

        Args:
            data (dict): JSON data to save.
        """
        with open(self.input_source, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
        print(f"Updated JSON saved to {self.input_source}")

    def get_html_content(self, url):
        """
        Retrieve HTML content from a given URL.

        Args:
            url (str): URL to fetch HTML content from.

        Returns:
            str: HTML content of the URL or None if the request fails.
        """
        try:
            response = requests.get(url)
            html_content = response.text
            return html_content
        except requests.RequestException:
            return None

    def add_html_content_to_dict(self, data_dict: dict) -> dict:
        """
        Add HTML content for each URL in the existing dictionary.

        Args:
            data_dict (dict): Existing dictionary with URLs as keys.

        Returns:
            dict: Updated dictionary with HTML content added.
        """
        updated_dict = {}
        for url_hash, data in data_dict.items():
            url = data['url']
            date = data['last_updated']
            html_content = self.get_html_content(url)
            updated_dict[url_hash] = {
                'url': url,
                'last_updated': date,
                'html_content': html_content
            }
        return updated_dict

    def process_json(self):
        """
        Process the JSON file, enriching it with HTML content for each URL, and save it back.
        
        Returns:
            dict: The enriched dictionary
        """
        logging.info("Starting JSON processing")
        
        if self.s3:
            logging.info("Downloading JSON from S3")
            data = self.download_json_from_s3()
        else:
            logging.info("Reading local JSON file")
            data = self.read_local_json()

        logging.info("Enriching dictionary with HTML content")
        updated_data = self.add_html_content_to_dict(data)

        # Use the new method to save to both S3 and local
        local_save_path = self.input_source or "enriched_data.json"
        self.save_enriched_dictionary(updated_data, local_save_path)
        
        logging.info("JSON processing completed")
        return updated_data

    def save_enriched_dictionary(self, data: dict, local_path: str) -> None:
        """
        Save the enriched dictionary to both S3 (if configured) and local storage.

        Args:
            data (dict): The enriched dictionary to save.
            local_path (str): The local file path to save the JSON data.
        """
        # Save to S3 if configured
        if self.s3 and self.bucket_name and self.updated_json_key:
            logging.info(f"Uploading enriched JSON to S3: {self.bucket_name}/{self.updated_json_key}")
            self.upload_json_to_s3(data)
        
        # Save locally
        logging.info(f"Saving enriched JSON locally to {local_path}")
        with open(local_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
        logging.info(f"Enriched dictionary saved locally to {local_path}")

    def add_html_content_to_dict(self, data_dict: dict) -> dict:
        """
        Add HTML content for each URL in the existing dictionary.

        Args:
            data_dict (dict): Existing dictionary with URLs as keys.

        Returns:
            dict: Updated dictionary with HTML content added.
        """
        updated_dict = {}
        total_urls = len(data_dict)
        for i, (url_hash, data) in enumerate(data_dict.items(), 1):
            url = data['url']
            date = data['last_updated']
            logging.info(f"Processing URL {i}/{total_urls}: {url}")
            html_content = self.get_html_content(url)
            updated_dict[url_hash] = {
                'url': url,
                'last_updated': date,
                'html_content': html_content
            }
        return updated_dict
    
    def save_enriched_dictionary(self, data: dict, local_path: str) -> None:
        """
        Save the enriched dictionary to both S3 (if configured) and local storage.

        Args:
            data (dict): The enriched dictionary to save.
            local_path (str): The local file path to save the JSON data.
        """
        # Save to S3 if configured
        if self.s3 and self.bucket_name and self.updated_json_key:
            logging.info(f"Uploading enriched JSON to S3: {self.bucket_name}/{self.updated_json_key}")
            self.upload_json_to_s3(data)
        
        # Save locally
        logging.info(f"Saving enriched JSON locally to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)  # Ensure the directory exists
        with open(local_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
        logging.info(f"Enriched dictionary saved locally to {local_path}")

    def upload_json_to_s3(self, data):
        """
        Upload a JSON file to an S3 bucket.

        Args:
            data (dict): JSON data to upload.
        """
        json_data = json.dumps(data, indent=2)
        self.s3.put_object(Bucket=self.bucket_name, Key=self.updated_json_key, Body=json_data)
        logging.info(f"Updated JSON uploaded to S3: {self.bucket_name}/{self.updated_json_key}")

    
if __name__ == "__main__":
    # S3 configuration
    bucket_name = 'hu-chatbot-schema'
    json_key = 'json_files/all_matches.json'
    updated_json_key = 'json_files/updated_all_matches.json'
    
    # Local configuration
    local_input_path = 'assets/json_files/all_matches.json'
    local_output_path = 'assets/json_files/enriched_all_matches.json'
    
    logging.info("Initializing ContentEnricher")
    processor = ContentEnricher(input_source=local_input_path, 
                                bucket_name=bucket_name, 
                                json_key=json_key, 
                                updated_json_key=updated_json_key)
    
    enriched_dict = processor.process_json()
    
    # Save the enriched dictionary locally and to S3
    processor.save_enriched_dictionary(enriched_dict, local_output_path)
    
    logging.info("Processing completed. Enriched dictionary is available.")
    logging.info(f"Total number of processed items: {len(enriched_dict)}")

# Usage example
# if __name__ == "__main__":
#     # For S3 processing
#     bucket_name = 'hu-chatbot-schema'
#     json_key = 'json_files/all_matches.json'
#     updated_json_key = 'json_files/updated_all_matches.json'
#     processor_s3 = ContentEnricher(bucket_name=bucket_name, json_key=json_key, updated_json_key=updated_json_key)
#     processor_s3.process_json()

#     # For local file processing
#     local_file_path = 'path/to/local/json_file.json'
#     processor_local = ContentEnricher(input_source=local_file_path)
#     processor_local.process_json()


#     enriched_dict = ContentEnricher().enrich_dictionary(separate_dict)
#     print(enriched_dict)