import boto3
import json
import requests
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from downloader_s3 import download_from_s3, upload_to_s3

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
        """
        if self.s3:
            data = self.download_json_from_s3()
        else:
            data = self.read_local_json()

        updated_data = self.add_html_content_to_dict(data)

        if self.s3:
            self.upload_json_to_s3(updated_data)
        else:
            self.save_local_json(updated_data)

    def enrich_dictionary(self, data_dict: dict) -> dict:
        """
        Enrich a given dictionary by adding HTML content for each URL.

        Args:
            data_dict (dict): Existing dictionary with URLs as keys.

        Returns:
            dict: Updated dictionary with HTML content added.
        """
        return self.add_html_content_to_dict(data_dict)

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

#     # For enriching a separate dictionary
#     separate_dict = {
#         '1': {'url': 'http://example.com', 'last_updated': '2024-07-18'},
#         '2': {'url': 'http://example.org', 'last_updated': '2024-07-19'}
#     }
#     enriched_dict = ContentEnricher().enrich_dictionary(separate_dict)
#     print(enriched_dict)