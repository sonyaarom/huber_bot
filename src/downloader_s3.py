import boto3
import json
import requests


class S3Downloader:
    def __init__(self, bucket_name, json_key):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.json_key = json_key

    def download_json(self):
        response = self.s3.get_object(Bucket=self.bucket_name, Key=self.json_key)
        file_content = response['Body'].read().decode('utf-8')
        return json.loads(file_content)

    def get_html_content(self, url):
        try:
            response = requests.get(url)
            html_content = response.text
            return html_content
        except:
            return None

    def add_html_content_to_dict(self, existing_dict: dict) -> dict:
        """
        Adds HTML content for each URL in the existing dictionary.

        Args:
            existing_dict (dict): Existing dictionary with URLs as keys.

        Returns:
            dict: Updated dictionary with HTML content added.
        """
        updated_dict = {}
        for url_hash, data in existing_dict.items():
            url = data['url']
            date = data['last_updated']
            html_content = self.get_html_content(url)
            updated_dict[url_hash] = {
                'url': url,
                'last_updated': date,
                'html_content': html_content
            }
        return updated_dict

    def process_json_file(self):
        # Download the JSON file from S3
        data = self.download_json()
        
        # Add HTML content to the dictionary
        updated_data = self.add_html_content_to_dict(data)
        print(updated_data[:2])

# Usage example
if __name__ == "__main__":
    bucket_name = 'hu-chatbot-schema'
    json_key = 'json_files/all_matches.json'
    updated_json_key = 'json_files/updated_all_matches.json'

    processor = S3Downloader(bucket_name, json_key)
    processor.process_json_file()