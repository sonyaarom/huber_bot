import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from pathlib import Path
from datetime import datetime
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from downloader_s3 import upload_to_s3, download_from_s3,upload_csv_to_s3


# Download the JSON data


class HTMLCleaner:
    def __init__(self, json_content):
        self.json_content = json_content

    def extract_and_clean_text(self, html_content) -> str:
        """
        Extracts text content under HTML tags with id="parent-fieldname-text" and cleans HTML tags.

        Args:
            html_content (str): HTML content.

        Returns:
            str: Cleaned text content without HTML tags.
        """
        if html_content is None or not html_content.strip():
            return ""  # Return empty string if input is None or empty

        try:
            # Extract text content
            soup = BeautifulSoup(html_content, 'html.parser')
            parent_fieldname_texts = soup.find_all('div', id='parent-fieldname-text')
            extracted_text = ' '.join(tag.get_text().strip() for tag in parent_fieldname_texts)

            # Clean HTML tags from extracted text
            clean_text = BeautifulSoup(extracted_text, "html.parser").get_text()
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            return clean_text
        except Exception as e:
            print("Error occurred while extracting and cleaning HTML content:", e)
            return ""  # Return empty string if an error occurs.
        
    def convert_to_date(self, datetime_string):
        datetime_obj = datetime.strptime(datetime_string, "%Y-%m-%dT%H:%M:%S%z")
        return datetime_obj.strftime("%Y-%m-%d")

    def process_data(self):
        # Convert the JSON content into a DataFrame

        data_dict = json.loads(self.json_content)
        data = pd.DataFrame(data_dict).transpose()
        data.index.name = "id"
        data.reset_index(inplace=True)

        # Ensure the 'html_content' and 'last_updated' columns exist
        if 'html_content' not in data.columns or 'last_updated' not in data.columns:
            raise ValueError("JSON content must include 'html_content' and 'last_updated' fields")

        # Apply the extraction and cleaning function to the 'html_content' column
        data['extracted_texts'] = data['html_content'].apply(self.extract_and_clean_text)
        
        # Convert 'last_updated' to a readable date format
        data['last_updated'] = data['last_updated'].apply(self.convert_to_date)
        
        # Calculate the length of the extracted text
        data['len'] = data['extracted_texts'].apply(len)
        
        return data





if __name__ == "__main__":

    BUCKET_NAME = 'hu-chatbot-schema'
    JSON_KEY = 'json_files/updated_all_matches.json'
    JSON_KEY_UPLOAD = 'csv_files/cleaned/data.csv' 


    export = download_from_s3(BUCKET_NAME, JSON_KEY)
    json_content = export['Body'].read().decode('utf-8')

    cleaner = HTMLCleaner(json_content)
    processed_data = cleaner.process_data()

    upload_csv_to_s3(BUCKET_NAME, JSON_KEY_UPLOAD, processed_data)   
