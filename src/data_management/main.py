import sys
import os
import boto3
import json
from datetime import datetime
import pandas as pd


sys.path.append('/Users/s.konchakova/Thesis/src')
sys.path.append('/Users/s.konchakova/Thesis/utils')

from sitemap_processor import process_sitemap
from web_utils import add_html_content_to_df
from content_extractor import add_extracted_content_to_df
from download_funcs import upload_to_s3


def main():
    # Sitemap processing configuration
    url = 'https://www.wiwi.hu-berlin.de/sitemap.xml.gz'
    exclude_extensions = ['.jpg', '.pdf', '.jpeg', '.png']
    exclude_patterns = ['view']
    include_patterns = ['/en/']
    allowed_base_url = 'https://www.wiwi.hu-berlin.de'
    
    # Process sitemap
    data_dict, total, filtered, safe, unsafe = process_sitemap(
        url, exclude_extensions, exclude_patterns, include_patterns, allowed_base_url
    )
    
    print(f"Total entries: {total}")
    print(f"Filtered entries: {filtered}")
    print(f"Safe entries: {safe}")
    print(f"Unsafe entries: {unsafe}")
    
    # S3 configuration
    bucket_name = 'hu-chatbot-schema'
    s3_key_prefix = f'sitemap_data/sitemap_data_{datetime.now().strftime("%Y")}.json'
    
    # Convert data_dict to JSON string
    json_data = json.dumps(data_dict)
    
    # Upload JSON data to S3
    if bucket_name:
        s3 = boto3.client('s3')
        s3.put_object(Body=json_data, Bucket=bucket_name, Key=s3_key_prefix)
        print(f"Data uploaded to S3 bucket: {bucket_name}")
        print(f"S3 key: {s3_key_prefix}")
    
    # Process the data
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    df.columns = ['id', 'url', 'last_updated']
    
    # Take a sample for testing (remove this line for full processing)
    df = df.head(5)
    
    # Add HTML content to the DataFrame
    df = add_html_content_to_df(df)
    
    # Extract and add content to the DataFrame
    df = add_extracted_content_to_df(df)
    
    # Display results
    print(df[['url', 'extracted_title', 'extracted_content']])
    
    # TODO: Add code to save the processed DataFrame back to S3 if needed

if __name__ == "__main__":
    main()

