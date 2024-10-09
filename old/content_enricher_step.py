import sys
import pandas as pd
import boto3
sys.path.append('/Users/s.konchakova/Thesis/utils')
sys.path.append('/Users/s.konchakova/Thesis/src')
from download_funcs import download_file_from_s3, upload_to_s3
from web_utils import add_html_content_to_df
from content_extractor import add_extracted_content_to_df

sys.path.append('/Users/s.konchakova/Thesis/src')


def main():
    # Connect to S3 and get the JSON file
    s3_client = boto3.client('s3')
    bucket_name = 'hu-chatbot-schema'
    s3_key_prefix = 'sitemap_data/sitemap_data_2024.json'
    file = download_file_from_s3(bucket_name, s3_key_prefix)

    # Process the data
    df = pd.read_json(file).transpose()
    df = df.reset_index()
    df.columns = ['id', 'url', 'last_updated']

    # Take a sample for testing
    sample = df.head(5)

    # Add HTML content to the DataFrame
    sample = add_html_content_to_df(sample)

    # Extract and add content to the DataFrame
    sample = add_extracted_content_to_df(sample)

    # Display results
    print(sample[['url', 'extracted_title', 'extracted_content']])

    # Upload the enriched data to S3
    enriched_s3_key_prefix = 'enriched_data/enriched_data_2024.json'
    upload_to_s3(bucket_name, enriched_s3_key_prefix, sample.to_json(orient='records'))

if __name__ == "__main__":
    main()
