import sys
import os
import boto3
import json
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

sys.path.append('/Users/s.konchakova/Thesis/huber_bot/src')
sys.path.append('/Users/s.konchakova/Thesis/huber_bot/utils')


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
    
    # Process the data
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    df.columns = ['id', 'url', 'last_updated']

    # df = df.head(10)
    
    # Add HTML content to the DataFrame
    df = add_html_content_to_df(df)
    
    # Extract and add content to the DataFrame
    df = add_extracted_content_to_df(df, skip_invalid = True)
    logger.info(f"Final DataFrame size after extraction: {df.size}")
    #save to csv
    df.to_csv('assets/csv/data_subset.csv', index=False)

    # Display results
    print(df[['url', 'extracted_title', 'extracted_content']])
    
    # TODO: Add code to save the processed DataFrame back to S3 if needed

if __name__ == "__main__":
    main()

