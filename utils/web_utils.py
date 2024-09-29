import requests
import logging
from typing import Optional
import pandas as pd

def get_html_content(url: str) -> Optional[str]:
    try:
        response = requests.get(url)
        return response.text
    except requests.RequestException:
        logging.error(f"Failed to retrieve HTML content from {url}")
        return None

def add_html_content_to_df(df: pd.DataFrame) -> pd.DataFrame:
    total_urls = len(df)
    html_contents = []
    for i, row in df.iterrows():
        url = row['url']
        logging.info(f"Processing URL {i+1}/{total_urls}: {url}")
        html_content = get_html_content(url)
        html_contents.append(html_content)
    
    df['html_content'] = html_contents
    return df