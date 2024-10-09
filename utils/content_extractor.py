import pandas as pd
import logging
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_info(html_content):
    if html_content is None or not isinstance(html_content, str):
        logger.warning("Received invalid HTML content")
        return {
            "title": "Title not found",
            "content": "Main content not found",
            "is_valid": False
        }
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = soup.find('h2', class_='documentFirstHeading')
        if not title:
            title = soup.find('h2')
        title_text = title.text.strip() if title else "Title not found"
        
        # Extract main content
        main_content = soup.find('div', id='parent-fieldname-text')
        if not main_content:
            main_content = soup.find('div', id='content-core')
        
        if main_content:
            # Remove script tags
            for script in main_content(["script", "style"]):
                script.decompose()
            
            # Get text and remove extra whitespace
            content_text = re.sub(r'\s+', ' ', main_content.get_text().strip())
            is_valid = True
        else:
            content_text = "Main content not found"
            is_valid = False
        
        return {
            "title": title_text,
            "content": title_text + " " + content_text,
            "is_valid": is_valid
        }
    except Exception as e:
        logger.error(f"Error extracting info from HTML: {str(e)}")
        return {
            "title": "Error extracting title",
            "content": "Error extracting content",
            "is_valid": False
        }

def add_extracted_content_to_df(df: pd.DataFrame, skip_invalid: bool = False) -> pd.DataFrame:
    # Check for None values in html_content
    null_html_count = df['html_content'].isnull().sum()
    if null_html_count > 0:
        logger.warning(f"Found {null_html_count} rows with null HTML content")
    
    # Apply extract_info function
    extracted_data = df['html_content'].apply(extract_info)
    
    df['extracted_title'] = extracted_data.apply(lambda x: x['title'])
    df['extracted_content'] = extracted_data.apply(lambda x: x['content'])
    df['is_valid'] = extracted_data.apply(lambda x: x['is_valid'])
    df['text'] = df['extracted_title'] + " " + df['extracted_content']
    
    # Log extraction results
    invalid_count = df[~df['is_valid']].shape[0]
    logger.info(f"Extracted content for {len(df)} rows, with {invalid_count} invalid extractions")
    
    if skip_invalid:
        df = df[df['is_valid']]
        logger.info(f"Removed {invalid_count} invalid rows. New DataFrame size: {len(df)}")
    
    return df