from bs4 import BeautifulSoup
import re
import pandas as pd

def extract_info(html_content):
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
    else:
        content_text = "Main content not found"
    
    return {
        "title": title_text,
        "content": title_text + " " + content_text
    }

def add_extracted_content_to_df(df: pd.DataFrame) -> pd.DataFrame:
    extracted_data = df['html_content'].apply(extract_info)
    
    df['extracted_title'] = extracted_data.apply(lambda x: x['title'])
    df['extracted_content'] = extracted_data.apply(lambda x: x['content'])
    
    return df