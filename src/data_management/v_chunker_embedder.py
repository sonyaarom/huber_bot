from langchain_community.embeddings import HuggingFaceEmbeddings

import pandas as pd
import sys
import os
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))


from downloader_s3 import upload_to_s3, download_from_s3,upload_csv_to_s3, download_csv_from_s3


BUCKET_NAME = 'hu-chatbot-schema'
S3_KEY = 'csv_files/cleaned/data.csv'



# download the csv from the S3

data = download_csv_from_s3(BUCKET_NAME, S3_KEY)

embed_model = HuggingFaceEmbeddings()

def chunk_text(text, max_length=450, overlap=100, separators=None):
    """
    Splits the input text into chunks of specified maximum length with optional overlap.

    Args:
        text (str): The input text to be chunked.
        max_length (int): The maximum length of each chunk. Default is 500 characters.
        overlap (int): The number of characters that overlap between chunks. Default is 100 characters.
        separators (list, optional): A list of separator strings to split the text on. Default is 
                                     ["\n\n", "\n", " ", ".", ",", ""].
    
    Returns:
        List[str]: A list of text chunks.
    """

    
    # Set default separators if none are provided
    if separators is None:
        separators = ["\n\n", "\n", " ", ".", ",", ""]
    
    chunks = []  # List to hold the text chunks
    start = 0  # Initial start position of the chunk
    
    # Loop until the entire text is processed
    while start < len(text):
        end = min(start + max_length, len(text))  # Calculate the end position of the chunk
        
        # Adjust the end position to the nearest separator before the max_length
        for sep in separators:
            sep_index = text.rfind(sep, start, end)
            if sep_index != -1:
                end = sep_index + len(sep)
                break
        
        # If no separator found or single word exceeds max_length
        if end == start:
            end = start + max_length
        
        chunks.append(text[start:end])  # Add the chunk to the list
        
        # Move the start position to the start of the next chunk, considering overlap and separators
        next_sep_index = -1
        for sep in separators:
            sep_index = text.find(sep, start + max_length - overlap, end)
            if sep_index != -1:
                next_sep_index = sep_index
                break
        
        # Update the start position for the next chunk
        start = next_sep_index + len(sep) if next_sep_index != -1 else min(start + max_length - overlap, len(text))
    
    return chunks  # Return the list of chunks



def expand_dataframe_with_embeddings(data, embed_model):
    """
    Expands the DataFrame by creating new rows for each chunk and embedding.
    Removes broken embeddings, e.g. when the english version of the website had German text.
    
    Args:
        data (pd.DataFrame): The input DataFrame with a 'chunk' column containing lists of text chunks.
        embed_model: The embedding model to generate embeddings for the text chunks.
    
    Returns:
        pd.DataFrame: A new DataFrame with each chunk and its embedding as separate rows.
    """
    new_rows = []

    # Iterate over the DataFrame
    for _, row in data.iterrows():
        for chunk in row['chunk']:
            # Ensure chunk is treated as a string, not as a list
            chunk_str = chunk if isinstance(chunk, str) else ' '.join(chunk)
            embedding = embed_model.embed_documents([chunk_str])
            new_rows.append({
                'id': row['id'],
                'url': row['url'],
                'last_updated': row['last_updated'],
                'html_content': row['html_content'],
                'text': chunk_str,  # Ensure text is stored as a string
                'len': len(chunk_str),
                'embedding': embedding
            })

    # Create a new DataFrame from the new rows
    expanded_df = pd.DataFrame(new_rows)

    #expanded_df = expanded_df[~expanded_df['embedding'].apply(lambda x: any(pd.isna(v) for v in x))]
    return expanded_df


def generate_documents(df):
    """
    Processes a dataframe to generate a list of documents with unique identifiers.

    Parameters:
    df (pandas.DataFrame): A dataframe containing the following columns:
        - url: The URL which can repeat multiple times.
        - values: The values associated with each row.
        - last_updated: The date when the information was last updated.
        - text: The text content related to each URL.
        - id: A general identifier for each row.

    Steps:
    1. Create Incremental Count for Each URL:
       - Uses groupby and cumcount to generate an incremental count for each occurrence of a URL.
    
    2. Generate Unique Identifier:
       - Combines the URL and its incremental count to create a unique identifier for each row.

    3. Construct Documents:
       - Iterates over each row of the dataframe.
       - For each row, creates a document with the following structure:
         - id: The unique identifier.
         - values: The value from the values column.
         - metadata: A dictionary containing:
           - url: The URL.
           - date: The date from the last_updated column.
           - text: The text content.
           - general_id: The general identifier from the id column.

    Returns:
    List[dict]: A list of dictionaries, where each dictionary represents a document with the specified structure.
    """
    
    # Create the incremental count for each URL
    df['url_count'] = df.groupby('url').cumcount() + 1

    # Generate unique identifier by combining URL and count
    df['unique_id'] = df.apply(lambda row: f"{row['id']}_{row['url_count']}", axis=1)
    
    # Generate documents list
    documents = []
    for _, row in df.iterrows():
        document = {
            "id": row["unique_id"],
            'values': row["embedding"],
            "metadata": {
                "url": row["url"],
                "date": row["last_updated"],
                "text": row["text"],
                'general_id': row['id']
            }
        }
        documents.append(document)
    
    return documents





#EXAMPLE USE
#data['text'] = data['text'].apply(str)
#data['chunk'] = data['text'].apply(chunk_text)
#new_data = expand_dataframe_with_embeddings(data, embed_model)
#document = generate_documents(new_data)
#upload_csv_to_s3(BUCKET_NAME,'csv_files/embeddings/embedded_data.csv', new_data)
