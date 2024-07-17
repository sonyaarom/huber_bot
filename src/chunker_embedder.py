
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from downloader_s3 import upload_to_s3, download_from_s3,upload_csv_to_s3
from html_retriever_cleaner import HTMLCleaner


BUCKET_NAME = 'hu-chatbot-schema'
S3_KEY = 'csv_files/cleaned/data.csv'

# Download the JSON data
file  = download_from_s3(BUCKET_NAME, S3_KEY)



processor = HTMLCleaner(content)
data = processor.process_data()


embed_model = HuggingFaceEmbedding()


def chunk_text(text, max_length=500, overlap=100, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", " ", ".", ",", ""]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        # Adjust end position to the nearest separator before the max_length
        for sep in separators:
            sep_index = text.rfind(sep, start, end)
            if sep_index != -1:
                end = sep_index + len(sep)
                break
        # If no separator found or single word exceeds max_length
        if end == start:
            end = start + max_length
        chunks.append(text[start:end])
        # Move start position to the start of the next chunk, considering overlap and separators
        next_sep_index = -1
        for sep in separators:
            sep_index = text.find(sep, start + max_length - overlap, end)
            if sep_index != -1:
                next_sep_index = sep_index
                break
        start = next_sep_index + len(sep) if next_sep_index != -1 else min(start + max_length - overlap, len(text))
    return chunks




data['chunk'] = data['extracted_texts'].apply(chunk_text)

chunk_identified = [f"{index}:{chunk}" for index, chunks in zip(data['index'], data['chunk']) for chunk in chunks]
meta = [{'index': index, 'url': url, 'last_updated': date} for index, url, date in zip(data['index'], data['url'], data['last_updated'])]




def expand_dataframe_with_embeddings(data, embed_model):
    """
    Expands the DataFrame by creating new rows for each chunk and embedding.
    
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
            embedding = embed_model.get_text_embedding(chunk)
            new_rows.append({
                'index': row['index'],
                'url': row['url'],
                'last_updated': row['last_updated'],
                'html_content': row['html_content'],
                'extracted_texts': row['extracted_texts'],
                'len': len(chunk),
                'chunk': chunk,
                'embedding': embedding
            })

    # Create a new DataFrame from the new rows
    expanded_df = pd.DataFrame(new_rows)
    return expanded_df


merged_df = merged_df[~merged_df['values'].apply(lambda x: any(pd.isna(v) for v in x))]
merged_df['url_count'] = merged_df.groupby('url').cumcount() + 1
merged_df['unique_id'] = merged_df.apply(lambda row: f"{row['id']}_{row['url_count']}", axis=1)



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
    df['unique_id'] = df.apply(lambda row: f"{row['url']}_{row['url_count']}", axis=1)
    
    # Generate documents list
    documents = []
    for _, row in df.iterrows():
        document = {
            "id": row["unique_id"],
            'values': row["values"],
            "metadata": {
                "url": row["url"],
                "date": row["last_updated"],
                "text": row["text"],
                'general_id': row['id']
            }
        }
        documents.append(document)
    
    return documents

# Upload documents in batches
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    index.upsert(vectors=batch)

# Example usage
json_content = [