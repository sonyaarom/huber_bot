# Text Chunking

This folder provides tools for chunking text data, generating embeddings, and uploading the results to Pinecone for efficient vector search capabilities. It is the next step after data_management folder.

## Project Structure

The project consists of the following main files:

- `main.py`: The entry point of the application, orchestrating the chunking and embedding process.
- `character_chunker.py`: Implements character-based text chunking.
- `recursive_chunker.py`: Implements recursive text chunking using LangChain.
- `semantic_chunker.py`: Implements semantic text chunking using TextTiling.
- `token_chunker.py`: Implements token-based text chunking using tiktoken.
- `shared_utils.py`: Contains shared utility functions used across the project.

## Features

- Multiple text chunking strategies: character-based, recursive, semantic, and token-based.
- Embedding generation using various models (Sentence Transformers, Hugging Face).
- BM25 sparse vector generation for improved search capabilities.
- GliNER for named entity recognition.
- Integration with Pinecone for vector storage and search.
- Flexible configuration options for chunking and embedding processes.

## Prerequisites

- Python 3.7+
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - sentence-transformers
  - langchain
  - pinecone-client
  - tqdm
  - python-dotenv
  - rank-bm25
  - tiktoken
  - llama-index

## Usage

1. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   ```

2. Run the main script with desired options:

```
python main.py
```

You can modify the `main()` function in `main.py` to customize the chunking types, embedding model, and other parameters.

## Configuration

- Chunk sizes and types can be configured in the `main()` function of `main.py`.
- Embedding models can be selected by modifying the `encode_model` parameter in `main()`.
- Pinecone index settings can be adjusted in the `create_pinecone_index()` function in `shared_utils.py`.

## Output

The project generates:
- JSON files containing chunked and embedded documents that are ready to be uploaded to Pinecone.

## Logging

The project uses Python's logging module to provide informative output during execution. Log level and format can be adjusted in `main.py`.

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Create a new Pull Request.