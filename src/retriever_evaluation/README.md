# Text Processing and Embedding System

This folder contains a system for processing large text datasets, generating embeddings, and uploading them to Pinecone for efficient retrieval and searching.

## Contents

1. `main_processor.py`: The main script for chunking text, generating embeddings, and uploading to Pinecone.
2. `Makefile.qa`: A Makefile to simplify running the scripts with different options.

## Main Features

- Text chunking with configurable chunk sizes
- Embedding generation using HuggingFaceEmbeddings
- Local storage of processed data in JSON format
- Upload of embeddings and metadata to Pinecone vector database
- Flexible processing pipeline with logging and error handling

## Prerequisites

- Python 3.7+
- Pinecone API key (set as an environment variable)
- Required Python packages (pandas, tqdm, dotenv, pinecone-client, langchain, etc.)

## Setup

1. Ensure you have Python 3.7+ installed on your system.
2. Install the required Python packages (consider adding a `requirements.txt` file for easy installation).
3. Set up your Pinecone API key as an environment variable:
   ```
   export PINECONE_API_KEY='your-api-key-here'
   ```

## Usage

### Running the Main Processor

To run the main processing script:

```
python main_processor.py
```

This will:
- Load data from the specified CSV file
- Process the data with multiple chunk sizes (128, 256, 512, 1024 by default)
- Generate embeddings for each chunk
- Save processed data locally
- Upload the data to Pinecone

### Using the Makefile

The `Makefile.qa` provides shortcuts for common tasks. To use it:

```
make -f Makefile.qa [target]
```

Available targets:
- `run`: Run the main processing script
- `clean`: Remove generated output files
- `help`: Display information about available targets

For example:
```
make -f Makefile.qa run
```

## Configuration

You can modify the following parameters in the `main_processor.py` script:

- `df_path`: Path to the input CSV file
- `chunk_lengths`: List of chunk sizes to process
- `embed_model_name`: Name of the embedding model to use
- `docs_path`: Path to save/load processed documents

## Note

Remember to always use `-f Makefile.qa` when running make commands, as the Makefile is not named with the default "Makefile" name.