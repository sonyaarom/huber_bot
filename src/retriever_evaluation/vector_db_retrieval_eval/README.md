# Pinecone Retrieval Evaluation

This part of the project provides a flexible framework for evaluating and benchmarking retrieval performance using Pinecone vector database. It supports both dense and hybrid (dense + sparse) search methods, with options for named entity recognition (NER) filtering and reranking.

## Project Structure

```
pinecone_retrieval_evaluation/
│
├── main.py
├── config.py
├── utils.py
├── models.py
├── evaluation.py
├── requirements.txt
└── README.md
```

## Dependencies

This project requires the following main libraries:

- pandas
- pinecone-client
- sentence-transformers
- wandb (Weights & Biases)
- gliner
- python-dotenv

For a complete list of dependencies, refer to the `requirements.txt` file.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables by creating a `.env` file in the project root:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   WANDB_API_KEY=your_wandb_api_key
   ```

## Configuration

Update the `config.py` file with your specific settings:

- `API_CONFIGS`: Configuration for different Pinecone API keys and their default embedding models
- `WANDB_PROJECT` and `WANDB_ENTITY`: Your Weights & Biases project and entity names
- `QA_DF_PATH`: Path to your question-answering dataset
- `BM25_VALUES_PATH`: Path to pre-computed BM25 values
- `DEFAULT_ALPHA_VALUES`: Default alpha values for hybrid search
- `DEFAULT_RERANKER_MODEL`: Default reranker model name
- `NER_MODEL` and `NER_LABELS`: NER model and labels for entity recognition

## Usage

Run the main script to start the evaluation:

```
python main.py
```

Follow the prompts to configure your evaluation:

1. Choose whether to use a reranker
2. Decide if you want to use NER for filtering
3. Select the search type (dense or hybrid)
4. Specify whether to trust remote code for embedding models
5. Enter the number of initial and final results to retrieve

The script will then run the evaluation across all configured Pinecone indexes and log the results.

## Components

- `main.py`: The entry point of the application, orchestrating the evaluation process
- `config.py`: Contains configuration variables and settings
- `utils.py`: Utility functions for data loading, vector conversion, etc.
- `models.py`: Defines wrapper classes for Pinecone and CrossEncoder models
- `evaluation.py`: Contains the core evaluation logic and metrics calculation

## Evaluation Metrics

The evaluation process calculates the following metrics:

- Mean Reciprocal Rank (MRR)
- Average Retrieval Time
- Hit@k (for k=1, 3, 5 by default, configurable)

## Logging and Visualization

This project uses Weights & Biases (wandb) for logging and visualizing results. After each evaluation run, you can view detailed metrics, comparisons, and visualizations in your wandb project dashboard.

---

For any questions or issues, please open an issue on the GitHub repository or contact the project maintainers.