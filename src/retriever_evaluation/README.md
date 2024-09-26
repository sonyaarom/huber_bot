# Retriever Evaluation

This project provides a comprehensive framework for evaluating and benchmarking retrieval systems, as well as generating and evaluating question-answer pairs. It consists of two main components:

1. Generation of QA couples for further Evaluation
1. Vector Database Retrieval Evaluation (using Pinecone and QA couples)


## Project Structure

```
retriever_evaluation/
│
├── vector_db_retrieval_eval/
│   ├── main.py
│   ├── config.py
│   ├── utils.py
│   ├── models.py
│   └── evaluation.py
│
├── generate_evaluate_questions.py
│
├── requirements.txt
└── README.md
```

## Dependencies

This project requires several libraries, including:

- pandas
- numpy
- pinecone-client
- sentence-transformers
- wandb (Weights & Biases)
- openai
- langchain
- tqdm
- python-dotenv

For a complete list of dependencies, refer to the `requirements.txt` file.

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables by creating a `.env` file in the project root:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   WANDB_API_KEY=your_wandb_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Configuration

### Vector DB Retrieval Evaluation

Update the `config.py` file in the `vector_db_retrieval_eval` directory with your specific settings:

- `API_CONFIGS`: Configuration for different Pinecone API keys and their default embedding models
- `WANDB_PROJECT` and `WANDB_ENTITY`: Your Weights & Biases project and entity names
- `QA_DF_PATH`: Path to your question-answering dataset
- `BM25_VALUES_PATH`: Path to pre-computed BM25 values
- `DEFAULT_ALPHA_VALUES`: Default alpha values for hybrid search
- `DEFAULT_RERANKER_MODEL`: Default reranker model name
- `NER_MODEL` and `NER_LABELS`: NER model and labels for entity recognition

### Question Generation and Evaluation

The question generation script uses command-line arguments for configuration. You can modify the default values in the script or provide them when running the script.

## Usage

### Vector DB Retrieval Evaluation

Run the main script to start the evaluation:

```
python vector_db_retrieval_eval/main.py
```

Follow the prompts to configure your evaluation:

1. Choose whether to use a reranker
2. Decide if you want to use NER for filtering
3. Select the search type (dense or hybrid)
4. Specify whether to trust remote code for embedding models
5. Enter the number of initial and final results to retrieve

### Question Generation and Evaluation

Run the question generation and evaluation script:

```
python question_generation/generate_evaluate_questions.py --input path/to/input.csv --output path/to/output.csv --max_questions 5 --sample 1000
```

Arguments:
- `--input`: Path to the input CSV file (default: "../../assets/csv/data_full.csv")
- `--output`: Path to the output CSV file (default: "../../assets/csv/evaluated_questions_with_scores.csv")
- `--max_questions`: Maximum number of questions to generate (default: 5)
- `--sample`: Number of rows to sample from the input data (0 means no sampling, default: 0)

## Components

### Vector DB Retrieval Evaluation

- `main.py`: The entry point of the application, orchestrating the evaluation process
- `config.py`: Contains configuration variables and settings
- `utils.py`: Utility functions for data loading, vector conversion, etc.
- `models.py`: Defines wrapper classes for Pinecone and CrossEncoder models
- `evaluation.py`: Contains the core evaluation logic and metrics calculation

### Question Generation and Evaluation

- `generate_evaluate_questions.py`: Generates questions from input text and evaluates their quality using OpenAI's API

## Evaluation Metrics

### Vector DB Retrieval Evaluation

- Mean Reciprocal Rank (MRR)
- Average Retrieval Time
- Hit@k (for k=1, 3, 5 by default, configurable)

### Question Generation and Evaluation

- Specificity (1-5 scale)
- Realism (1-5 scale)
- Clarity (1-5 scale)
- Average Score

## Logging and Visualization

The Vector DB Retrieval Evaluation component uses Weights & Biases (wandb) for logging and visualizing results. After each evaluation run, you can view detailed metrics, comparisons, and visualizations in your wandb project dashboard.

The Question Generation and Evaluation component logs its progress and results to the console and saves the evaluated questions with scores to a CSV file.

---

For any questions or issues, please open an issue on the GitHub repository or contact the project maintainers.