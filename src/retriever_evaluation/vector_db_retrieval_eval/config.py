import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configurations
API_CONFIGS = {
    # "all-mini-eucl": {
    #     "api_key": '2e55369e-62f2-4317-8273-faf282511d2b',
    #     "default_embedding_model": "sentence-transformers/all-mpnet-base-v2"
    # },
    "all-mini-cosine": {
        "api_key": '2966f5ad-7763-45be-a7ab-7c24afc01ff7',
        "default_embedding_model": "all-MiniLM-L6-v2"
    # },
    # "all-mini-dotproduct": {
    #     "api_key": 'b4f8e710-fc88-4b6a-a2b8-433f60359b10',
    #     "default_embedding_model": "all-MiniLM-L6-v2"
    }

    
}

# File paths
QA_DF_PATH = '/Users/s.konchakova/Thesis/assets/csv/qa_df.csv'
BM25_VALUES_PATH = '/Users/s.konchakova/Thesis/assets/bm25_values.json'

# NER settings
NER_MODEL = "urchade/gliner_small-v2.1"
NER_LABELS = ["person", "course", "date", "research_paper", "research_project", "teams", "city", "address", "organisation", "phone_number", "url", "other"]

# Wandb settings
WANDB_PROJECT = "pinecone_index_evaluation"
WANDB_ENTITY = None

# Default search settings
DEFAULT_K_VALUES = [1, 3, 5]
DEFAULT_ALPHA_VALUES = [1]
DEFAULT_INITIAL_K = 10
DEFAULT_FINAL_K = 5

# Reranker model
DEFAULT_RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'