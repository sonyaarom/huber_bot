import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configurations
API_CONFIGS = {
    "PROJECT_NAME": {
        "api_key": API_KEY,
        "default_embedding_model": "sentence-transformers/all-mpnet-base-v2"
    },
    
}

# File paths
QA_DF_PATH = ''
BM25_VALUES_PATH = ''

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
