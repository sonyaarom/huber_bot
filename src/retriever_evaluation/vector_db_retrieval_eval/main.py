import pandas as pd
import wandb
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from gliner import GLiNER

from ignore import (
    WANDB_PROJECT, WANDB_ENTITY, QA_DF_PATH, BM25_VALUES_PATH, 
    DEFAULT_ALPHA_VALUES, DEFAULT_RERANKER_MODEL, NER_MODEL, NER_LABELS,
    API_CONFIGS, DEFAULT_K_VALUES
)
from utils import (
    load_bm25_values, get_embedding_model, convert_question_to_vector,
    convert_entities_to_label_name_dict
)
from models import PineconeWrapper, CrossEncoderWrapper
from evaluation import test_all_pinecone_indexes, print_comparison_and_best_sizes

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={
        "description": "Flexible Evaluation of Pinecone indexes for retrieval performance"
    })

    qa_df = pd.read_csv(QA_DF_PATH, index_col=0)
    
    # User inputs
    use_reranker = input("Do you want to use a reranker? (yes/no): ").lower() == 'yes'
    use_ner = input("Do you want to use NER for filtering? (yes/no): ").lower() == 'yes'
    search_type = input("Enter search type (dense/hybrid): ").lower()
    trust_remote = input("Do you want to trust remote code? (yes/no): ").lower() == 'yes'
    use_parent_chunk_retriever = input("Do you want to use the parent chunk retriever? (yes/no): ").lower() == 'yes'

    # Load BM25 values only if hybrid search is selected
    bm25_values = load_bm25_values(BM25_VALUES_PATH) if search_type == 'hybrid' else None

    # Set alpha values based on search type
    alpha_values = DEFAULT_ALPHA_VALUES if search_type == 'hybrid' else [1]

    # Initialize reranker model if selected
    reranker_model = CrossEncoderWrapper(DEFAULT_RERANKER_MODEL) if use_reranker else None

    # Check if 'entities' column exists, if not, create it using GLiNER (only if NER is used)
    if use_ner and 'entities' not in qa_df.columns:
        logger.info("'entities' column not found. Creating it using GLiNER...")
        ner_model = GLiNER.from_pretrained(NER_MODEL)
        qa_df['entities'] = qa_df['question'].apply(lambda x: convert_entities_to_label_name_dict(ner_model.predict_entities(x, NER_LABELS)))
        logger.info("'entities' column created successfully.")
    elif use_ner:
        # Ensure 'entities' column contains dictionaries
        qa_df['entities'] = qa_df['entities'].apply(lambda x: {} if pd.isna(x) else eval(x) if isinstance(x, str) else x)

    initial_k = int(input("Enter the number of initial results to retrieve: "))
    final_k = int(input("Enter the number of final results to keep: "))

    if use_parent_chunk_retriever:
        max_chunks_per_id = int(input("Enter the maximum number of chunks to retrieve per general ID: "))
    else:
        max_chunks_per_id = None

    logger.info(f"Starting evaluation with NER: {use_ner}, Reranker: {use_reranker}, Search Type: {search_type}, "
                f"Initial K: {initial_k}, Final K: {final_k}, Parent Chunk Retriever: {use_parent_chunk_retriever}")
    
    results = test_all_pinecone_indexes(
        qa_df=qa_df,
        bm25_values=bm25_values, 
        alpha_values=alpha_values, 
        use_ner=use_ner, 
        reranker_model=reranker_model, 
        initial_k=initial_k, 
        final_k=final_k,
        trust_remote=[trust_remote],
        api_configs=API_CONFIGS,
        get_embedding_model=get_embedding_model,
        convert_question_to_vector=convert_question_to_vector,
        pinecone_wrapper_class=PineconeWrapper,
        use_parent_chunk_retriever=use_parent_chunk_retriever,
        max_chunks_per_id=max_chunks_per_id
    )
    
    if results:
        print_comparison_and_best_sizes(results)

        table = wandb.Table(columns=["API_Key_Index_Alpha_NER_Reranker_PCR", "MRR", "Avg Retrieval Time"] + [f"Hit@{k}" for k in DEFAULT_K_VALUES])
        for index_name, result in results.items():
            table.add_data(index_name, result['MRR'], result['Avg Retrieval Time'], 
                           result['HitatK'][1], result['HitatK'][3], result['HitatK'][5])
        
        wandb.log({"results_summary": table})
    else:
        logger.warning("No results to display. All indexes were skipped or encountered errors.")

    wandb.finish()