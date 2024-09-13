import os
import pinecone
import pandas as pd
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
import wandb

from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize HuggingFaceEmbeddings
#embed_ = HuggingFaceEmbeddings()


#embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embed_model = SentenceTransformer('Snowflake/snowflake-arctic-embed-l', trust_remote_code=True)


# def convert_question_to_vector(query):
#     try:
#         query_vector = embed_model.embed_query(query)
#         return query_vector
#     except Exception as e:
#         print(f"Error converting question to vector: {e}")
#         return None
    
def convert_question_to_vector(embed_model: Any, query: List[str]) -> List[List[float]]:
    """
    A wrapper function to get embeddings from different types of models.

    Args:
        embed_model (Any): The embedding model (either HuggingFaceEmbeddings or a model with encode_documents method).
        texts (List[str]): A list of texts to embed.

    Returns:
        List[List[float]]: A list of embeddings.
    """
    if hasattr(embed_model, 'embed_documents'):
        return embed_model.embed_documents(query)
    elif hasattr(embed_model, 'encode'):
        return embed_model.encode(query)
    else:
        raise AttributeError("The provided model doesn't have 'embed_documents' or 'encode' method.")


def calculate_mrr(question_id: str, general_ids: List[str]) -> Tuple[int, float]:
    if question_id in general_ids:
        rank = general_ids.index(question_id) + 1
        reciprocal_rank = 1 / rank
    else:
        rank = 0
        reciprocal_rank = 0
    return rank, reciprocal_rank

def calculate_hit_at_k(question_id: str, general_ids: List[str], k: int) -> int:
    return int(question_id in general_ids[:k])

def evaluate_retriever(qa_df, docsearch, convert_question_to_vector, k_values=[1, 3, 5]):
    total_mrr = 0
    total_retrieval_time = 0
    hit_at_k = {k: 0 for k in k_values}
    num_questions = len(qa_df)

    for _, row in qa_df.iterrows():
        question = row['question']
        question_id = row['id']

        # Measure retrieval time
        start_time = time.time()
        embed_question = convert_question_to_vector(embed_model, question)
        search_results = docsearch.similarity_search_by_vector_with_score(embed_question, k=max(k_values))
        end_time = time.time()
        retrieval_time = end_time - start_time
        total_retrieval_time += retrieval_time

        # Extract general_ids from search results
        general_ids = [item[0].metadata['general_id'] for item in search_results]

        # Calculate MRR
        _, reciprocal_rank = calculate_mrr(question_id, general_ids)
        total_mrr += reciprocal_rank

        # Calculate Hit@K
        for k in k_values:
            hit_at_k[k] += calculate_hit_at_k(question_id, general_ids, k)

    # Calculate averages
    avg_mrr = total_mrr / num_questions
    avg_retrieval_time = total_retrieval_time / num_questions
    avg_hit_at_k = {k: hits / num_questions for k, hits in hit_at_k.items()}

    return {
        "MRR": avg_mrr,
        "Avg Retrieval Time": avg_retrieval_time,
        "HitatK": avg_hit_at_k
    }

def print_comparison_and_best_sizes(all_results):
    # Comparing results across chunk sizes
    print("\nComparison across chunk sizes:")
    metrics = ["MRR", "Avg Retrieval Time"] + [f"Hit_{k}" for k in [1, 3, 5]]

    for metric in metrics:
        print(f"\n{metric}:")
        for chunk_size, results in all_results.items():
            if metric == "Avg Retrieval Time":
                value = results[metric]
            elif metric.startswith("Hit_"):
                k = int(metric.split("_")[1])
                value = results["HitatK"][k]
            else:
                value = results[metric]
            print(f"  Chunk size {chunk_size}: {value:.4f}")

def test_all_pinecone_indexes(qa_df: pd.DataFrame, convert_question_to_vector, k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
    # Get Pinecone credentials from environment variables
    #api_key = os.getenv('PINECONE_API_KEY')
    api_key = "84ec63c2-829b-4234-9a56-dfbaea240ffe"
    if not api_key:
        raise ValueError("Pinecone API key not found in .env file")

    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)

    # Get list of all indexes
    index_list = pc.list_indexes().names()
    
    all_results = {}

    for index_name in index_list:
        print(f"\nEvaluating index: {index_name}")

        # Connect to the index
        index = pc.Index(index_name)

        # Create a wrapper for the Pinecone index
        class PineconeWrapper:
            def __init__(self, index):
                self.index = index

            def similarity_search_by_vector_with_score(self, query_vector, k):
                results = self.index.query(vector=query_vector, top_k=k, include_metadata=True)
                return [(type('obj', (), {'metadata': item['metadata']})(), item['score']) for item in results['matches']]

        docsearch = PineconeWrapper(index)

        # Evaluate the index
        index_results = evaluate_retriever(qa_df, docsearch, convert_question_to_vector, k_values)
        all_results[index_name] = index_results

        # Log results to wandb
        wandb.log({
            f"{index_name}/MRR": index_results['MRR'],
            f"{index_name}/Avg_Retrieval_Time": index_results['Avg Retrieval Time'],
            **{f"{index_name}/Hit@{k}": hit_rate for k, hit_rate in index_results['HitatK'].items()}
        })

        print(f"Average MRR: {index_results['MRR']:.4f}")
        print(f"Average Retrieval Time: {index_results['Avg Retrieval Time']:.4f} seconds")
        for k, hit_rate in index_results['HitatK'].items():
            print(f"Hit@{k}: {hit_rate:.4f}")

    return all_results

# Main execution
if __name__ == "__main__":
    # Initialize wandb with more flexible options
    wandb.init(project="pinecone_index_evaluation", entity=None, config={
        "description": "Evaluation of Pinecone indexes for retrieval performance"
    })

    qa_df = pd.read_csv('../assets/csv/qa_df.csv', index_col=0)
    results = test_all_pinecone_indexes(qa_df, convert_question_to_vector)
    print_comparison_and_best_sizes(results)

    # Create a summary table in wandb
    table = wandb.Table(columns=["Index", "MRR", "Avg Retrieval Time"] + [f"Hit@{k}" for k in [1, 3, 5]])
    for index_name, result in results.items():
        table.add_data(index_name, result['MRR'], result['Avg Retrieval Time'], 
                       result['HitatK'][1], result['HitatK'][3], result['HitatK'][5])
    
    wandb.log({"results_summary": table})

    # Finish the wandb run
    wandb.finish()