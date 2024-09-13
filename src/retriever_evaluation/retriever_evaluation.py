import time
from typing import List, Dict, Any
import wandb
import logging
from config import load_config
from helpers import (
    convert_question_to_vector,
    calculate_mrr,
    calculate_hit_at_k,
    connect_pinecone
)
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_retriever(qa_df, docsearch, convert_question_to_vector, k_values: List[int]) -> Dict[str, Any]:
    """
    Evaluate the retriever's performance on a set of questions.
    
    Args:
    qa_df: DataFrame containing questions and their IDs
    docsearch: The document search object
    convert_question_to_vector: Function to convert questions to vectors
    k_values: List of k values for Hit@K calculation

    Returns:
    Dict containing MRR, average retrieval time, and Hit@K metrics
    """
    total_mrr = 0
    total_retrieval_time = 0
    hit_at_k = {k: 0 for k in k_values}
    num_questions = len(qa_df)

    for _, row in qa_df.iterrows():
        question = row['question']
        question_id = row['id']

        start_time = time.time()
        embed_question = convert_question_to_vector(question)
        search_results = docsearch.similarity_search_by_vector_with_score(embed_question, k=max(k_values))
        retrieval_time = time.time() - start_time
        total_retrieval_time += retrieval_time

        general_ids = [item[0].metadata['general_id'] for item in search_results]

        _, reciprocal_rank = calculate_mrr(question_id, general_ids)
        total_mrr += reciprocal_rank

        for k in k_values:
            hit_at_k[k] += calculate_hit_at_k(question_id, general_ids, k)

    avg_mrr = total_mrr / num_questions
    avg_retrieval_time = total_retrieval_time / num_questions
    avg_hit_at_k = {k: hits / num_questions for k, hits in hit_at_k.items()}

    return {
        "MRR": avg_mrr,
        "Avg Retrieval Time": avg_retrieval_time,
        "HitatK": avg_hit_at_k
    }

def evaluate_all_chunk_sizes(qa_df, chunk_sizes: List[int], convert_question_to_vector, k_values: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate the retriever for all specified chunk sizes and log results to wandb.
    
    Args:
    qa_df: DataFrame containing questions and their IDs
    chunk_sizes: List of chunk sizes to evaluate
    convert_question_to_vector: Function to convert questions to vectors
    k_values: List of k values for Hit@K calculation

    Returns:
    Dict containing results for each chunk size
    """
    results = {}
    
    # Initialize wandb run
    wandb.init(project="RAG_Chunk_Size_Evaluation", config={"k_values": k_values})

    for chunk_size in chunk_sizes:
        index_name = f"docs-chunk-{chunk_size}"

        # Create a new run for each chunk size
        with wandb.init(project="RAG_Chunk_Size_Evaluation", name=f"chunk_size_{chunk_size}", config={"chunk_size": chunk_size}):
            docsearch = connect_pinecone(index_name)

            logging.info(f"Evaluating chunk size: {chunk_size}")
            chunk_results = evaluate_retriever(qa_df, docsearch, convert_question_to_vector, k_values)
            results[chunk_size] = chunk_results

            # Log metrics to wandb
            wandb.log({
                "MRR": chunk_results['MRR'],
                "Avg_Retrieval_Time": chunk_results['Avg Retrieval Time'],
                **{f"Hit@{k}": hit_rate for k, hit_rate in chunk_results['HitatK'].items()}
            })

            logging.info(f"Average MRR: {chunk_results['MRR']:.4f}")
            logging.info(f"Average Retrieval Time: {chunk_results['Avg Retrieval Time']:.4f} seconds")
            for k, hit_rate in chunk_results['HitatK'].items():
                logging.info(f"Hit@{k}: {hit_rate:.4f}")

    return results

def print_comparison_and_best_sizes(all_results: Dict[int, Dict[str, Any]]):
    """
    Print comparison of results across chunk sizes and identify the best chunk size for each metric.
    
    Args:
    all_results: Dict containing results for each chunk size
    """
    logging.info("\nComparison across chunk sizes:")
    metrics = ["MRR", "Avg Retrieval Time"] + [f"Hit_{k}" for k in [1, 3, 5]]

    # Create a wandb Table
    table = wandb.Table(columns=["Chunk Size"] + metrics)

    for chunk_size, results in all_results.items():
        row = [chunk_size]
        for metric in metrics:
            if metric == "Avg Retrieval Time":
                value = results[metric]
            elif metric.startswith("Hit_"):
                k = int(metric.split("_")[1])
                value = results["HitatK"][k]
            else:
                value = results[metric]
            row.append(value)
            logging.info(f"  Chunk size {chunk_size}, {metric}: {value:.4f}")
        table.add_data(*row)

    # Log the table to wandb
    wandb.log({"Results Comparison": table})

    best_chunk_sizes = defaultdict(list)
    for metric in metrics:
        if metric == "Avg Retrieval Time":
            best_value = min(results[metric] for results in all_results.values())
        else:
            best_value = max(results[metric] if metric == "MRR" else results["HitatK"][int(metric.split("_")[1])] 
                             for results in all_results.values())

        for chunk_size, results in all_results.items():
            if metric == "Avg Retrieval Time":
                value = results[metric]
            elif metric.startswith("Hit_"):
                k = int(metric.split("_")[1])
                value = results["HitatK"][k]
            else:
                value = results[metric]

            if value == best_value:
                best_chunk_sizes[metric].append(chunk_size)

    logging.info("\nBest chunk size(s) for each metric:")
    for metric, sizes in best_chunk_sizes.items():
        logging.info(f"{metric}: {sizes}")

    # Log best chunk sizes to wandb
    wandb.log({"Best Chunk Sizes": best_chunk_sizes})

if __name__ == "__main__":
    config = load_config()
    
    wandb.login()  # Make sure to log in to wandb
    
    chunk_sizes = config['chunk_sizes']
    k_values = config['k_values']
    
    qa_df = load_qa_data(config['qa_data_path'])  # You need to implement this function
    
    all_results = evaluate_all_chunk_sizes(qa_df, chunk_sizes, convert_question_to_vector, k_values)
    print_comparison_and_best_sizes(all_results)

    wandb.finish()  # Close the wandb run