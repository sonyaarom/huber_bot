import traceback
from typing import Dict, List, Any, Tuple
import pandas as pd
from pinecone import Pinecone
import wandb
from config import API_CONFIGS
from utils import get_embedding_model, convert_question_to_vector, generate_sparse_vector, hybrid_scale, create_complex_filter, calculate_mrr, calculate_hit_at_k
from models import PineconeWrapper


def rerank_results(question: str, search_results: List[Tuple[Any, float]], reranker_model: Any, final_k: int) -> List[Tuple[Any, float]]:
    """
    Rerank the search results using the provided reranker model.
    
    Args:
    question (str): The original question.
    search_results (List[Tuple[Any, float]]): The initial search results.
    reranker_model (Any): The reranker model to use.
    final_k (int): The number of top results to return after reranking.
    
    Returns:
    List[Tuple[Any, float]]: The reranked results.
    """
    logger.info(f"Reranking results for question: '{question}'")
    logger.info(f"Number of results to rerank: {len(search_results)}")
    
    passages = [result[0].metadata['text'] for result in search_results]
    
    logger.info("Sample of passages to rerank:")
    for i, passage in enumerate(passages[:3]):  # Log first 3 passages
        logger.info(f"Passage {i + 1}: {passage[:100]}...")  # Log first 100 chars of each passage
    
    reranked_scores = reranker_model.rerank(query=question, passages=passages)
    
    logger.info("Sample of reranked scores:")
    for i, score in enumerate(reranked_scores[:5]):  # Log first 5 scores
        logger.info(f"Score {i + 1}: {score}")
    
    reranked_results = [(search_results[i][0], score) for i, score in enumerate(reranked_scores)]
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Number of results after reranking: {len(reranked_results[:final_k])}")
    
    return reranked_results[:final_k]


import time
import logging
from typing import Any, Callable, Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

def evaluate_retriever(qa_df: pd.DataFrame, 
                       docsearch: Any, 
                       convert_question_to_vector: Callable[[str], List[float]], 
                       bm25_values: dict, 
                       k_values: List[int] = [1, 3, 5], 
                       alpha: float = 0.5, 
                       use_ner: bool = False,
                       reranker_model: Any = None, 
                       initial_k: int = 10, 
                       final_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate the performance of a retriever system on a question-answering dataset.

    This function processes a dataset of questions, retrieves relevant documents using
    a specified search method (dense or hybrid), optionally applies reranking, and
    calculates various performance metrics.

    Args:
        qa_df (pd.DataFrame): DataFrame containing questions, their IDs, and optionally, entities.
                              Expected columns: ['question', 'id', 'entities' (if use_ner is True)]
        docsearch (Any): An object (like PineconeWrapper) that provides search functionality.
        convert_question_to_vector (Callable[[str], List[float]]): Function to convert questions to dense vectors.
        bm25_values (dict): BM25 values for sparse vector generation. Used only if alpha < 1.
        k_values (List[int], optional): Values of k for calculating Hit@k. Defaults to [1, 3, 5].
        alpha (float, optional): Interpolation factor for hybrid search. Defaults to 0.5.
        use_ner (bool, optional): Whether to use named entity recognition for filtering. Defaults to False.
        reranker_model (Any, optional): Model for reranking results. If None, no reranking is applied.
        initial_k (int, optional): Number of initial results to retrieve. Defaults to 10.
        final_k (int, optional): Number of results to keep after reranking. Defaults to 5.

    Returns:
        Dict[str, Any]: A dictionary containing the following metrics:
            - "MRR": Mean Reciprocal Rank
            - "Avg Retrieval Time": Average time taken for retrieval per question
            - "HitatK": Dictionary of Hit@k values for each k in k_values

    Raises:
        Exception: Any exception during processing of individual questions is caught and logged.

    Notes:
        - The function logs detailed information about each question's processing and results.
        - It supports both dense and hybrid (dense + sparse) search methods.
        - NER can be used for filtering if 'entities' column is present in qa_df and use_ner is True.
        - Reranking is applied if a reranker_model is provided.

    Example:
        results = evaluate_retriever(qa_df, pinecone_wrapper, embed_question, bm25_values,
                                     k_values=[1, 5, 10], alpha=0.7, use_ner=True,
                                     reranker_model=cross_encoder, initial_k=20, final_k=10)
        print(f"MRR: {results['MRR']}")
        print(f"Average Retrieval Time: {results['Avg Retrieval Time']} seconds")
        print(f"Hit@k: {results['HitatK']}")
    """
    total_mrr = 0
    total_retrieval_time = 0
    hit_at_k = {k: 0 for k in k_values}
    num_questions = len(qa_df)

    for _, row in qa_df.iterrows():
        question = row['question']
        question_id = row['id']
        entities = row['entities'] if use_ner else None

        try:
            start_time = time.time()
            dense_vec = convert_question_to_vector(question)
            
            if alpha < 1 and bm25_values:
                sparse_vec = generate_sparse_vector(question, bm25_values)
                dense_vec, sparse_vec = hybrid_scale(dense_vec, sparse_vec, alpha)
            else:
                sparse_vec = None

            filter_dict = create_complex_filter(entities) if use_ner else None
            logger.info(f"Processing question: '{question}'")
            logger.info(f"Entities: {entities}")
            logger.info(f"Filter: {filter_dict}")
            
            if sparse_vec:
                search_results = docsearch.hybrid_search(dense_vec, sparse_vec, k=initial_k, filter_dict=filter_dict)
            else:
                search_results = docsearch.dense_search(dense_vec, k=initial_k, filter_dict=filter_dict)
            
            logger.info(f"Number of results before reranking: {len(search_results)}")
            logger.info("Top 3 results before reranking:")
            for i, result in enumerate(search_results[:3]):
                logger.info(f"Result {i + 1}: ID = {result[0].metadata['general_id']}, Score = {result[1]}")
            
            if reranker_model:
                logger.info("Applying reranking...")
                search_results = rerank_results(question, search_results, reranker_model, final_k)
                logger.info("Top 3 results after reranking:")
                for i, result in enumerate(search_results[:3]):
                    logger.info(f"Result {i + 1}: ID = {result[0].metadata['general_id']}, Score = {result[1]}")
            else:
                logger.info("Reranker not applied. Truncating to final_k results.")
                search_results = search_results[:final_k]
            
            end_time = time.time()
            retrieval_time = end_time - start_time
            total_retrieval_time += retrieval_time

            general_ids = [item[0].metadata['general_id'] for item in search_results]
            _, reciprocal_rank = calculate_mrr(question_id, general_ids)
            total_mrr += reciprocal_rank

            for k in k_values:
                hit_at_k[k] += calculate_hit_at_k(question_id, general_ids, k)

            logger.info(f"Retrieved IDs: {general_ids[:5]}...")
            logger.info(f"Retrieval time: {retrieval_time:.4f} seconds")
            logger.info(f"MRR for this question: {reciprocal_rank:.4f}")
            logger.info(f"Hit@k for this question: {[calculate_hit_at_k(question_id, general_ids, k) for k in k_values]}")

        except Exception as e:
            logger.error(f"Error processing question '{question}': {str(e)}")
            continue

    avg_mrr = total_mrr / num_questions
    avg_retrieval_time = total_retrieval_time / num_questions
    avg_hit_at_k = {k: hits / num_questions for k, hits in hit_at_k.items()}

    logger.info(f"Overall results:")
    logger.info(f"Average MRR: {avg_mrr:.4f}")
    logger.info(f"Average Retrieval Time: {avg_retrieval_time:.4f} seconds")
    logger.info(f"Average Hit@k: {avg_hit_at_k}")

    return {
        "MRR": avg_mrr,
        "Avg Retrieval Time": avg_retrieval_time,
        "HitatK": avg_hit_at_k
    }



def test_all_pinecone_indexes(qa_df: pd.DataFrame, 
                              bm25_values: dict, 
                              k_values: List[int] = [1, 3, 5], 
                              alpha_values: List[float] = [0, 0.2, 0.5, 0.8, 1], 
                              use_ner: bool = False,
                              reranker_model: Any = None, 
                              initial_k: int = 10, 
                              final_k: int = 5,
                              trust_remote: List[bool] = [True, False],
                              api_configs: Dict[str, Dict[str, Any]] = None,
                              get_embedding_model: Callable = None,
                              convert_question_to_vector: Callable = None,
                              pinecone_wrapper_class: Any = None) -> Dict[str, Any]:
    """
    Evaluate retrieval performance across multiple Pinecone indexes and configurations.

    This function iterates through different Pinecone API keys and their associated indexes,
    evaluating retrieval performance for each index across various alpha values (for hybrid search).
    It uses the evaluate_retriever function to perform detailed evaluation for each configuration.

    Args:
        qa_df (pd.DataFrame): DataFrame containing questions, their IDs, and optionally, entities.
        bm25_values (dict): BM25 values for sparse vector generation.
        k_values (List[int], optional): Values of k for calculating Hit@k. Defaults to [1, 3, 5].
        alpha_values (List[float], optional): Alpha values for hybrid search interpolation. 
                                              Defaults to [0, 0.2, 0.5, 0.8, 1].
        use_ner (bool, optional): Whether to use named entity recognition for filtering. Defaults to False.
        reranker_model (Any, optional): Model for reranking results. If None, no reranking is applied.
        initial_k (int, optional): Number of initial results to retrieve. Defaults to 10.
        final_k (int, optional): Number of results to keep after reranking. Defaults to 5.
        api_configs (Dict[str, Dict[str, Any]], optional): Configuration for different API keys.

    Returns:
        Dict[str, Any]: A nested dictionary containing results for each tested configuration.
        The keys are formatted as "{api_key_name}_{index_name}_alpha{alpha}_ner{use_ner}_reranker{reranker_used}",
        and the values are dictionaries containing evaluation metrics (MRR, Avg Retrieval Time, HitatK).

    Raises:
        Exception: Catches and logs any exceptions that occur during the evaluation process.

    Notes:
        - The function uses the API_CONFIGS from the config module to iterate through different API keys.
        - For each API key and index, it evaluates performance across all specified alpha values.
        - Results are logged to wandb (Weights & Biases) for each configuration.
        - Detailed error logging is implemented for debugging purposes.

    Example:
        results = test_all_pinecone_indexes(qa_df, bm25_values, 
                                            k_values=[1, 5, 10], 
                                            alpha_values=[0, 0.5, 1], 
                                            use_ner=True,
                                            reranker_model=cross_encoder_model)
        for config, metrics in results.items():
            print(f"Configuration: {config}")
            print(f"MRR: {metrics['MRR']}")
            print(f"Avg Retrieval Time: {metrics['Avg Retrieval Time']}")
            print(f"Hit@k: {metrics['HitatK']}")
    """
    all_results = {}

    for api_key_name, config in api_configs.items():
        api_key = config['api_key']
        default_embedding_model = config['default_embedding_model']

        if not api_key:
            print(f"Warning: API key for {api_key_name} not found. Skipping...")
            continue

        print(f"\nEvaluating indexes for API key: {api_key_name}")

        try:
            pc = Pinecone(api_key=api_key)
            index_list = pc.list_indexes()

            if not index_list:
                print(f"No indexes found for API key {api_key_name}. Skipping...")
                continue

            for index_info in index_list:
                index_name = index_info.name
                print(f"\nEvaluating index: {index_name}")

                try:
                    index = pc.Index(index_name)
                    embed_model = get_embedding_model(default_embedding_model, trust_remote_code=trust_remote)
                    docsearch = pinecone_wrapper_class(index)

                    def local_convert_question_to_vector(query: str) -> List[float]:
                        return convert_question_to_vector(embed_model, query)

                    for alpha in alpha_values:
                        print(f"Evaluating with alpha = {alpha}")
                        index_results = evaluate_retriever(qa_df, docsearch, local_convert_question_to_vector, 
                                                           bm25_values, k_values, alpha, use_ner, reranker_model,
                                                           initial_k=initial_k, final_k=final_k)
                        result_key = f"{api_key_name}_{index_name}_alpha{alpha}_ner{use_ner}_reranker{reranker_model is not None}"
                        all_results[result_key] = index_results

                        wandb.log({
                            f"{result_key}/MRR": index_results['MRR'],
                            f"{result_key}/Avg_Retrieval_Time": index_results['Avg Retrieval Time'],
                            **{f"{result_key}/Hit@{k}": hit_rate for k, hit_rate in index_results['HitatK'].items()}
                        })

                        print(f"Average MRR: {index_results['MRR']:.4f}")
                        print(f"Average Retrieval Time: {index_results['Avg Retrieval Time']:.4f} seconds")
                        for k, hit_rate in index_results['HitatK'].items():
                            print(f"Hit@{k}: {hit_rate:.4f}")

                except Exception as e:
                    print(f"Error occurred while evaluating index '{index_name}': {str(e)}")
                    traceback.print_exc()

        except Exception as e:
            print(f"Error occurred while processing API key {api_key_name}: {str(e)}")
            traceback.print_exc()

    return all_results

def print_comparison_and_best_sizes(all_results: Dict[str, Dict[str, Any]]):
    """
    Print a comparison of retrieval performance metrics across different indexes and configurations.

    This function takes the results from the test_all_pinecone_indexes function and prints
    a formatted comparison of key metrics (MRR, Average Retrieval Time, and Hit@k) for each
    tested configuration. It provides a clear overview of performance across different
    indexes, alpha values, and other configuration parameters.

    Args:
        all_results (Dict[str, Dict[str, Any]]): A nested dictionary containing evaluation results.
            The outer dictionary keys are configuration identifiers (e.g., "api_key_index_name_alpha0.5_nerFalse_rerankerTrue"),
            and the inner dictionaries contain the evaluation metrics for each configuration.

    Prints:
        A formatted comparison of metrics across all configurations, organized by metric type.
        For each metric, it prints the value for each configuration, allowing easy comparison.

    Metrics displayed:
        - MRR (Mean Reciprocal Rank)
        - Avg Retrieval Time
        - Hit@1, Hit@3, Hit@5 (or other k values as specified in the results)

    Note:
        - The function assumes that all configurations in all_results have the same set of metrics.
        - It automatically detects the Hit@k values present in the results.
        - The output is formatted for easy reading and comparison, with metrics rounded to 4 decimal places.

    Example output:
        Comparison across indexes and alpha values:

        MRR:
          api1_index1_alpha0.5_nerFalse_rerankerFalse: 0.7500
          api1_index1_alpha1.0_nerFalse_rerankerFalse: 0.8000
          api2_index1_alpha0.5_nerTrue_rerankerTrue: 0.8500

        Avg Retrieval Time:
          api1_index1_alpha0.5_nerFalse_rerankerFalse: 0.1200
          api1_index1_alpha1.0_nerFalse_rerankerFalse: 0.1000
          api2_index1_alpha0.5_nerTrue_rerankerTrue: 0.1500

        Hit_1:
          api1_index1_alpha0.5_nerFalse_rerankerFalse: 0.7000
          api1_index1_alpha1.0_nerFalse_rerankerFalse: 0.7500
          api2_index1_alpha0.5_nerTrue_rerankerTrue: 0.8000

        ... (continues for other Hit@k values)
    """
    print("\nComparison across indexes and alpha values:")
    metrics = ["MRR", "Avg Retrieval Time"] + [f"Hit_{k}" for k in [1, 3, 5]]

    for metric in metrics:
        print(f"\n{metric}:")
        for index_name, results in all_results.items():
            if metric == "Avg Retrieval Time":
                value = results[metric]
            elif metric.startswith("Hit_"):
                k = int(metric.split("_")[1])
                value = results["HitatK"][k]
            else:
                value = results[metric]
            print(f"  {index_name}: {value:.4f}")