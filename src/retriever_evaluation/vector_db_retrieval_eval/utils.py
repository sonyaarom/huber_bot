import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Callable, Optional
from collections import defaultdict



def load_bm25_values(file_path: str) -> dict:
    """
    Load BM25 values from a JSON file.

    This function reads a JSON file containing pre-computed BM25 values
    and returns them as a dictionary. These values are typically used
    for generating sparse vectors in hybrid search scenarios.

    Args:
        file_path (str): The path to the JSON file containing BM25 values.

    Returns:
        dict: A dictionary containing the loaded BM25 values.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    Example:
        bm25_values = load_bm25_values('/path/to/bm25_values.json')
    """
    with open(file_path, 'r') as f:
        return json.load(f)
    
#TODO: update to trust remote 
def get_embedding_model(model_name: str, trust_remote_code: Optional[bool] = None) -> SentenceTransformer:
    """
    Get a SentenceTransformer embedding model.

    This function initializes and returns a SentenceTransformer model based on the provided model name.
    It allows for explicit control over whether to trust remote code.

    Args:
        model_name (str): The name or path of the model to load.
        trust_remote_code (Optional[bool]): Whether to trust remote code. If None, it will be set to True
                                            for specific models known to require it, and False otherwise.

    Returns:
        SentenceTransformer: The initialized SentenceTransformer model.

    Raises:
        ValueError: If the model_name is empty or None.
        Exception: Any exception raised during model initialization.

    Example:
        model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        model = get_embedding_model("Snowflake/snowflake-arctic-embed-l", trust_remote_code=True)
    """
    if not model_name:
        raise ValueError("Model name must be provided")

    # List of models known to require trust_remote_code=True
    models_requiring_trust = ["Snowflake/snowflake-arctic-embed-l", "sentence-transformers/all-MiniLM-L6-v2"]

    # Set trust_remote_code if not explicitly provided
    if trust_remote_code is None:
        trust_remote_code = model_name in models_requiring_trust

    try:
        return SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    except Exception as e:
        raise Exception(f"Failed to initialize SentenceTransformer model '{model_name}': {str(e)}")
    

def convert_question_to_vector(embed_model: Any, query: str) -> List[float]:
    """
    Convert a question (query) into a vector representation using the provided embedding model.

    This function takes an embedding model and a query string, and returns the vector
    representation of the query. It supports different types of embedding models by
    checking for either an 'encode' or 'embed_documents' method.

    Args:
        embed_model (Any): The embedding model to use for converting the query to a vector.
                           This model should have either an 'encode' or 'embed_documents' method.
        query (str): The question or query string to be converted into a vector.

    Returns:
        List[float]: A list of floats representing the vector encoding of the input query.

    Raises:
        AttributeError: If the provided embed_model doesn't have either an 'encode'
                        or 'embed_documents' method.

    Example:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query = "What is the capital of France?"
        vector = convert_question_to_vector(model, query)

    Note:
        The function first checks for an 'encode' method, which is common in many embedding
        models like SentenceTransformer. If that's not found, it looks for an 'embed_documents'
        method, which is used in some other embedding model implementations. The result is
        always converted to a Python list for consistency.
    """
    if hasattr(embed_model, 'encode'):
        return embed_model.encode(query).tolist()
    elif hasattr(embed_model, 'embed_documents'):
        return embed_model.embed_documents(query).tolist()
    else:
        raise AttributeError("The provided model doesn't have an 'encode' or 'embed_documents' method.")
    

def calculate_mrr(question_id: str, general_ids: List[str]) -> Tuple[int, float]:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a single question.

    This function calculates the reciprocal rank of a specific question ID within a list of
    retrieved document IDs. The reciprocal rank is the multiplicative inverse of the rank
    of the first correct answer. If the correct answer is not in the list, the reciprocal
    rank is 0.

    Args:
        question_id (str): The ID of the correct answer or relevant document.
        general_ids (List[str]): A list of retrieved document IDs, ordered by relevance
                                 (most relevant first).

    Returns:
        Tuple[int, float]: A tuple containing:
            - rank (int): The position of the correct answer (1-based index).
                          Returns 0 if the correct answer is not in the list.
            - reciprocal_rank (float): The reciprocal of the rank.
                                       Returns 0 if the correct answer is not in the list.

    Example:
        question_id = "Q1"
        general_ids = ["Q2", "Q1", "Q3", "Q4"]
        rank, rr = calculate_mrr(question_id, general_ids)
        # Returns: (2, 0.5)

    Note:
        - The MRR is typically used to evaluate the performance of question-answering
          or information retrieval systems.
        - This function calculates the reciprocal rank for a single question. To get
          the overall MRR, you would average the reciprocal ranks over multiple questions.
        - The rank is 1-based, meaning the first position has a rank of 1, not 0.
    """
    if question_id in general_ids:
        rank = general_ids.index(question_id) + 1
        reciprocal_rank = 1 / rank
    else:
        rank = 0
        reciprocal_rank = 0
    return rank, reciprocal_rank


def hybrid_scale(dense: List[float], sparse: Dict[str, List], alpha: float) -> Tuple[List[float], Dict[str, List]]:
    """
    Scale dense and sparse vectors for hybrid search.

    This function scales dense and sparse vectors based on an alpha value,
    allowing for a weighted combination of dense and sparse representations
    in hybrid search systems.

    Args:
        dense (List[float]): The dense vector representation.
        sparse (Dict[str, List]): The sparse vector representation,
                                  expected to have 'indices' and 'values' keys.
        alpha (float): The scaling factor, between 0 and 1. It determines the
                       weight of the dense vector in the hybrid representation.

    Returns:
        Tuple[List[float], Dict[str, List]]: A tuple containing:
            - hdense (List[float]): The scaled dense vector.
            - hsparse (Dict[str, List]): The scaled sparse vector.

    Raises:
        ValueError: If alpha is not between 0 and 1.

    Example:
        dense = [0.1, 0.2, 0.3]
        sparse = {'indices': [0, 2], 'values': [0.5, 0.7]}
        alpha = 0.6
        hdense, hsparse = hybrid_scale(dense, sparse, alpha)

    Note:
        - The dense vector is scaled by alpha.
        - The sparse vector values are scaled by (1 - alpha).
        - This scaling allows for a balanced combination of dense and sparse
          representations in hybrid search systems.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def calculate_hit_at_k(question_id: str, general_ids: List[str], k: int) -> int:
    """
    Calculate the Hit@k metric for a single question.

    Hit@k is a binary measure that indicates whether the correct answer
    is present in the top k retrieved results. It returns 1 if the correct
    answer is within the first k results, and 0 otherwise.

    Args:
        question_id (str): The ID of the correct answer or relevant document.
        general_ids (List[str]): A list of retrieved document IDs, ordered by
                                 relevance (most relevant first).
        k (int): The number of top results to consider.

    Returns:
        int: 1 if the correct answer is in the top k results, 0 otherwise.

    Example:
        question_id = "Q1"
        general_ids = ["Q2", "Q1", "Q3", "Q4"]
        hit_at_3 = calculate_hit_at_k(question_id, general_ids, 3)
        # Returns: 1 (because "Q1" is within the top 3)

    Note:
        - This metric is commonly used in information retrieval and
          recommender systems to evaluate the quality of top-k results.
        - It does not consider the specific position within the top k,
          only whether the correct answer is present or not.
    """
    return int(question_id in general_ids[:k])


def generate_sparse_vector(query: str, bm25_values: dict) -> Dict[str, Any]:
    """
    Generate a sparse vector representation of a query using BM25 scoring.

    This function creates a sparse vector for a given query based on pre-computed BM25 values.
    It uses the BM25 algorithm, which is a ranking function used by search engines to rank
    matching documents according to their relevance to a given search query.

    Args:
        query (str): The input query string.
        bm25_values (dict): A dictionary containing pre-computed BM25 values, including:
            - 'vocabulary': List of terms in the corpus vocabulary.
            - 'idf': Dict of inverse document frequency values for each term.
            - 'k1': BM25 parameter for term frequency scaling.
            - 'b': BM25 parameter for length normalization.
            - 'avgdl': Average document length in the corpus.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'indices': List of indices (positions) of non-zero elements in the sparse vector.
            - 'values': List of corresponding non-zero values.

    Example:
        query = "information retrieval"
        bm25_values = {
            'vocabulary': ['information', 'retrieval', 'system', ...],
            'idf': {'information': 1.5, 'retrieval': 2.0, ...},
            'k1': 1.2,
            'b': 0.75,
            'avgdl': 20
        }
        sparse_vector = generate_sparse_vector(query, bm25_values)

    Note:
        - The function uses the BM25 scoring formula to calculate the weight of each term.
        - Terms not present in the query will have a score of 0 and are omitted from the output.
        - This sparse representation is efficient for large vocabularies where most terms
          in a given query have zero weight.
        - The resulting sparse vector can be used in hybrid search systems or for
          efficient similarity computations.
    """
    query_terms = query.lower().split()
    vector = {}
    for i, term in enumerate(bm25_values['vocabulary']):
        if term in query_terms:
            tf = query_terms.count(term)
            idf = bm25_values['idf'].get(term, 0)
            k1, b = bm25_values['k1'], bm25_values['b']
            avgdl = bm25_values['avgdl']
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * len(query_terms) / avgdl)
            vector[i] = idf * (numerator / denominator)
    
    return {"indices": list(vector.keys()), "values": list(vector.values())}


def convert_entities_to_label_name_dict(entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Convert a list of entity dictionaries into a dictionary organized by entity labels.

    This function takes the output of a Named Entity Recognition (NER) system and
    reorganizes it into a format where each entity label (category) is associated
    with a list of unique entity texts found for that label.

    Args:
        entities (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
            represents an entity and contains at least two keys:
            - 'text': The text of the entity.
            - 'label': The label (category) of the entity.

    Returns:
        Dict[str, List[str]]: A dictionary where:
            - Keys are entity labels (categories).
            - Values are lists of unique entity texts associated with each label.

    Example:
        input_entities = [
            {'text': 'John Doe', 'label': 'PERSON'},
            {'text': 'New York', 'label': 'LOCATION'},
            {'text': 'Jane Doe', 'label': 'PERSON'},
            {'text': 'New York', 'label': 'LOCATION'}
        ]
        result = convert_entities_to_label_name_dict(input_entities)
        # Result:
        # {
        #     'PERSON': ['john doe', 'jane doe'],
        #     'LOCATION': ['new york']
        # }

    Note:
        - Entity texts are converted to lowercase and stripped of leading/trailing whitespace.
        - Duplicate entity texts within the same label are removed (using a set internally).
        - This function is useful for preparing NER results for further processing,
          such as entity-based filtering or query expansion in information retrieval systems.
    """
    label_name_dict = defaultdict(set)
    for entity in entities:
        text = entity['text'].strip().lower()
        label_name_dict[entity['label']].add(text)
    return {k: list(v) for k, v in label_name_dict.items()}


def create_complex_filter(entities: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Create a complex filter dictionary for Pinecone vector database queries based on entity types and values.

    This function constructs a filter that can be used in Pinecone queries to filter results
    based on multiple entity types and their corresponding values. It creates an OR condition
    across different entity types, allowing for flexible and powerful filtering capabilities.

    Args:
        entities (Dict[str, List[str]]): A dictionary where:
            - Keys are entity types (e.g., 'PERSON', 'LOCATION', 'ORGANIZATION')
            - Values are lists of entity values for each type

    Returns:
        Dict[str, Any]: A complex filter dictionary compatible with Pinecone queries. 
        The structure depends on the input:
            - If multiple entity types are present, it returns an OR condition.
            - If only one entity type is present, it returns a simple condition.
            - If no valid entity types are present, it returns an empty dictionary.

    Examples:
        1. Multiple entity types:
           Input: {
               'PERSON': ['John Doe', 'Jane Smith'],
               'LOCATION': ['New York', 'London']
           }
           Output: {
               '$or': [
                   {'PERSON': {'$in': ['John Doe', 'Jane Smith']}},
                   {'LOCATION': {'$in': ['New York', 'London']}}
               ]
           }

        2. Single entity type:
           Input: {'ORGANIZATION': ['Google', 'Microsoft']}
           Output: {'ORGANIZATION': {'$in': ['Google', 'Microsoft']}}

        3. No valid entities:
           Input: {} or {'PERSON': []}
           Output: {}

    Note:
        - The function uses Pinecone's query syntax, where '$in' represents an inclusion filter
          and '$or' represents a logical OR operation.
        - Empty lists for any entity type are ignored in the filter creation.
        - This function is particularly useful for creating metadata filters in Pinecone queries,
          allowing for entity-based filtering of vector search results.
    """
    filter_conditions = []
    for entity_type, entity_values in entities.items():
        if entity_values:  # Only add non-empty entity lists to the filter
            filter_conditions.append({f"{entity_type}": {"$in": entity_values}})
    
    # Combine all conditions with $or
    if len(filter_conditions) > 1:
        return {"$or": filter_conditions}
    elif len(filter_conditions) == 1:
        return filter_conditions[0]
    else:
        return {}