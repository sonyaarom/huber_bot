from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Union
from sentence_transformers import CrossEncoder
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BM25Vectorizer:
    """
    A class for creating sparse vector representations of queries using the BM25 algorithm.

    BM25 (Best Matching 25) is a ranking function used by search engines to rank matching
    documents according to their relevance to a given search query. This class uses the
    BM25 algorithm to convert text queries into sparse vector representations.

    Attributes:
        bm25 (BM25Okapi): An instance of the BM25Okapi model from the rank_bm25 library.
        vocabulary (List[str]): A list of unique words from the corpus.

    Example:
        corpus = [
            ["hello", "world"],
            ["goodbye", "world"],
            ["hello", "there"]
        ]
        vectorizer = BM25Vectorizer(corpus)
        query = "hello world"
        sparse_vector = vectorizer.get_sparse_vector(query)
    """

    def __init__(self, corpus: List[List[str]]):
        """
        Initialize the BM25Vectorizer with a corpus of documents.

        Args:
            corpus (List[List[str]]): A list of documents, where each document
                                      is represented as a list of words.

        Note:
            The corpus is used to train the BM25 model and build the vocabulary.
            Each document in the corpus should be pre-processed (tokenized, lowercased, etc.)
            before being passed to this constructor.
        """
        self.bm25 = BM25Okapi(corpus)
        self.vocabulary = list(set(word for doc in corpus for word in doc))

    def get_sparse_vector(self, query: str) -> Dict[str, List[Any]]:
        """
        Convert a query string into a sparse vector representation using BM25 scores.

        This method calculates BM25 scores for each term in the query against the corpus,
        and returns a sparse vector representation of these scores.

        Args:
            query (str): The input query string.

        Returns:
            Dict[str, List[Any]]: A dictionary containing:
                - 'indices': List of indices (positions) of non-zero elements in the sparse vector.
                - 'values': List of corresponding non-zero BM25 scores.

        Note:
            - If no terms in the query match the corpus vocabulary, a default vector
              is returned with the first index set to 1.0.
            - The returned sparse vector can be used in hybrid search systems or for
              efficient similarity computations.

        Example:
            query = "hello world"
            sparse_vector = vectorizer.get_sparse_vector(query)
            # Might return something like:
            # {'indices': [0, 1], 'values': [0.5, 0.3]}
            # where 0 and 1 are the indices of "hello" and "world" in the vocabulary,
            # and 0.5 and 0.3 are their respective BM25 scores.
        """
        query_terms = query.split()
        scores = self.bm25.get_scores(query_terms)
        indices = [i for i, score in enumerate(scores) if score > 0]
        values = [scores[i] for i in indices]
        
        # If no terms matched, use the first term as a fallback
        if not indices:
            indices = [0]
            values = [1.0]
        
        return {"indices": indices, "values": values}
    

logger = logging.getLogger(__name__)
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class PineconeWrapper:
    def __init__(self, index):
        self.index = index

    def flexible_hybrid_search(self, dense_vec: List[float], sparse_vec: Dict[str, List[Any]],
                               k: int, filter_dict: Dict[str, Any] = None,
                               use_parent_chunk_retriever: bool = False,
                               max_chunks_per_id: int = 10) -> List[Dict[str, Any]]:
        try:
            if not sparse_vec or not sparse_vec['indices']:
                logger.info("Performing dense search due to empty sparse vector")
                initial_results = self.dense_search(dense_vec, k, filter_dict)
            else:
                logger.info("Performing hybrid search")
                initial_results = self.hybrid_search(dense_vec, sparse_vec, k, filter_dict)

            if not initial_results:
                logger.warning("No results found in initial search")
                return []

            if not use_parent_chunk_retriever:
                logger.info("Parent chunk retriever not used, returning initial results")
                return [{'id': item[0].metadata.get('id', ''), 'score': item[1], 'metadata': item[0].metadata} for item in initial_results]

            all_chunks = []
            for item in initial_results:
                general_id = item[0].metadata.get('general_id')
                if general_id:
                    chunk_results = self.dense_search(dense_vec, max_chunks_per_id, {"general_id": {"$eq": general_id}})
                    all_chunks.extend(chunk_results)

            if not all_chunks:
                logger.warning("No chunks found for parent chunk retrieval")
                return [{'id': item[0].metadata.get('id', ''), 'score': item[1], 'metadata': item[0].metadata} for item in initial_results]

            # Group chunks by general_id
            grouped_results = {}
            for chunk in all_chunks:
                general_id = chunk[0].metadata.get('general_id')
                if general_id not in grouped_results:
                    grouped_results[general_id] = {
                        'id': general_id,
                        'score': chunk[1],
                        'metadata': chunk[0].metadata,
                        'chunks': []
                    }
                grouped_results[general_id]['chunks'].append({
                    'id': chunk[0].metadata.get('id', ''),
                    'score': chunk[1],
                    'metadata': chunk[0].metadata
                })

            # Sort grouped results by the highest chunk score and return top k
            sorted_results = sorted(grouped_results.values(), key=lambda x: x['score'], reverse=True)[:k]
            return sorted_results

        except Exception as e:
            logger.error(f"Error in flexible_hybrid_search: {str(e)}")
            return []  # Return an empty list instead of None

    def hybrid_search(self, dense_vec: List[float], sparse_vec: Dict[str, List[Any]], 
                      k: int, filter_dict: Dict[str, Any] = None) -> List[Tuple[Any, float]]:
        try:
            results = self.index.query(
                vector=dense_vec,
                sparse_vector=sparse_vec,
                filter=filter_dict,
                top_k=k,
                include_metadata=True
            )
            return [(type('obj', (), {'metadata': item.metadata})(), item.score) for item in results.matches]
        except Exception as e:
            logger.error(f"Error in hybrid_search: {str(e)}")
            return []

    def dense_search(self, dense_vec: List[float], k: int, 
                     filter_dict: Dict[str, Any] = None) -> List[Tuple[Any, float]]:
        try:
            results = self.index.query(
                vector=dense_vec,
                filter=filter_dict,
                top_k=k,
                include_metadata=True
            )
            return [(type('obj', (), {'metadata': item.metadata})(), item.score) for item in results.matches]
        except Exception as e:
            logger.error(f"Error in dense_search: {str(e)}")
            return []



class CrossEncoderWrapper:
    """
    A wrapper class for using CrossEncoder models to rerank passages.

    This class provides an interface to initialize a CrossEncoder model and use it
    for reranking a list of passages based on their relevance to a given query.
    CrossEncoder models are particularly effective for reranking tasks as they
    can directly estimate the relevance between a query and a passage.

    Attributes:
        model (CrossEncoder): The underlying CrossEncoder model used for reranking.

    Example:
        reranker = CrossEncoderWrapper('cross-encoder/ms-marco-MiniLM-L-6-v2')
        query = "What is machine learning?"
        passages = ["Machine learning is a subfield of AI.", "Machine learning uses data to improve."]
        scores = reranker.rerank(query, passages)
    """

    def __init__(self, model_name: str):
        """
        Initialize the CrossEncoderWrapper with a specific CrossEncoder model.

        Args:
            model_name (str): The name or path of the CrossEncoder model to use.
                              This should be a model compatible with the sentence-transformers library.

        Note:
            Common CrossEncoder models for reranking include:
            - 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            - 'cross-encoder/ms-marco-electra-base'
            The choice of model can affect both performance and accuracy.
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """
        Rerank a list of passages based on their relevance to the given query.

        This method uses the CrossEncoder model to compute relevance scores for
        each query-passage pair and returns these scores as a list.

        Args:
            query (str): The query string to rank passages against.
            passages (List[str]): A list of passage strings to be ranked.

        Returns:
            List[float]: A list of relevance scores corresponding to each passage.
                         Higher scores indicate higher relevance to the query.

        Note:
            - The scores are returned in the same order as the input passages.
            - These scores can be used to sort the passages by relevance.
            - The exact range and interpretation of scores may vary depending on the specific CrossEncoder model used.

        Example:
            query = "Impact of climate change"
            passages = [
                "Climate change affects global temperatures.",
                "Economic policies in developing nations.",
                "Rising sea levels due to global warming."
            ]
            scores = reranker.rerank(query, passages)
            # scores might look like [0.95, 0.2, 0.85]
        """
        pairs = [[query, passage] for passage in passages]
        scores = self.model.predict(pairs)
        return scores.tolist()