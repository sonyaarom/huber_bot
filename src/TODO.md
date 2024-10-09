1. Build simple vanilla RAG where the whole code is being chunked and stored simply in the vector database
2. What should be retriever metric?  -**INCLUDE MLFLOW!**
3. Build and deploy final vanilla RAG


-------
4. Create CRON or Airflow Job that would download new schema 
5. Compare versions
6. How to update Pinecone?
--------

--------
Literature:
1. Jiang et al., (2023) [5] introduce FLARE (Forward-Looking Active Retrieval Augmented Generation), a novel method designed to enhance RAG systems. :
    * Evaluation metrics include exact match (EM), token-level F1, precision, recall, RoBERTa-based QA score (DisambigF1), ROUGE, and an overall DR score combining DisambigF1 and ROUGE.
    * Pierre et al., (2024) - Query Optimisation
    * Jin et al., (2024) - Dynamic Retriever
    * Johnson, Douze and J´egou (2017)  - FAISS algorithm for indexing vector embeddings in vector databases



TO DO CURRENT:
1. Basic retriever and LLM: experiments with different chunk sizes. What should be the metrics? (128, 256, 512, or 1024.)+
2. Metrics:
 * Recall@K:
 * Precision@K
 * Mean Average Precision (MAP)
 * Normalized Discounted Cumulative Gain (NDCG)
 * F1 Score 
 * Answer Accuracy
 * RAGAS?

Retrievers:
1. Parent Document Retriever Chain [3]
2. Ensemble Retriever Chain [3]

Feedback Loops:
1. Develop mechanisms to collect and incorporate user feedback for continuous improvement. [2]


Model Management:
1. Develop techniques for the model to acknowledge uncertainty or lack of information.???


1.Gao et al.’s [12] 
2. https://medium.com/@bijit211987/strategies-for-optimal-performance-of-rag-6faa1b79cd45
3. https://towardsai.net/p/machine-learning/evaluating-rag-metrics-across-different-retrieval-methods
4. https://arxiv.org/pdf/2309.03409




TODO: 18th September
- Prompt: what are the approaches?
- Where to place the chunk?
- Prompt versioning, how to store?

