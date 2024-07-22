import os
from typing import List, Optional
from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

class Model:
    def __init__(self, model_path: str, index_name: str, pinecone_api_key: str, max_tokens: int = 512):
        self.set_pinecone_api_key(pinecone_api_key)
        self.llm = self.load_model(model_path)
        self.docsearch = self.connect_pinecone(index_name)
        self.max_tokens = max_tokens

    @staticmethod
    def set_pinecone_api_key(api_key: str):
        os.environ['PINECONE_API_KEY'] = api_key

    def load_model(self, model_path: str):
        try:
            return CTransformers(model=model_path, callbacks=[StreamingStdOutCallbackHandler()])
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @staticmethod
    def connect_pinecone(index_name: str) -> Optional[PineconeVectorStore]:
        try:
            return PineconeVectorStore.from_existing_index(
                index_name=index_name, 
                embedding=HuggingFaceEmbeddings()
            )
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            return None

    def get_relevant_chunks(self, query: str, k: int = 1) -> List[str]:
        try:
            chunk_list = self.docsearch.similarity_search_with_relevance_scores(query, k=k)
            return [chunk[0].page_content for chunk in chunk_list if hasattr(chunk[0], 'page_content')]
        except Exception as e:
            print(f"Error getting relevant chunks: {e}")
            return []

    def get_relevant_urls(self, query: str, k: int = 1) -> List[str]:
        try:
            chunk_list = self.docsearch.similarity_search_with_relevance_scores(query, k=k)
            return [
                chunk[0].metadata['url']
                for chunk in chunk_list
                if hasattr(chunk[0], 'metadata') and 'url' in chunk[0].metadata
            ]
        except Exception as e:
            print(f"Error getting relevant URLs: {e}")
            return []

    @staticmethod
    def create_prompt(query: str, chunk_list: List[str]) -> str:
        chunk_text = "\n".join(chunk_list)
        return f"<s>[INST] Here is the question: {query}\nPlease, generate me the answer based on the following text chunks: \n{chunk_text}\n. [/INST]"

    def generate_answer(self, prompt: str) -> str:
        return self.llm(prompt)

    @staticmethod
    def generate_full_answer(llm_answer: str, urls: List[str]) -> str:
        return f"{llm_answer}\n\nSources: {', '.join(urls)}"

    @staticmethod
    def split_text(text: str, max_tokens: int) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_query(self, query: str, num_chunks: int = 1, num_urls: int = 1) -> str:
        relevant_chunks = self.get_relevant_chunks(query, k=num_chunks)
        if not relevant_chunks:
            return "No relevant information found."

        relevant_urls = self.get_relevant_urls(query, k=num_urls)

        prompt = self.create_prompt(query, relevant_chunks)
        prompt_chunks = self.split_text(prompt, self.max_tokens)

        responses = []
        for chunk in prompt_chunks:
            responses.append(self.generate_answer(chunk))

        llm_answer = ' '.join(responses)
        return self.generate_full_answer(llm_answer, relevant_urls)