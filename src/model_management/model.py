import os
from typing import List
from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings



class Model:
    def __init__(self, model_path: str, index_name: str, pinecone_api_key: str):
        self.set_pinecone_api_key(pinecone_api_key)
        self.llm = self.load_model(model_path)
        self.docsearch = self.connect_pinecone(index_name)

    @staticmethod
    def set_pinecone_api_key(api_key: str):
        """
        Set the Pinecone API key as an environment variable.
        
        Args:
            api_key (str): The Pinecone API key.
        """
        os.environ['PINECONE_API_KEY'] = api_key

    def load_model(self, model_path: str):
        """
        Load the language model.
        
        Args:
            model_path (str): The path to the model.
        
        Returns:
            CTransformers: The loaded model.
        """
        try:
            return CTransformers(model=model_path, callbacks=[StreamingStdOutCallbackHandler()])
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def connect_pinecone(self, index_name: str):
        """
        Connect to Pinecone index.
        
        Args:
            index_name (str): The name of the Pinecone index.
        
        Returns:
            PineconeVectorStore: The connected Pinecone vector store.
        """
        try:
            return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=HuggingFaceEmbeddings())
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            raise

    def get_relevant_chunks(self, query: str) -> List[str]:
        """
        Get relevant chunks of text based on the query.
        
        Args:
            query (str): The query to search for relevant chunks.
        
        Returns:
            List[str]: A list of relevant text chunks.
        """
        try:
            chunk_list = self.docsearch.similarity_search_with_relevance_scores(query)
            return [chunk[0].page_content for chunk in chunk_list[:1]]
        except Exception as e:
            print(f"Error getting relevant chunks: {e}")
            raise

    def create_prompt(self, query: str, chunk_list: List[str]) -> str:
        """
        Create a prompt for the model based on the query and text chunks.
        
        Args:
            query (str): The user's query.
            chunk_list (List[str]): A list of relevant text chunks.
        
        Returns:
            str: The generated prompt.
        """
        prompt = (f"<s>[INST] Here is the question: {query}\n"
                  f"Please, generate me the answer based on the following text chunks: \n{chunk_list}\n. [INST]")
        return prompt

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer based on the prompt.
        
        Args:
            prompt (str): The prompt to generate an answer for.
        
        Returns:
            str: The generated answer.
        """
        try:
            return self.llm(prompt)
        except Exception as e:
            print(f"Error generating answer: {e}")
            raise

# Example usage:
# Ensure that the PINECONE_API_KEY environment variable is set before initializing the Model
# You can set it outside the script or use the set_pinecone_api_key method
# model = Model(model_path="path_to_model", index_name="index_name", pinecone_api_key="your_pinecone_api_key")
# response = model.generate_answer(model.create_prompt("your_query", model.get_relevant_chunks("your_query")))
# print(response)