from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM



from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv("./.env")

class Retriver():
    def __init__(self, collection_name: str):
        self.openai_client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
        )

        self.llm = OllamaLLM(
            model="qwen3:4b",
            temperature=0.4,
            num_ctx=8192,
            num_predict=1024,
            repeat_penalty=1.05,
            base_url="http://localhost:11434"  # explicit is better
        )
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("GITHUB_TOKEN"),
            openai_api_base="https://models.github.ai/inference",
        )
        self.vector_db = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name=collection_name,
            embedding=self.embedding_model,
        )
    def similarity_search(self,query:str,k:int=10):
        search_results = self.vector_db.similarity_search(query=query , k=k)
        context= "\n\n\n".join([f"Page Content : {result.page_content} \n Page Number : {result.metadata['page_label']}\nfile Location : {result.metadata['source']}" for result in search_results])
        return context

    def generate_response(self,query:str,context:str):
        System_prompt=f"""
            You are a helpful assistant who answers user query based on the available context.
            retreived from the PDF file along with page_contents and page_number.

            You should only answer the user based on the following context and navigate the 
            user to open the right page number to know more about the topic.

            CONTEXT:
            {context}
        """
        response=self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":System_prompt},
                {"role":"user","content":query}
            ]
        )
        return response.choices[0].message.content

    def stream_response(self, query: str, context: str):
        """Yields text chunks for streaming UI."""
        System_prompt=f"""
            You are a helpful assistant who answers user query based on the available context.
            retreived from the PDF file along with page_contents and page_number.

            You should only answer the user based on the following context and navigate the 
            user to open the right page number to know more about the topic.

            CONTEXT:
            {context}
        """
        prompt = f"{System_prompt}\n\nUser Question:\n{query}"
        stream = self.llm.stream(prompt)
        for chunk in stream:
            yield chunk
        

    def answer(self, query: str, k: int = 10) -> str:
        context = self.similarity_search(query, k)
        return self.generate_response(query, context)

    def answer_stream(self, query: str, k: int = 10):
        """Pipeline: retrieve context then stream LLM response."""
        context = self.similarity_search(query, k)
        yield from self.stream_response(query, context)

