from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
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
            You are a helpful multimodal RAG assistant.
            The retrieved context can come from both PDF pages and image-derived embeddings
            (OCR/visual descriptions extracted from uploaded images).

            Answer only from the provided context. Do not invent facts.
            cite where the information came from using available metadata such as
            page number for PDFs and file location/source for images.
            If the answer is not present in the context, clearly say that the context does not
            contain enough information.

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
            You are a helpful multimodal RAG assistant.
            The retrieved context can come from both PDF pages and image-derived embeddings
            (OCR/visual descriptions extracted from uploaded images).

            Answer only from the provided context. Do not invent facts.
            When relevant, cite where the information came from using available metadata such as
            page number for PDFs and file location/source for images.
            If the answer is not present in the context, clearly say that the context does not
            contain enough information.

            CONTEXT:
            {context}
        """
        stream = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":System_prompt},
                {"role":"user","content":query}
            ],
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def answer(self, query: str, k: int = 10) -> str:
        context = self.similarity_search(query, k)
        return self.generate_response(query, context)

    def answer_stream(self, query: str, k: int = 10):
        """Pipeline: retrieve context then stream LLM response."""
        context = self.similarity_search(query, k)
        yield from self.stream_response(query, context)

