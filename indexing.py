from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import  QdrantVectorStore
from dotenv import load_dotenv
import os


load_dotenv('./.env')

class Indexer():
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.embedding_model = OpenAIEmbeddings(
            api_key=os.getenv('GITHUB_TOKEN'),
            model="text-embedding-3-large",
            openai_api_base="https://models.github.ai/inference",
        )

    def index(self):
        # Step 1: load
        loader = PyPDFLoader(file_path=str(self.file_path))
        docs = loader.load()

        # Step 2: chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400,
        )
        chunks = text_splitter.split_documents(documents=docs)
        print(f"Total Chunks created: {len(chunks)}")

        # Step 3: embed & index
        self.vector_db = QdrantVectorStore.from_documents(
            documents=chunks,
            url="http://localhost:6333",
            collection_name=self.file_path.name,
            embedding=self.embedding_model,
        )
        print(f"Indexing done → collection: '{self.file_path.name}'")