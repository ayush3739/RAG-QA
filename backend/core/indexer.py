from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import  QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import pickle, hashlib, os
from backend.core.utils import simple_tokenize
from dotenv import load_dotenv



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
        try:
            self._delete_existing_collection()

            # Step 1: load
            loader = PyPDFLoader(file_path=str(self.file_path))
            docs = loader.load()

            # Step 2: chunk
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=150,
            )
            chunks = text_splitter.split_documents(documents=docs)
            print(f"Total Chunks created: {len(chunks)}")

            # Stamp each chunk with a deterministic id before storing it anywhere.
            for chunk in chunks:
                normalized_text = " ".join(chunk.page_content.split())
                page_label = chunk.metadata.get("page_label", "")
                source = chunk.metadata.get("source", str(self.file_path))
                chunk_id = self.make_chunk_id(normalized_text, page_label, source)
                chunk.metadata["chunk_id"] = chunk_id

            # Build and persist BM25 index and metadata
            try:
                texts = [c.page_content for c in chunks]
                tokenized = [simple_tokenize(t) for t in texts]
                bm25 = BM25Okapi(tokenized)
                meta = [
                    {
                        "chunk_id": c.metadata.get("chunk_id"),
                        "page_content": t,
                        "page_label": c.metadata.get("page_label"),
                        "source": c.metadata.get("source"),
                    }
                    for t, c in zip(texts, chunks)
                ]
                os.makedirs("data/bm25", exist_ok=True)
                with open(f"data/bm25/{self.file_path.name}_bm25.pkl", "wb") as f:
                    pickle.dump({"bm25": bm25, "meta": meta}, f)
                print(f"✓ BM25 persisted → data/bm25/{self.file_path.name}_bm25.pkl")
            except Exception as e:
                print(f"Warning: failed to persist BM25 index: {e}")

            # Step 3: embed & index
            self.vector_db = QdrantVectorStore.from_documents(
                documents=chunks,
                url="http://localhost:6333",
                collection_name=self.file_path.name,
                embedding=self.embedding_model,
            )
            print(f"Indexing done → collection: '{self.file_path.name}'")
        except ConnectionError as e:
            raise RuntimeError(f"Unable to connect to Qdrant: {e}")
        except Exception as e:
            raise RuntimeError(f"Indexing failed: {str(e)}")

    def _delete_existing_collection(self):
        client = QdrantClient(url="http://localhost:6333")
        try:
            existing = [c.name for c in client.get_collections().collections]
            if self.file_path.name in existing:
                client.delete_collection(collection_name=self.file_path.name)
        except Exception as e:
            print(f"Error occurred while deleting collection: {e}")
        finally:
            client.close()

    def make_chunk_id(self, text: str, page_label: str, source: str):
        h = hashlib.sha256(f"{source}|{page_label}|{text}".encode("utf-8")).hexdigest()
        return h[:16]