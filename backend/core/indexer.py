import asyncio
import hashlib
import os
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.utils import simple_tokenize
from backend.models.models import Chunk
from rank_bm25 import BM25Okapi


class Indexer:
    def __init__(self, file_path: str, db_session: AsyncSession, document_id: int):
        self.file_path = Path(file_path)
        self.db = db_session
        self.document_id = document_id
        self.embedding_model = OpenAIEmbeddings(
            api_key=settings.github_token,
            model="text-embedding-3-small",
            openai_api_base="https://models.github.ai/inference",
        )

    async def index(self):
        try:
            # Step 1: load PDF — CPU/IO bound, offload to thread
            loader = PyPDFLoader(file_path=str(self.file_path))
            docs = await asyncio.to_thread(loader.load)

            # Step 2: chunk — CPU bound, offload to thread
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=150,
            )
            chunks = await asyncio.to_thread(text_splitter.split_documents, docs)
            print(f"Total chunks created: {len(chunks)}")

            # Stamp deterministic chunk_id on each chunk
            for chunk in chunks:
                normalized_text = " ".join(chunk.page_content.split())
                page_label = chunk.metadata.get("page_label", "")
                source = chunk.metadata.get("source", str(self.file_path))
                chunk.metadata["chunk_id"] = self.make_chunk_id(
                    normalized_text, page_label, source
                )

            # Step 3: BM25 — CPU bound, offload to thread
            await asyncio.to_thread(self._persist_bm25, chunks)

            # Step 4: embed — network + CPU bound, offload to thread
            texts = [c.page_content for c in chunks]
            embeddings = await asyncio.to_thread(
                self.embedding_model.embed_documents, texts
            )

            # Step 5: bulk insert chunks — pure async DB write
            rows = [
                Chunk(
                    document_id=self.document_id,
                    chunk_id=ch.metadata["chunk_id"],
                    page_content=ch.page_content,
                    page_number=ch.metadata.get("page"),
                    chunk_index=idx,
                    source=ch.metadata.get("source"),
                    embedding=emb,
                )
                for idx, (ch, emb) in enumerate(zip(chunks, embeddings))
            ]

            self.db.add_all(rows)
            await self.db.commit()
            print(f"✓ {len(rows)} chunks committed to DB for document_id={self.document_id}")

        except Exception as e:
            await self.db.rollback()
            raise RuntimeError(f"Indexing failed: {e}")

    def _persist_bm25(self, chunks) -> None:
        """Sync — called via asyncio.to_thread."""
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
            path = f"data/bm25/{self.document_id}_bm25.pkl"
            with open(path, "wb") as f:
                pickle.dump({"bm25": bm25, "meta": meta}, f)
            print(f"✓ BM25 persisted → {path}")
        except Exception as e:
            print(f"Warning: BM25 persist failed: {e}")

    def make_chunk_id(self, text: str, page_label: str, source: str) -> str:
        h = hashlib.sha256(
            f"{source}|{page_label}|{text}".encode("utf-8")
        ).hexdigest()
        return h[:16]