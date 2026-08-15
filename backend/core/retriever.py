from langchain_community.embeddings import JinaEmbeddings
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import pickle
from backend.services.llm_provider import LLMProvider
import asyncio
from pathlib import Path
from backend.core.utils import simple_tokenize
from backend.core.config import settings
from backend.models import models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import time

class RetrievalQuery(BaseModel):
    original_query: str
    semantic_query: str
    keyword_query: str

t0 = time.perf_counter()
class Retriever():
    def __init__(self, document_ids: list[int], db: AsyncSession):
        try:
            self.document_ids = document_ids
            self.db = db
            self.llm_provider = LLMProvider()

            self.llm = OllamaLLM(
                model=settings.ollama_model,
                temperature=0.4,
                num_ctx=8192,
                num_predict=1024,
                repeat_penalty=1.05,
                base_url=settings.ollama_base_url,
            )
            self.embedding_model = JinaEmbeddings(
                jina_api_key=settings.jina_key,
                model_name="jina-embeddings-v3",
            )
            BASE_DIR = Path(__file__).resolve().parent.parent
            
            self.bm25 = None
            self.bm25_texts = []
            self.bm25_meta = []
            self.bm25_loaded = False
            self.base_dir = BASE_DIR
            
            # Setup Groq clients for round-robin query rewriting
            self.groq_clients = []
            if settings.groq_api_key:
                self.groq_clients.append(ChatOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1", model="llama-3.1-8b-instant", temperature=0.3, max_tokens=100))
            if settings.groq_api_secondary:
                self.groq_clients.append(ChatOpenAI(api_key=settings.groq_api_secondary, base_url="https://api.groq.com/openai/v1", model="llama-3.1-8b-instant", temperature=0.3, max_tokens=100))
            if settings.groq_api_third:
                self.groq_clients.append(ChatOpenAI(api_key=settings.groq_api_third, base_url="https://api.groq.com/openai/v1", model="llama-3.1-8b-instant", temperature=0.3, max_tokens=100))
            self._groq_index = 0
            
        except Exception as e:
            raise RuntimeError(f"Retriever initialization failed: {str(e)}")

    @staticmethod
    def _validate_env():
        # kept for compatibility; settings.github_token is optional now
        return True

    def sanitize_query(self, query: str) -> str:
        if len(query) >1000:
            query = query[:1000]
        injection_patterns = ["ignore all instructions","ignore previous", "you are now"]
        if any(p in query.lower() for p in injection_patterns):
            raise ValueError("Invalid query detected.")
        return query.strip()

    def reciprocal_rank_fusion(self, vector_results, bm25_results, k: int = 60):
        """Merge ranked lists using Reciprocal Rank Fusion.

        This keeps the hybrid-search step working even when one side is weaker
        or when we only have partial ranked lists available.
        """
        
        scores = {}

        def add_results(results):
            for rank, result in enumerate(results, start=1):
                # prefer stable chunk_id when available for deduplication
                try:
                    key = result.metadata.get("chunk_id")
                except Exception:
                    key = None
                if not key:
                    key = result.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

        add_results(vector_results)
        add_results(bm25_results)

        # merge by chunk id when available to avoid duplicates
        merged_map = {}
        for result in list(vector_results) + list(bm25_results):
            try:
                cid = result.metadata.get("chunk_id") or result.page_content
            except Exception:
                cid = result.page_content
            if cid not in merged_map:
                merged_map[cid] = result
            else:
                existing = merged_map[cid]

                if result.metadata.get("bm25_score") is not None:
                    existing.metadata["bm25_score"] = result.metadata["bm25_score"]

                if result.metadata.get("vector_score") is not None:
                    existing.metadata["vector_score"] = result.metadata["vector_score"]

        def score_for_result(result):
            try:
                cid = result.metadata.get("chunk_id") or result.page_content
            except Exception:
                cid = result.page_content
            return scores.get(cid, 0.0)

        ordered = sorted(merged_map.values(), key=score_for_result, reverse=True)
        return ordered
    

    async def _load_bm25_from_db(self):
        if self.bm25_loaded:
            return
        self.bm25_loaded = True
        try:
            result = await self.db.execute(
                select(models.Document).where(models.Document.id.in_(self.document_ids))
            )
            docs = result.scalars().all()
            for doc in docs:
                if not doc.bm25_path: continue
                bm25_file = self.base_dir / doc.bm25_path
                if bm25_file.exists():
                    with open(bm25_file, "rb") as f:
                        d = pickle.load(f)
                        meta = d.get("meta", [])
                        self.bm25_meta.extend(meta)
                        self.bm25_texts.extend([m.get("page_content", "") for m in meta])
                else:
                    print(f"can't load the bm25 for document {doc.id} at {bm25_file}")
            
            if self.bm25_texts:
                tokenized_corpus = [simple_tokenize(doc) for doc in self.bm25_texts]
                self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            print(f"Failed to load BM25 from DB paths: {e}")



    async def similarity_search(self, query: str | RetrievalQuery, k: int = 10):
        await self._load_bm25_from_db()
        try:
            if isinstance(query, str):
                query = self.sanitize_query(query)
                semantic_query = query
                keyword_query = query
                original_query = query
            else:
                semantic_query = query.semantic_query
                keyword_query = query.keyword_query
                original_query = query.original_query

            t0 = time.perf_counter()
            query_embedding = await self.embedding_model.aembed_query(semantic_query)
            print("Embedding:", time.perf_counter() - t0)

            t1 = time.perf_counter()
            class _DocLike:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata

            vector_results = []

            try:
                results = await self.db.execute(
                    select(
                        models.Chunk,
                        models.Chunk.embedding.cosine_distance(
                            query_embedding
                        ).label("distance")
                    )
                    .where(
                        models.Chunk.document_id.in_(self.document_ids)
                    )
                    .order_by("distance")
                    .limit(max(k, settings.vector_output_chunks))
                )

                rows = results.all()

                for chunk, distance in rows:
                    vector_results.append(
                        _DocLike(
                            chunk.page_content,
                            {
                                "chunk_id": chunk.chunk_id,
                                "page_label": chunk.page_number,
                                "source": chunk.source,
                                "vector_score": round(
                                    1 - float(distance), 4),
                            }
                        )
                    )

            except Exception as e:
                print(f"Vector search failed: {e}")
                vector_results = []
            print("Vector Search:", time.perf_counter() - t1)
            t2 = time.perf_counter()
            # BM25 retrieval continues here...
            bm25_results = []
            try:
                if self.bm25 and len(self.bm25_texts) > 0:
                    combined_bm25_query = f"{original_query} {keyword_query}"
                    q_tokens = simple_tokenize(combined_bm25_query)
                    scores = self.bm25.get_scores(q_tokens)
                    # get top indices
                    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:settings.vector_output_chunks]
                    class _DocLike:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata

                    for i in ranked_idx:
                        meta = self.bm25_meta[i]
                        bm25_results.append(
                            _DocLike(
                                self.bm25_texts[i],
                                {
                                    "chunk_id": meta.get("chunk_id"),
                                    "page_label": meta.get("page_label"),
                                    "source": meta.get("source"),
                                    "bm25_score": float(scores[i]) if scores is not None else None,
                                },
                            )
                        )
            except Exception:
                bm25_results = []

            print("BM25:", time.perf_counter() - t2)

            merged_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
            
            t3 = time.perf_counter()
            # Rerank top merged results using CrossEncoder, fall back gracefully
            top_for_rerank = merged_results[:settings.reranker_input_chunks]
            try:
                ranked_chunks, max_score = await self.rerank_(original_query, top_for_rerank, top_n=settings.llm_context_chunks)
                chunks_for_context = ranked_chunks
            except Exception as e:
                import traceback
                print(f"Reranker failed: {e}")
                print(traceback.format_exc())
                chunks_for_context = top_for_rerank[:10]
            print("Rerank:", time.perf_counter() - t3)

            # Build structured chunk list for output
            structured_chunks = []
            for c in chunks_for_context:
                meta = getattr(c, 'metadata', {}) or {}
                structured_chunks.append(
                    {
                        "chunk_id": meta.get("chunk_id"),
                        "page_label": meta.get("page_label"),
                        "source": meta.get("source"),
                        "text": c.page_content,
                        "bm25_score": meta.get("bm25_score"),
                        "reranker_score": float(meta.get("reranker_score", 0.0)) if meta.get("reranker_score") is not None else None,
                        "vector_score": meta.get("vector_score"),
                        "bm25_len": len(bm25_results)
                    }
                )

            debug_info = {
                "vector_results": [{"chunk_id": getattr(c, "metadata", {}).get("chunk_id"), "score": getattr(c, "metadata", {}).get("vector_score"), "text": getattr(c, "page_content", "")} for c in vector_results],
                "bm25_results": [{"chunk_id": getattr(c, "metadata", {}).get("chunk_id"), "score": getattr(c, "metadata", {}).get("bm25_score"), "text": getattr(c, "page_content", "")} for c in bm25_results],
                "merged_results": [{"chunk_id": getattr(c, "metadata", {}).get("chunk_id"), "vector_score": getattr(c, "metadata", {}).get("vector_score"), "bm25_score": getattr(c, "metadata", {}).get("bm25_score"), "text": getattr(c, "page_content", "")} for c in merged_results],
                "top_for_rerank": [{"chunk_id": getattr(c, "metadata", {}).get("chunk_id"), "vector_score": getattr(c, "metadata", {}).get("vector_score"), "bm25_score": getattr(c, "metadata", {}).get("bm25_score"), "reranker_score": getattr(c, "metadata", {}).get("reranker_score"), "text": getattr(c, "page_content", "")} for c in top_for_rerank]
            }

            result_payload = {
                "chunks": structured_chunks,
                "used_vector_db":len(vector_results) > 0,
                "debug": debug_info,
                "confidence": float(max_score) if 'max_score' in locals() else None,
            }
            print("Total:", time.perf_counter() - t0)
            return result_payload
        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {str(e)}")
            
    async def rerank_(self, query: str, chunks: list,top_n: int = 5) -> list:
        if not chunks:
            return [], 0.0
            
        passages = [c.page_content for c in chunks]
        
        if not settings.jina_key:
            print("Warning: JINA_KEY not set. Skipping reranker.")
            return chunks[:top_n], 0.0

        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.jina.ai/v1/rerank",
                    headers={
                        "Authorization": f"Bearer {settings.jina_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "jina-reranker-v3",
                        "query": query,
                        "documents": passages,
                        "top_n": len(passages)
                    },
                    timeout=15.0
                )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
        except Exception as e:
            print(f"Jina API failed: {e}")
            return chunks[:top_n], 0.0
        
        if not results:
            return chunks[:top_n], 0.0

        # Sort the results by index to match original 'chunks' array order,
        # or we can just apply the scores to chunks by mapping index back.
        # Jina returns them sorted by relevance_score, but gives us the 'index' of the original document.
        scores_by_index = {item["index"]: item["relevance_score"] for item in results}
        
        for i, c in enumerate(chunks):
            s = scores_by_index.get(i, 0.0)
            try:
                c.metadata["reranker_score"] = float(s)
            except Exception:
                c.metadata = getattr(c, "metadata", {}) or {}
                c.metadata["reranker_score"] = float(s)
        
        ranked = sorted(chunks, reverse=True, key=lambda x: x.metadata.get("reranker_score", 0.0))
        return ranked[:top_n], float(ranked[0].metadata.get("reranker_score", 0.0))

    async def generate_response(self, query: str, retrieval_result: dict):
        # defensive sanitization: ensure the query used with the LLM is safe
        try:
            query = self.sanitize_query(query)
        except Exception:
            # if the query is invalid, blank it so the model sees only the context
            query = ""

        chunks = retrieval_result.get("chunks", []) if isinstance(retrieval_result, dict) else []
        context = "\n\n".join(
            [
                f"[chunk_id={c.get('chunk_id')} | page={c.get('page_label')} | source={c.get('source')}] {c.get('text', '')}"
                for c in chunks[:8]
            ]
        )

        system_prompt = f"""
    You are a helpful assistant that answers questions strictly based on context
    retrieved from the uploaded document(s).

Rules:
- Answer ONLY using the provided context chunks. Do not use prior knowledge.
- If the answer spans multiple chunks, synthesize them into one clear response.
- Always cite the relevant page number(s) (if available) at the end, e.g., (Page 4, 12).
- If chunks partially relate but don't fully answer the question, say what
  you found and note what's missing.
- If chunks contradict each other, mention both findings and their pages.
- If the context doesn't contain the answer, respond with:
  "I could not find this information in the provided document."
- Keep answers under 200 words by default.
- If the user asks for a summary, detailed explanation, in-depth answer, or specifies a longer length, provide the requested depth up to 1000 words.
- If the user asks for more than 1000 words, keep the answer under 1000 words and focus on the most useful details.
- Do not infer or extrapolate beyond what is explicitly stated in the chunks.

CONTEXT:
{context}
"""

        response_text = await self.llm_provider.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ])
        
        answer_text = response_text or ""
        citations = [
            {
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "page_label": c.get("page_label"),
                "excerpt": c.get("text", "")[:240],
            }
            for c in chunks[:5]
        ]

        return {
            "answer": answer_text,
            "citations": citations,
            "used_vector_db": retrieval_result.get("used_vector_db", False),
            "chunks": chunks,
            "debug": retrieval_result.get("debug", {}),
            "confidence": retrieval_result.get("confidence"),
        }


    async def answer(self, query: str, k: int = 10) -> dict:
        try:
            q = self.sanitize_query(query)
        except ValueError:
            return {
                "answer": "Invalid query detected.",
                "citations": [],
                "used_vector_db": False,
                "chunks": [],
                "debug": {"error": "invalid_query"},
                "confidence": None,
            }

        retrieval_result = await self.similarity_search(q, k)
  



        return await self.generate_response(q, retrieval_result)
    


