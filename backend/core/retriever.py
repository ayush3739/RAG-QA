from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
import pickle,os
from pathlib import Path
from backend.core.utils import simple_tokenize
load_dotenv("./.env")

class Retriever():
    def __init__(self, collection_name: str):
        try:
            # validate required env vars at instance creation
            self._validate_env()
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
            # Try to connect to Qdrant; if unavailable, fall back to BM25-only mode
            try:
                self.vector_db = QdrantVectorStore.from_existing_collection(
                    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                    collection_name=collection_name,
                    embedding=self.embedding_model,
                )
                self.qdrant_available = True
                self.qdrant_error = None
            except Exception as e:
                # Do not fail initialization; continue with BM25-only retriever
                self.vector_db = None
                self.qdrant_available = False
                self.qdrant_error = str(e)
                print(f"Warning: Qdrant not available. Proceeding without vector DB. Error: {self.qdrant_error}")
            bm25_file = Path("data/bm25") / f"{collection_name}_bm25.pkl"
            self.bm25 = None
            self.bm25_texts = []
            self.bm25_meta = []
            if bm25_file.exists():
                with open(bm25_file, "rb") as f:
                    d = pickle.load(f)
                    self.bm25 = d.get("bm25")
                    # meta is a list of dicts with page_content, page_label, source
                    self.bm25_meta = d.get("meta", [])
                    self.bm25_texts = [m.get("page_content", "") for m in self.bm25_meta]
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            raise RuntimeError(f"Retriever initialization failed: {str(e)}")

    @staticmethod
    def _validate_env():
        missing = [v for v in ["GITHUB_TOKEN"] if not os.getenv(v)]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

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

        def score_for_result(result):
            try:
                cid = result.metadata.get("chunk_id") or result.page_content
            except Exception:
                cid = result.page_content
            return scores.get(cid, 0.0)

        ordered = sorted(merged_map.values(), key=score_for_result, reverse=True)
        return ordered
    


    
    def similarity_search(self,query:str,k:int=10):
        try:
            query = self.sanitize_query(query)
            # Vector search only if Qdrant is available
            search_results = []
            if getattr(self, 'vector_db', None) is not None:
                try:
                    search_results = self.vector_db.similarity_search(query=query, k=max(k, 15))
                except Exception:
                    search_results = []

            # BM25 retrieval (if available)
            bm25_results = []
            try:
                if self.bm25 and len(self.bm25_texts) > 0:
                    q_tokens = simple_tokenize(query)
                    scores = self.bm25.get_scores(q_tokens)
                    # get top indices
                    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:15]
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

            merged_results = self.reciprocal_rank_fusion(search_results, bm25_results)

            # Rerank top merged results using CrossEncoder, fall back gracefully
            top_for_rerank = merged_results[:15]
            try:
                ranked_chunks, max_score = self.rerank_(query, top_for_rerank, top_n=10)
                chunks_for_context = ranked_chunks
            except Exception:
                chunks_for_context = top_for_rerank[:10]

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
                        "vector_score": None,
                    }
                )

            result_payload = {
                "chunks": structured_chunks,
                "used_vector_db": bool(getattr(self, 'qdrant_available', False)),
                "debug": {"qdrant_error": getattr(self, 'qdrant_error', None)},
                "confidence": float(max_score) if 'max_score' in locals() else None,
            }

            return result_payload
        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {str(e)}")
        
    def rerank_(self, query: str, chunks: list,top_n: int = 5) -> list:
        if not chunks:
            return [], 0.0
        pairs = [(query, c.page_content) for c in chunks]
        scores = self.reranker.predict(pairs)
        if len(scores) == 0:
            return chunks[:top_n], 0.0
        for c, s in zip(chunks, scores):
            try:
                c.metadata["reranker_score"] = float(s)
            except Exception:
                c.metadata = getattr(c, "metadata", {}) or {}
                c.metadata["reranker_score"] = float(s)
        ranked = sorted(chunks, reverse=True, key=lambda x: x.metadata.get("reranker_score", 0.0))
        return ranked[:top_n], float(max(scores))

    def generate_response(self, query: str, retrieval_result: dict):
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
                for c in chunks[:6]
            ]
        )

        system_prompt = f"""
    You are a helpful assistant that answers questions strictly based on context
retrieved from a PDF document.

Rules:
- Answer ONLY using the provided context chunks. Do not use prior knowledge.
- If the answer spans multiple chunks, synthesize them into one clear response.
- Always cite the relevant page number(s) at the end, e.g., (Page 4, 12).
- If chunks partially relate but don't fully answer the question, say what
  you found and note what's missing.
- If chunks contradict each other, mention both findings and their pages.
- If the context doesn't contain the answer, respond with:
  "I could not find this information in the provided document."
- Keep answers under 200 words unless the question requires more detail.
- Do not infer or extrapolate beyond what is explicitly stated in the chunks.

CONTEXT:
{context}
"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )

        answer_text = response.choices[0].message.content or ""
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


    def answer(self, query: str, k: int = 10) -> dict:
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

        retrieval_result = self.similarity_search(q, k)
        return self.generate_response(q, retrieval_result)
    


