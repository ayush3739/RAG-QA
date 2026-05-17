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
            self.vector_db = QdrantVectorStore.from_existing_collection(
                url="http://localhost:6333",
                collection_name=collection_name,
                embedding=self.embedding_model,
            )
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
        except ConnectionError as e:
            raise RuntimeError(f"Unable to connect to Qdrant: {e}")
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
                key = result.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

        add_results(vector_results)
        add_results(bm25_results)

        ordered = sorted(
            {result.page_content: result for result in list(vector_results) + list(bm25_results)}.values(),
            key=lambda result: scores.get(result.page_content, 0.0),
            reverse=True,
        )
        return ordered
    


    
    def similarity_search(self,query:str,k:int=10):
        try:
            query = self.sanitize_query(query)
            search_results = self.vector_db.similarity_search(query=query, k=max(k, 15))

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
                        bm25_results.append(_DocLike(self.bm25_texts[i], {"page_label": meta.get("page_label"), "source": meta.get("source")}))
            except Exception:
                bm25_results = []

            merged_results = self.reciprocal_rank_fusion(search_results, bm25_results)
            # Rerank top merged results using CrossEncoder, fall back gracefully
            top_for_rerank = merged_results[:15]
            try:
                ranked_chunks, _ = self.rerank_(query, top_for_rerank, top_n=10)
                chunks_for_context = ranked_chunks
            except Exception:
                chunks_for_context = top_for_rerank[:10]

            context = "\n\n\n".join([
                f"Page Content : {result.page_content} \n Page Number : {result.metadata['page_label']}\nfile Location : {result.metadata['source']}"
                for result in chunks_for_context
            ])
            return context
        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {str(e)}")
        
    def rerank_(self, query: str, chunks: list,top_n: int = 5) -> list:
        if not chunks:
            return [], 0.0
        pairs = [(query, c.page_content) for c in chunks]
        scores = self.reranker.predict(pairs)
        if len(scores) == 0:
            return chunks[:top_n], 0.0
        ranked = sorted(zip(scores, chunks), reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in ranked[:top_n]], float(max(scores))

    def generate_response(self,query:str,context:str):
        System_prompt=f"""
            You are a helpful assistant who answers user query based on the available context.
            retrieved from the PDF file along with page_contents and page_number.

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


    def answer(self, query: str, k: int = 10) -> str:
        context = self.similarity_search(query, k)
        return self.generate_response(query, context)
    


