from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration for DocuMind API."""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # API Keys (optional at runtime; required only for GitHub-backed features)
    github_token: Optional[str] = None
    test_key: Optional[str] = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    # Web Search (Tavily)
    tavily_api_key: str = ""
    enable_web_search: bool = True

    # RAG Configuration
    top_k: int = 15
    rerank_top_n: int = 5
    confidence_threshold: float = 0.3
    chunk_size: int = 600
    chunk_overlap: int = 150

    # Database URL
    DATABASE_URL: str 

    model_config = {"env_file": "../.env", "extra": "ignore"}



settings = Settings()
