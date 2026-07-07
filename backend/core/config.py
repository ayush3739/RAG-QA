from pydantic import SecretStr
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """Configuration for DocuMind API."""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # --- LLM API Keys ---
    github_token: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    open_router_key: Optional[str] = None

    # Active provider/model (overridden at the bottom of this file)
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"

    # Auth
    test_key: Optional[str] = None
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week

    # Ollama (local)
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

    # Mail
    mail_server: str = "localhost"
    mail_port: int = 587
    mail_username: str = ""
    mail_password: SecretStr = SecretStr("")
    mail_from: str = "noreply@example.com"
    mail_use_tls: bool = True

    # Database
    DATABASE_URL: str

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        extra="ignore",
        # Allow case-insensitive env var matching (Gemini_api_key → gemini_api_key)
        case_sensitive=False,
    )


settings = Settings()

# Active provider — Gemini first since Groq has hit its free daily limit.
# Fallback chain: Gemini → Groq → OpenRouter → GitHub AI → Ollama
settings.llm_provider = "gemini"
settings.llm_model = "gemini-2.0-flash"