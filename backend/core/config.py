
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
    jina_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    groq_api_secondary: Optional[str] = None
    groq_api_third: Optional[str] = None
    gemini_api_key: Optional[str] = None
    open_router_key: Optional[str] = None
    nvidia_nim: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    aws_region_name: Optional[str] = "us-east-1"

    # Active provider/model (overridden at the bottom of this file)
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"

    # Auth - GitHub
    GITHUB_CLIENT_ID: SecretStr = SecretStr("")
    GITHUB_CLIENT_SECRET: SecretStr = SecretStr("")
    GITHUB_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/github/callback"

    # Auth - Google
    GOOGLE_CLIENT_ID: SecretStr = SecretStr("")
    GOOGLE_CLIENT_SECRET: SecretStr = SecretStr("")
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"

    # Cookie - AUTH
    cookie_secure: bool = False  # Set to True in production (HTTPS)
    cookie_domain: Optional[str] = None
    cookie_samesite: str = "lax"

    # Frontend Redirect Base URL
    FRONTEND_URL: str = "http://localhost:3000"
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    issuer: str = "rag-qa-v1"

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    # Web Search (Tavily)
    tavily_api_key: str = ""
    enable_web_search: bool = True

    # RAG Configuration
    reranker_input_chunks: int = 25  # Chunks to send to the reranker
    vector_output_chunks: int = 20  # Number of chunks to retrieve from vector DB
    llm_context_chunks: int = 10    # Chunks to give to the LLM
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
    resend_api_key: SecretStr = SecretStr("")

    # Database
    DATABASE_URL: str

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        extra="ignore",
        # Allow case-insensitive env var matching (Gemini_api_key → gemini_api_key)
        case_sensitive=False,
    )


settings = Settings()

# Active provider — Groq first.
# Fallback chain: Groq → Groq Secondary
settings.llm_provider = "groq"
settings.llm_model = "llama-3.3-70b-versatile"