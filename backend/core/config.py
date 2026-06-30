from pydantic import SecretStr, Field, AliasChoices
from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """Configuration for DocuMind API."""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # API Keys (optional at runtime; required only for GitHub-backed features)

    github_token: Optional[str] = None
    test_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    llm_provider: str = "github"
    llm_model : str ="gpt-4o-mini" or "openai/gpt-oss-20b"
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week

    # Clerk Auth
    clerk_publishable_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("clerk_publishable_key", "CLERK_PUBLISHABLE_KEY", "VITE_CLERK_PUBLISHABLE_KEY")
    )
    clerk_secret_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("clerk_secret_key", "CLERK_SECRET_KEY")
    )


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

    #mailtrap
    mail_server: str = "localhost"
    mail_port: int = 587
    mail_username: str = ""
    mail_password: SecretStr = SecretStr("")
    mail_from: str = "noreply@example.com"
    mail_use_tls: bool = True

    # Database URL
    DATABASE_URL: str 

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        extra="ignore"
    )



settings = Settings()
