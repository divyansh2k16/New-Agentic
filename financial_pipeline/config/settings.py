"""
Centralised application settings using Pydantic BaseSettings.
Reads from environment variables / .env file automatically.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Financial PDF Pipeline"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # LLM
    anthropic_api_key: str = ""
    primary_llm_model: str = "claude-sonnet-4-6"
    fast_llm_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "all-MiniLM-L6-v2"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "financial-pdf-pipeline"

    # Vector Store
    chroma_persist_dir: str = "./data/vectorstore"
    chroma_collection_name: str = "financial_documents"

    # Security
    secret_key: str = "dev-secret-change-in-prod"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Files
    upload_dir: str = "./data/pdfs"
    max_file_size_mb: int = 50
    allowed_extensions: str = ".pdf"

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # AWS (optional)
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def ensure_dirs(self):
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
