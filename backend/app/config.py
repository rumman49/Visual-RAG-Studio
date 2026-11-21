"""Validated application configuration with safe local defaults."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DATA_CANDIDATE = _BACKEND_DIR / "data"
_HAS_BACKEND_DATA = (_DATA_CANDIDATE / "visual_rag_studio.db").exists()

_DEFAULT_DATA_PATH = _DATA_CANDIDATE if _HAS_BACKEND_DATA else Path("./data")
_DEFAULT_DB_URL = f"sqlite:///{(_DATA_CANDIDATE / 'visual_rag_studio.db').as_posix()}" if _HAS_BACKEND_DATA else "sqlite:///./data/visual_rag_studio.db"
_DEFAULT_CHROMA_PATH = (_DATA_CANDIDATE / "chroma-recovered") if (_DATA_CANDIDATE / "chroma-recovered").exists() else Path("./data/chroma-recovered")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VRS_",
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Visual RAG Studio API"
    environment: str = Field(default="development", alias="VRS_ENV")
    host: str = "127.0.0.1"
    port: int = Field(default=8001, ge=1, le=65535)
    database_url: str = _DEFAULT_DB_URL
    data_path: Path = _DEFAULT_DATA_PATH
    chroma_path: Path = _DEFAULT_CHROMA_PATH
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str | None = None
    embedding_provider: Literal["sentence_transformers", "ollama", "fast_local"] = "sentence_transformers"
    sentence_transformers_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sentence_transformers_local_files_only: bool = False
    ollama_generation_model: str | None = None
    generation_provider: Literal["groq", "ollama", "huggingface", "extractive", "openai", "gemini", "anthropic"] = "groq"
    groq_generation_model: str = "llama-3.1-8b-instant"
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_timeout_seconds: int = Field(default=90, ge=1, le=600)
    openai_generation_model: str = "gpt-4o-mini"
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    gemini_generation_model: str = "gemini-2.0-flash"
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    anthropic_generation_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")
    hf_generation_model: str | None = None
    hf_provider_policy: Literal["fastest", "cheapest", "preferred"] = "fastest"
    generation_max_tokens: int = Field(default=512, ge=1, le=8192)
    generation_temperature: float = Field(default=0.2, ge=0, le=2)
    ollama_timeout_seconds: int = Field(default=90, ge=1, le=600)
    hf_timeout_seconds: int = Field(default=90, ge=1, le=600)
    max_context_chars: int = Field(default=18000, ge=1000, le=100000)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)

    @model_validator(mode="after")
    def require_protection_outside_loopback(self) -> "Settings":
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "Non-loopback binding is disabled until an authentication/access-control "
                "boundary is implemented. Set VRS_HOST=127.0.0.1 for local development."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
