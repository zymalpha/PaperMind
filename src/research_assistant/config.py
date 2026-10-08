from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _nested(data: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return data.get(section, {}).get(key, default)


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    upload_dir: Path
    index_dir: Path
    log_dir: Path
    app_name: str
    deepseek_api_key: str = field(repr=False)
    llm_base_url: str
    llm_model: str
    llm_timeout_seconds: float
    llm_max_retries: int
    llm_temperature: float
    llm_max_tokens: int
    embedding_provider: str
    embedding_model: str
    embedding_device: str
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    collection_name: str
    min_relevance: float

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.upload_dir, self.index_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def llm_configured(self) -> bool:
        return bool(self.deepseek_api_key and self.deepseek_api_key != "your_deepseek_api_key_here")


def load_settings(config_path: Path | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    path = config_path or PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    data_dir = PROJECT_ROOT / str(_nested(data, "app", "data_dir", "data"))
    settings = Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        upload_dir=data_dir / "uploads",
        index_dir=data_dir / "index",
        log_dir=data_dir / "logs",
        app_name=str(_nested(data, "app", "name", "智能科研助理")),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        llm_base_url=os.getenv("DEEPSEEK_BASE_URL", str(_nested(data, "llm", "base_url", "https://api.deepseek.com"))),
        llm_model=os.getenv("DEEPSEEK_MODEL", str(_nested(data, "llm", "model", "deepseek-flash"))),
        llm_timeout_seconds=float(_nested(data, "llm", "timeout_seconds", 45)),
        llm_max_retries=int(_nested(data, "llm", "max_retries", 2)),
        llm_temperature=float(_nested(data, "llm", "temperature", 0.2)),
        llm_max_tokens=int(_nested(data, "llm", "max_tokens", 4096)),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", str(_nested(data, "embedding", "provider", "local"))),
        embedding_model=os.getenv("EMBEDDING_MODEL", str(_nested(data, "embedding", "model", "BAAI/bge-small-zh-v1.5"))),
        embedding_device=str(_nested(data, "embedding", "device", "cpu")),
        embedding_batch_size=int(_nested(data, "embedding", "batch_size", 32)),
        chunk_size=int(_nested(data, "chunking", "chunk_size", 700)),
        chunk_overlap=int(_nested(data, "chunking", "chunk_overlap", 120)),
        top_k=int(_nested(data, "retrieval", "top_k", 5)),
        collection_name=str(_nested(data, "retrieval", "collection_name", "research_papers")),
        min_relevance=float(_nested(data, "retrieval", "min_relevance", 0.12)),
    )
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    settings.ensure_directories()
    return settings
