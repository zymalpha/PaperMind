from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedding:
    """Local embedding adapter. Document contents never leave this process."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers，请先执行 pip install -r requirements.txt"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            # BGE is public; explicitly disable implicit credentials so an expired
            # machine-wide Hugging Face token cannot break anonymous downloads.
            self._model = SentenceTransformer(model_name, device=device, token=False)
        except Exception as exc:
            raise RuntimeError(
                f"本地 Embedding 模型加载失败 ({model_name})。"
                "首次运行需联网下载，下载后可离线使用。"
            ) from exc

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("查询文本不能为空")
        return self.embed_documents([text])[0]


def create_embedding_provider(
    provider: str,
    model_name: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> EmbeddingProvider:
    normalized = provider.strip().lower()
    if normalized in {"local", "sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerEmbedding(model_name, device=device, batch_size=batch_size)
    raise ValueError(
        f"不支持的 Embedding provider: {provider}。"
        "第一阶段支持 local/sentence-transformers。"
    )
