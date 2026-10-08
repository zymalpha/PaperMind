from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .models import Chunk, SearchResult
from .vector_store import ChromaVectorStore


def tokenize(text: str) -> list[str]:
    """Tokenize Latin terms and Chinese words/bigrams without external corpus data."""
    tokens: list[str] = []
    for run in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9_+.-]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", run):
            if len(run) <= 2:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run)
    return tokens


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.avg_len = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        self.df: Counter[str] = Counter(token for doc in self.tokens for token in set(doc))

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        n = len(self.chunks)
        for index, doc_tokens in enumerate(self.tokens):
            frequencies = Counter(doc_tokens)
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * len(doc_tokens) / max(self.avg_len, 1))
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        maximum = scores[0][1] if scores else 1.0
        return [SearchResult(self.chunks[index], score / maximum) for index, score in scores[:top_k]]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], rrf_k: int = 60, weights: Sequence[float] | None = None
) -> list[SearchResult]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings")
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk.id] = scores.get(result.chunk.id, 0.0) + weight / (rrf_k + rank)
            chunks[result.chunk.id] = result.chunk
    ordered = sorted(scores, key=scores.get, reverse=True)
    maximum = max((scores[key] for key in ordered), default=1.0)
    return [SearchResult(chunks[key], scores[key] / maximum) for key in ordered]


class Reranker(Protocol):
    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]: ...


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu") -> None:
        try:
            from sentence_transformers import CrossEncoder
            from huggingface_hub import snapshot_download

            # Resolve only from the persistent Hugging Face cache. Model downloads
            # are an explicit setup task, never an implicit application startup side effect.
            model_path = Path(model_name)
            project_model = Path(__file__).resolve().parents[2] / "models" / "bge-reranker-base"
            if not model_path.is_dir() and project_model.is_dir():
                model_path = project_model
            if not model_path.is_dir():
                model_path = Path(snapshot_download(repo_id=model_name, local_files_only=True))
            if not (model_path / "config.json").is_file() or not any(
                (model_path / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")
            ):
                raise FileNotFoundError("model config or weight file is missing from the local snapshot")
            self._model = CrossEncoder(
                str(model_path), device=device,
                automodel_args={"token": False}, local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"BGE Reranker 本地模型不可用 ({model_name})；请先运行 scripts/download_reranker.py。"
            ) from exc
        self.model_name = model_name

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, result.chunk.text) for result in results]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(results, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
        return [SearchResult(result.chunk, _sigmoid(float(score))) for result, score in ranked[:top_k]]


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class HybridRetriever:
    MODES = {"vector", "hybrid", "hybrid_rerank"}

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.vector_store, self.reranker = vector_store, reranker
        self.rrf_k, self.candidate_k = rrf_k, candidate_k
        self.last_warning: str | None = None

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid_rerank", document_id: str | None = None) -> list[SearchResult]:
        if mode not in self.MODES:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        vector = self.vector_store.query(query, self.candidate_k if mode != "vector" else top_k, document_id)
        if mode == "vector":
            return vector[:top_k]
        chunks = self.vector_store.all_chunks(document_id)
        bm25 = BM25Retriever(chunks).search(query, self.candidate_k)
        fused = reciprocal_rank_fusion([vector, bm25], self.rrf_k)
        self.last_warning = None
        if mode == "hybrid_rerank":
            if self.reranker is None:
                self.last_warning = "reranker_not_configured"
            else:
                try:
                    return self.reranker.rerank(query, fused[: self.candidate_k], top_k)
                except Exception as exc:
                    self.last_warning = f"reranker_unavailable:{type(exc).__name__}"
        return fused[:top_k]
