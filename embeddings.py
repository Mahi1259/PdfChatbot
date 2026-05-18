"""FAISS-backed semantic search over PDF chunks."""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_processor import Chunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")

        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.chunks = chunks

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        if self.index is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        k = min(k, len(self.chunks))
        scores, ids = self.index.search(q_emb, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results
