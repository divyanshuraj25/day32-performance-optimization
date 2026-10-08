from __future__ import annotations
import asyncio
from dataclasses import dataclass
import faiss
import numpy as np
from openai import AsyncOpenAI
from .config import settings

DOCUMENTS = [
    "Async I/O allows independent network operations to overlap instead of waiting sequentially.",
    "Time-to-first-token measures how long a user waits before the first streamed model token arrives.",
    "FAISS provides efficient similarity search over dense vectors and is useful for retrieval augmented generation.",
    "Model selection can reduce latency when a smaller model is sufficient for a sub-task.",
    "Caching embeddings avoids repeatedly paying the network and compute cost for unchanged documents.",
    "FastAPI supports asynchronous endpoints and streaming responses for low-latency applications.",
    "Performance profiling should measure each stage separately before optimization.",
    "A good optimization hypothesis predicts both the expected latency reduction and the quality impact.",
]

@dataclass
class RetrievalResult:
    text: str
    score: float
    source: str

class Retriever:
    def __init__(self, client: AsyncOpenAI | None):
        self.client = client
        self.index = None
        self.vectors = None

    async def _embed_one(self, text: str) -> list[float]:
        if not self.client:
            seed = sum(ord(c) for c in text) % 997
            rng = np.random.default_rng(seed)
            vector = rng.normal(size=settings.embedding_dimensions).astype("float32")
            vector /= np.linalg.norm(vector)
            return vector.tolist()

        response = await self.client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimensions,
        )
        return response.data[0].embedding

    async def build_index(self) -> None:
        # Independent API calls are overlapped with asyncio.gather().
        vectors = await asyncio.gather(*(self._embed_one(doc) for doc in DOCUMENTS))
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self.vectors = matrix
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    async def search(self, query: str, k: int = 3) -> list[RetrievalResult]:
        if self.index is None:
            await self.build_index()

        query_vector = np.asarray([await self._embed_one(query)], dtype="float32")
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)

        return [
            RetrievalResult(DOCUMENTS[i], float(scores[0][rank]), f"doc-{i+1}")
            for rank, i in enumerate(indices[0])
            if i >= 0
        ]
