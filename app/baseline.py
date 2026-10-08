from __future__ import annotations
from .pipeline import PerformancePipeline
from .retrieval import DOCUMENTS

class BaselinePipeline(PerformancePipeline):
    """Sequential embedding baseline used to compare async parallelism."""

    async def build_sequential_index(self):
        vectors = []
        for document in DOCUMENTS:
            vectors.append(await self.retriever._embed_one(document))
        return vectors
