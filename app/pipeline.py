from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable
from openai import AsyncOpenAI
from .config import settings
from .retrieval import Retriever, RetrievalResult

@dataclass
class PipelineResult:
    answer: str
    model: str
    timings_ms: dict[str, float]
    total_ms: float
    sources: list[str]

class PerformancePipeline:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client
        self.retriever = Retriever(client)
        self._mock = client is None

    async def _timed(self, name: str, timings: dict[str, float], fn: Callable):
        start = time.perf_counter()
        value = await fn()
        timings[name] = (time.perf_counter() - start) * 1000
        return value

    def _build_prompt(self, query: str, results: list[RetrievalResult]) -> str:
        context = "\n".join(f"[{r.source}] {r.text}" for r in results)
        return (
            "Answer the user's question using only the supplied context. "
            "Be concise and accurate.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )

    async def _complete(self, prompt: str, model: str) -> str:
        if self._mock:
            await asyncio.sleep(0.08)
            return "Async parallelism reduces latency by overlapping independent I/O operations."

        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise technical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    async def run(self, query: str, use_small_model: bool = True) -> PipelineResult:
        timings = {}
        start = time.perf_counter()

        results = await self._timed(
            "embedding_and_faiss_search",
            timings,
            lambda: self.retriever.search(query),
        )
        prompt = await self._timed(
            "prompt_construction",
            timings,
            lambda: asyncio.to_thread(self._build_prompt, query, results),
        )
        model = settings.fast_model if use_small_model else settings.quality_model
        answer = await self._timed(
            "llm_generation",
            timings,
            lambda: self._complete(prompt, model),
        )
        formatted = await self._timed(
            "response_formatting",
            timings,
            lambda: asyncio.to_thread(self._format, answer),
        )

        timings["total_pipeline"] = (time.perf_counter() - start) * 1000
        return PipelineResult(
            answer=formatted,
            model=model,
            timings_ms=timings,
            total_ms=timings["total_pipeline"],
            sources=[r.source for r in results],
        )

    @staticmethod
    def _format(answer: str) -> str:
        return answer.strip()

    async def stream(self, query: str, use_small_model: bool = True) -> AsyncIterator[tuple[str, float]]:
        start = time.perf_counter()
        results = await self.retriever.search(query)
        prompt = self._build_prompt(query, results)
        model = settings.fast_model if use_small_model else settings.quality_model

        if self._mock:
            for token in "Async parallelism reduces latency by overlapping independent I/O operations.".split():
                await asyncio.sleep(0.03)
                yield token + " ", (time.perf_counter() - start) * 1000
            return

        stream = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise technical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta, (time.perf_counter() - start) * 1000
