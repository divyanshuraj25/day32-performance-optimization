from __future__ import annotations
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from .config import settings
from .models import AnswerResponse, ProfileResponse, QueryRequest
from .pipeline import PerformancePipeline

app = FastAPI(title="Day 32 — AI Performance Optimization API", version="1.0.0")

client = None
if settings.api_key:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.api_key)

pipeline = PerformancePipeline(client)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "real" if client else "mock",
        "embedding_model": settings.embedding_model,
        "fast_model": settings.fast_model,
    }

@app.post("/v1/answer", response_model=AnswerResponse)
async def answer(request: QueryRequest):
    result = await pipeline.run(request.query, request.use_small_model)
    return AnswerResponse(
        answer=result.answer,
        model=result.model,
        timings_ms=result.timings_ms,
        total_ms=result.total_ms,
        sources=result.sources,
    )

@app.post("/v1/profile", response_model=ProfileResponse)
async def profile(request: QueryRequest):
    result = await pipeline.run(request.query, request.use_small_model)
    total = result.total_ms or 1
    breakdown = {
        name: round(value / total * 100, 2)
        for name, value in result.timings_ms.items()
        if name != "total_pipeline"
    }
    return ProfileResponse(
        query=request.query,
        timings_ms=result.timings_ms,
        total_ms=result.total_ms,
        percentage_breakdown=breakdown,
    )

@app.post("/v1/answer/stream")
async def stream_answer(request: QueryRequest):
    async def event_stream():
        first_token = True
        async for chunk, elapsed_ms in pipeline.stream(
            request.query, request.use_small_model
        ):
            payload = {
                "token": chunk,
                "elapsed_ms": round(elapsed_ms, 2),
                "first_token": first_token,
            }
            first_token = False
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
