import asyncio
from app.pipeline import PerformancePipeline

def test_mock_pipeline():
    result = asyncio.run(PerformancePipeline(None).run("How does async I/O reduce latency?"))
    assert result.answer
    assert result.total_ms > 0
    assert "llm_generation" in result.timings_ms

def test_mock_stream():
    async def collect():
        items = []
        async for token, elapsed in PerformancePipeline(None).stream("What is TTFT?"):
            items.append((token, elapsed))
        return items

    items = asyncio.run(collect())
    assert items
    assert items[0][1] > 0
