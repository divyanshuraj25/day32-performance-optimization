from __future__ import annotations
import asyncio, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.baseline import BaselinePipeline
from app.pipeline import PerformancePipeline

async def main():
    baseline = BaselinePipeline(None)
    optimized = PerformancePipeline(None)

    t = time.perf_counter()
    await baseline.build_sequential_index()
    sequential = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    await optimized.retriever.build_index()
    parallel = (time.perf_counter() - t) * 1000

    reduction = ((sequential - parallel) / sequential * 100) if sequential else 0
    print(f"Sequential embedding/index build: {sequential:.2f} ms")
    print(f"Parallel embedding/index build:   {parallel:.2f} ms")
    print(f"Measured reduction:                {reduction:.2f}%")
    print("Mock-mode values are demonstrations; real API latency should be measured separately.")

if __name__ == "__main__":
    asyncio.run(main())
