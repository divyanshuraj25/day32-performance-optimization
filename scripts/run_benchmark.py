from __future__ import annotations
import argparse, asyncio, csv, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.pipeline import PerformancePipeline

QUERIES = [
    "How does async I/O reduce latency?",
    "What is time-to-first-token?",
    "Why is FAISS useful for RAG?",
    "When should a smaller model be used?",
    "Why cache embeddings?",
    "What does FastAPI streaming provide?",
    "How should an AI pipeline be profiled?",
    "What is a good optimization hypothesis?",
    "How can independent API calls be parallelized?",
    "Why compare TTFT with full response time?",
    "What is vector similarity search?",
    "Why does model selection affect latency?",
    "What does response formatting cost?",
    "How can a pipeline avoid repeated embedding work?",
    "Why measure before and after optimization?",
    "What is the difference between TTFT and total latency?",
    "Why is sequential I/O slower?",
    "How does asyncio.gather work?",
    "What is a production performance baseline?",
    "Why should quality be checked after optimization?",
]

async def main(mode: str, count: int):
    client = None
    if mode == "real":
        if not settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.api_key)

    pipeline = PerformancePipeline(client)
    rows = []

    for i, query in enumerate(QUERIES[:count], 1):
        start = time.perf_counter()
        result = await pipeline.run(query, use_small_model=True)
        wall = (time.perf_counter() - start) * 1000
        rows.append({
            "query_number": i,
            "query": query,
            "embedding_and_faiss_search_ms": result.timings_ms.get("embedding_and_faiss_search", 0),
            "prompt_construction_ms": result.timings_ms.get("prompt_construction", 0),
            "llm_generation_ms": result.timings_ms.get("llm_generation", 0),
            "response_formatting_ms": result.timings_ms.get("response_formatting", 0),
            "total_pipeline_ms": result.total_ms,
            "wall_clock_ms": wall,
            "model": result.model,
        })
        print(f"[{i:02d}/{count}] {wall:.2f} ms | {result.model}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "benchmark_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    stages = [
        "embedding_and_faiss_search_ms",
        "prompt_construction_ms",
        "llm_generation_ms",
        "response_formatting_ms",
        "total_pipeline_ms",
    ]
    averages = {s: statistics.mean(r[s] for r in rows) for s in stages}
    total = averages["total_pipeline_ms"] or 1
    breakdown = {
        s: round(v / total * 100, 2)
        for s, v in averages.items() if s != "total_pipeline_ms"
    }

    with (report_dir / "before_after.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "optimized_average_ms", "percentage_of_pipeline"])
        for stage, value in averages.items():
            writer.writerow([stage, round(value, 3), breakdown.get(stage, 100)])

    lines = [
        "# Performance Profiling Report",
        "",
        f"Mode: **{mode}**",
        f"Queries measured: **{count}**",
        "",
        "## Average Latency",
        "",
    ]
    for stage, value in averages.items():
        lines.append(f"- `{stage}`: **{value:.2f} ms**")
    lines += ["", "## Percentage Breakdown", ""]
    for stage, pct in breakdown.items():
        lines.append(f"- `{stage}`: **{pct:.2f}%** of total pipeline time")
    lines += [
        "",
        "## Optimization Hypotheses",
        "",
        "1. LLM generation is expected to be a major latency contributor; use `gpt-4o-mini` when full capability is unnecessary.",
        "2. Independent document embedding calls are parallelized with `asyncio.gather()`.",
        "3. Streaming improves perceived responsiveness by exposing TTFT separately from full-response time.",
        "",
        "## Quality Guardrail",
        "",
        "Run `python scripts/run_eval.py --mode real` and compare the result with the Day 29 baseline before claiming quality preservation.",
    ]
    (report_dir / "performance_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nReports written to reports/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.mode, max(1, min(args.queries, 20))))
