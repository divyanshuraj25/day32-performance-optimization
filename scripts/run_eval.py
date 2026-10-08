from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.evaluation import evaluate
from app.pipeline import PerformancePipeline

async def main(mode: str):
    client = None
    if mode == "real":
        if not settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.api_key)

    results = await evaluate(PerformancePipeline(client))
    payload = {
        "mode": mode,
        "average_score": sum(r.score for r in results) / len(results),
        "cases": [{"query": r.query, "answer": r.answer, "score": r.score} for r in results],
    }
    path = ROOT / "reports" / "evaluation_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Average quality score: {payload['average_score']:.3f}")
    print(f"Saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    args = parser.parse_args()
    asyncio.run(main(args.mode))
