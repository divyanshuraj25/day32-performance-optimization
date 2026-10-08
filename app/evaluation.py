from __future__ import annotations
from dataclasses import dataclass
from .pipeline import PerformancePipeline

EVAL_CASES = [
    ("What is async I/O?", "overlap independent network operations"),
    ("Why use FAISS?", "similarity search over dense vectors"),
    ("What is TTFT?", "time before the first streamed token"),
    ("Why use a smaller model?", "reduce latency when full capability is not needed"),
    ("Why cache embeddings?", "avoid repeated embedding work"),
    ("Why stream responses?", "show output before the full response finishes"),
    ("What should profiling measure?", "each stage separately"),
    ("What makes an optimization hypothesis testable?", "expected latency and quality impact"),
]

@dataclass
class EvalResult:
    query: str
    answer: str
    score: float

def keyword_score(answer: str, expected: str) -> float:
    words = [w.lower() for w in expected.split() if len(w) > 2]
    hits = sum(1 for word in words if word in answer.lower())
    return hits / max(1, len(words))

async def evaluate(pipeline: PerformancePipeline) -> list[EvalResult]:
    results = []
    for query, expected in EVAL_CASES:
        output = await pipeline.run(query, use_small_model=True)
        results.append(EvalResult(query, output.answer, keyword_score(output.answer, expected)))
    return results
