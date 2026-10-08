# Day 32 — AI Pipeline Performance Optimization

Production-style FastAPI project for profiling and optimizing an AI/RAG pipeline.

## What is implemented

- `time.perf_counter()` around every major pipeline step
- 20-query benchmark runner with average latency and percentage breakdown
- FAISS retrieval
- Async parallel document embedding with `asyncio.gather()`
- Model selection using `gpt-4o-mini` for lightweight requests
- End-to-end streaming
- TTFT (time-to-first-token) measurement
- Time-to-full-response measurement
- Day 29 quality-evaluation guardrail
- Before/after comparison templates
- Mock mode so the project can be tested without API credits

## Pipeline

`query → embedding → FAISS search → prompt construction → LLM generation → response formatting`

## Windows PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your OpenAI key in `.env`, then:

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Benchmark

Offline/demo:

```powershell
python scripts/run_benchmark.py --mode mock --queries 20
```

Real OpenAI measurement:

```powershell
python scripts/run_benchmark.py --mode real --queries 20
```

## Evaluation

```powershell
python scripts/run_eval.py --mode mock
python scripts/run_eval.py --mode real
```

## Parallelism test

```powershell
python scripts/compare_parallelism.py
```

## API

- `GET /health`
- `POST /v1/answer`
- `POST /v1/profile`
- `POST /v1/answer/stream`

Example request:

```json
{
  "query": "How does async I/O reduce latency?",
  "use_small_model": true,
  "stream": false
}
```

## Important submission note

The repository intentionally does **not** invent performance or quality numbers. Run the real benchmark and evaluation with your API key before submitting measured claims. The generated reports will contain your actual environment's measurements.
