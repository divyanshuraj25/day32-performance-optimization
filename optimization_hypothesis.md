# Optimization Hypotheses

## 1. Model Selection
Use `gpt-4o-mini` for lightweight sub-tasks where full model capability is unnecessary.
Expected result: lower generation latency.
Guardrail: quality must remain at or above the Day 29 baseline.

## 2. Parallel Independent I/O
Use `asyncio.gather()` for independent document embedding calls.
Expected result: lower index-building latency than sequential requests.
Quality impact: none expected because the same documents/model are used.

## 3. Streaming
Stream the model response and measure TTFT separately from time-to-full-response.
Expected result: lower perceived waiting time.
Quality impact: none expected because only delivery changes.
