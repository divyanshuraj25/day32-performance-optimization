from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    use_small_model: bool = True
    stream: bool = False

class AnswerResponse(BaseModel):
    answer: str
    model: str
    timings_ms: dict[str, float]
    total_ms: float
    sources: list[str]

class ProfileResponse(BaseModel):
    query: str
    timings_ms: dict[str, float]
    total_ms: float
    percentage_breakdown: dict[str, float]
