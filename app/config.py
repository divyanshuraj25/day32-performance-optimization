from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    fast_model: str = os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")
    quality_model: str = os.getenv("OPENAI_QUALITY_MODEL", "gpt-4o")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

settings = Settings()
