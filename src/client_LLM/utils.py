from dataclasses import dataclass
from typing import List

@dataclass
class LLMResponse:
    content: str
    raw: dict

@dataclass
class LLMConfig:
    messages: List
    max_tokens: int
    temperature: float
    engine_name: str
