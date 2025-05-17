import signal
import time
import openai
from dataclasses import dataclass
from typing import Optional
from GFlowFuzz.client_LLM import LLMConfig


@dataclass
class DistillerConfig:
    folder: str
    llm_config: Optional[LLMConfig] = None
    system_message: str = "You are an auto-prompting tool"
    instruction: str = "Please summarize the above documentation in a concise manner to describe the usage and functionality of the target"

