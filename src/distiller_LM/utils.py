from dataclasses import dataclass
from client_LLM import LLMConfig


@dataclass
class DistillerConfig:
    api_name: str
    llm_config: LLMConfig
    system_message: str = "You are an auto-prompting tool"
    instruction: str = "Please summarize the above documentation in a concise manner to describe the usage and functionality of the target"

