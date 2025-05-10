from typing import List
from .base_coder import BaseCoder
from .utils import CoderConfig

class DeepSeekCoder(BaseCoder):
    def __init__(self, coder_config: CoderConfig):
        super().__init__(coder_config)

    def format_prompt(self, prompt: str) -> str:
        # DeepSeek doesn’t need special tokens; return as-is
        return prompt
