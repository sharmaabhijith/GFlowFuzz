from typing import List
from .base_coder import BaseCoder
from .utils import CoderConfig

class StarCoder(BaseCoder):
    def __init__(self, coder_config: CoderConfig):
        super().__init__(coder_config)
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"

    def format_prompt(self, prompt: str) -> str:
        return f"{self.prefix_token}{prompt}{self.suffix_token}"