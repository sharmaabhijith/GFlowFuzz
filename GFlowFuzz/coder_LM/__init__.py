from typing import List
from abc import ABC, abstractmethod
from .starcoder import StarCoder
from .deepseek import DeepSeekCoder
from .utils import CoderConfig
from .base_coder import BaseCoderLocal, BaseCoderAPI

class BaseCoder(ABC):
    @abstractmethod
    def generate_code(self, prompt: str, **kwargs) -> List[str]:
        pass

def Coder(coder_config: CoderConfig, api_driven: bool = True) -> BaseCoder:
    """Returns a coder coder instance (optional: using the configuration file)."""
    if api_driven:
        return BaseCoderAPI(coder_config)
    else:
        # print the coder config
        print("=== coder Config ===")
        print(f"coder_name: {coder_config.coder_name}")
        for k, v in coder_config.__dict__.items():
            print(f"{k}: {v}")
        coder_class = (
            DeepSeekCoder if "deepseek" in coder_config.coder_name.lower()
            else StarCoder
        )
        coder = coder_class(coder_config)
        coder_class_name = coder.__class__.__name__
        print(f"coder_obj (class name): {coder_class_name}")
        print("====================")
        return coder
