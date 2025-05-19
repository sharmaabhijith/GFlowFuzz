import os
import time
import logging
import requests
from openai import OpenAI
from dataclasses import asdict
from concurrent.futures import TimeoutError as FuturesTimeout
from requests.exceptions import RequestException
from logger import GlobberLogger, LEVEL
from .utils import LLMConfig, LLMResponse


class LLMClient:
    """
    Factory-based LLM client for OpenAI and DeepInfra with retries, timeouts,
    and version-compatible API handling.
    """
    def __init__(
            self, 
            api_name: str = "openai", 
            engine_name: str = "gpt-3.5-turbo", 
            timeout: int = 120
        ):
        self.timeout = timeout
        self.engine_name = engine_name 
        self.logger = GlobberLogger("llmclient.log", level=LEVEL.INFO)

        if api_name.lower() == "deepinfra":
            api_key = os.getenv("DEEPINFRA_API_KEY")
            base_url = "https://api.deepinfra.com/v1/openai"
        elif api_name.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
        else:
            raise ValueError(f"Unsupported api_name: {self.api_name}")
        
        if not api_key:
            raise ValueError(f"{api_name} API key must be provided.")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)


    def request(self, config: LLMConfig):
        """
        Makes a request with retries and timeout.
        """
        config_dict = asdict(config)
        config_dict.pop("engine_name")
        retries = 5
        for attempt in range(retries):
        
            try:
                config_dict["model"] = self.engine_name
                data = self.client.chat.completions.create(**config_dict)
                content = data.choices[0].message.content
                return LLMResponse(content=content, raw=data)
            
            except (FuturesTimeout, requests.Timeout):
                logging.warning(f"Timeout on atempt {attempt + 1}/{retries}")
            except requests.HTTPError as e:
                logging.error(f"HTTP error: {e}")
            except RequestException as e:
                logging.error(f"Request failed: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                time.sleep(2 ** attempt)

        raise RuntimeError("Failed to get a valid response after retries.")
