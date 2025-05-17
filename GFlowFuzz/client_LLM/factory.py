import os
import time
import logging
import requests
import openai
from dataclasses import asdict
from concurrent.futures import TimeoutError as FuturesTimeout
from requests.exceptions import RequestException
from GFlowFuzz.logger import GlobberLogger, LEVEL
from .utils import LLMConfig, LLMResponse


class FactoryLLMClient:
    """
    Factory-based LLM client for OpenAI and DeepInfra with retries, timeouts,
    and version-compatible API handling.
    """
    def __init__(
            self, 
            provider: str = "openai", 
            model: str = "gpt-3.5-turbo", 
            timeout: int = 120
        ):
        self.provider = provider.lower()
        self.timeout = timeout
        self.model = model or ("gpt-3.5-turbo" if self.provider == "openai" else "default-model")
        self.logger = GlobberLogger("llmclient.log", level=LEVEL.INFO)

        if self.provider == "deepinfra":
            self.api_key = os.getenv("DEEPINFRA_API_KEY")
            if not self.api_key:
                raise ValueError("DEEPINFRA_API_KEY must be provided as env variable.")
            self.endpoint = "https://api.deepinfra.com/v1/chat/completions"
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided.")
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    def request(self, config: LLMConfig):
        """
        Makes a request with retries and timeout.
        """
        config_dict = asdict(config)
        retries = 5
        for attempt in range(retries):
            try:
                if "model" not in config_dict:
                    config_dict["model"] = self.model
                if self.provider == "deepinfra":
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    response = requests.post(
                        self.endpoint, 
                        headers=headers, 
                        json=config_dict, 
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                elif self.provider == "openai":
                    data = self.client.chat.completions.create(**config_dict)
                    content = data.choices[0].message.content
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                return LLMResponse(content=content, raw=data)
            
            except (FuturesTimeout, requests.Timeout):
                logging.warning(f"Timeout on attempt {attempt + 1}/{retries}")
            except requests.HTTPError as e:
                logging.error(f"HTTP error: {e}")
            except RequestException as e:
                logging.error(f"Request failed: {e}")
            except openai.RateLimitError as e:
                logging.warning(f"Rate limit hit: {e}")
                time.sleep(5)
            except openai.APIConnectionError as e:
                logging.warning(f"API connection error: {e}")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                time.sleep(2 ** attempt)

        raise RuntimeError("Failed to get a valid response after retries.")
