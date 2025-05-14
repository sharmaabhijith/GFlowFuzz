import os
import signal
import time
import random
import openai
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from GFlowFuzz.logger import GlobberLogger

openai.api_key = os.environ.get("OPENAI_API_KEY", "dummy")
client = openai.OpenAI()

@dataclass
class OpenAIConfig:
    prev: Dict
    messages: List
    max_tokens: int
    temperature: float = 2
    engine_name: str = "gpt-3.5-turbo"
    stop: Optional[str] = None
    top_p: int = 1

@dataclass
class DistillerConfig:
    folder: str
    logger: GlobberLogger
    wrap_prompt_func: Callable[[str], str]
    validate_prompt_func: Callable[[str], float]
    prompt_components: Dict[str, str] = field(default_factory=dict)
    openai_config: OpenAIConfig = None
    system_message: str = "You are an auto-prompting tool"
    instruction: str = "Please summarize the above documentation in a concise manner to describe the usage and functionality of the target"


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("I have become end of time")


# Handles requests to OpenAI API
def request_engine(config):
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(120)  # wait 10
            ret = client.chat.completions.create(**config)
            signal.alarm(0)
        except openai._exceptions.BadRequestError as e:
            print(e)
            signal.alarm(0)
        except openai._exceptions.RateLimitError as e:
            print("Rate limit exceeded. Waiting...")
            print(e)
            signal.alarm(0)  # cancel alarm
            time.sleep(5)
        except openai._exceptions.APIConnectionError as e:
            print("API connection error. Waiting...")
            signal.alarm(0)  # cancel alarm
            time.sleep(5)
        except Exception as e:
            print(e)
            print("Unknown error. Waiting...")
            signal.alarm(0)  # cancel alarm
            time.sleep(1)
    return ret
