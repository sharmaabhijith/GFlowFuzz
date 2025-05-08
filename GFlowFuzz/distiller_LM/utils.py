import os
import signal
import time
import random

import openai

from GFlowFuzz.SUT.base_sut import FResult

openai.api_key = os.environ.get("OPENAI_API_KEY", "dummy")
client = openai.OpenAI()


def create_openai_config(
    prompt,
    engine_name="code-davinci-002",
    stop=None,
    max_tokens=200,
    top_p=1,
    n=1,
    temperature=0,
):
    return {
        "engine": engine_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "temperature": temperature,
        "logprobs": 1,
        "n": n,
        "stop": stop,
    }


def create_config(
    prev: dict,
    messages: list,
    max_tokens: int,
    temperature: float = 2,
    model: str = "gpt-3.5-turbo",
):
    if prev == {}:
        return {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
    else:
        return prev


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

def update_strategy(self, new_code: str) -> str:
        while 1:
            strategy = random.randint(0, self.p_strategy)
            # generate new code using separator
            if strategy == 0:
                return f"\n{new_code}\n{self.prompt_used['separator']}\n"
            # mutate existing code
            elif strategy == 1:
                return f"\n{new_code}\n{self.m_prompt}\n"
            # semantically equivalent code generation
            elif strategy == 2:
                return f"\n{new_code}\n{self.se_prompt}\n"
            # combine previous two code generations
            else:
                if self.prev_example is not None:
                    return f"\n{self.prev_example}\n{self.prompt_used['separator']}\n{self.prompt_used['begin']}\n{new_code}\n{self.c_prompt}\n"

# update
def update(self, **kwargs):
    new_code = ""
    for result, code in kwargs["prev"]:
        if (
            result == FResult.SAFE
            and self.filter(code)
            and self.clean_code(code) != self.prev_example
        ):
            new_code = self.clean_code(code)
    if new_code != "" and self.p_strategy != -1:
        self.prompt = (
            self.initial_prompt
            + self.update_strategy(new_code)
            + self.prompt_used["begin"]
            + "\n"
        )
        self.prev_example = new_code
