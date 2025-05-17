import re
from enum import Enum
from dataclasses import dataclass

class FResult(Enum):
    SAFE = 1  # validation returns okay
    FAILURE = 2  # validation contains error (something wrong with validation)
    ERROR = 3  # validation returns a potential error (look into)
    LLM_WEAKNESS = (
        4  # the generated input is ill-formed due to the weakness of the language model
    )
    TIMED_OUT = 10  # timed out, can be okay in certain targets

@dataclass
class SUTConfig:
    language: str
    path_documentation: str
    path_example_code: str
    trigger_to_generate_input: str
    input_hint: str
    path_hand_written_prompt: str | None
    SUT_string: str
    timeout: int
    folder: str
    batch_size: int
    temperature: float
    max_length: int
    device: str
    log_level: str
    template: str | None
    lambda_hyper: float
    beta1_hyper: float
    special_eos: str | None
    oracle_type: str


def comment_remover(text, lang="cpp"):
    if lang == "cpp" or lang == "go" or lang == "java":

        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE,
        )
        return re.sub(pattern, replacer, text)
    elif lang == "smt2":
        return re.sub(r";.*", "", text)
    else:
        # TODO (Add other lang support): temp, only facilitate basic c/cpp syntax
        # raise NotImplementedError("Only cpp supported for now")
        return text
