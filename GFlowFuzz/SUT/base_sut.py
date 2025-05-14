import glob
import time
from enum import Enum
from typing import Any, Dict
from rich.progress import track
from GFlowFuzz.logger import GlobberLogger, LEVEL
import traceback


class FResult(Enum):
    SAFE = 1  # validation returns okay
    FAILURE = 2  # validation contains error (something wrong with validation)
    ERROR = 3  # validation returns a potential error (look into)
    LLM_WEAKNESS = (
        4  # the generated input is ill-formed due to the weakness of the language model
    )
    TIMED_OUT = 10  # timed out, can be okay in certain targets


# base class file for target, used for user defined system targets
# the point is to separately define oracles/fuzzing specific functions/and usages
# target should be a stateful objects which has some notion of history (keeping a state of latest prompts)
class base_SUT(object):
    def __init__(self, language="c", timeout=10, folder="/", **kwargs):
        self.language = language
        self.folder = folder
        self.timeout = timeout
        self.CURRENT_TIME = time.time()
        # model based variables
        self.batch_size = kwargs["bs"]
        self.temperature = kwargs["temperature"]
        self.max_length = kwargs["max_length"]
        self.device = kwargs["device"]
        # loggers
        self.logger = GlobberLogger("sut.log", level=kwargs.get("level", LEVEL.INFO))
        self.logger.log("base_SUT initialized.", LEVEL.INFO)
        self.prompt_used = None
        

    @staticmethod
    def _create_prompt_from_config(config_dict: Dict[str, Any]) -> Dict:
        """Read the prompt ingredients via a config file."""
        documentation, example_code, hand_written_prompt = None, None, None

        # read the prompt ingredients from the config file
        target = config_dict["target"]
        path_documentation = target["path_documentation"]
        if path_documentation is not None:
            documentation = open(path_documentation, "r").read()
        path_example_code = target["path_example_code"]
        if path_example_code is not None:
            example_code = open(path_example_code, "r").read()
        trigger_to_generate_input = target["trigger_to_generate_input"]
        input_hint = target["input_hint"]
        path_hand_written_prompt = target["path_hand_written_prompt"]
        if path_hand_written_prompt is not None:
            hand_written_prompt = open(path_hand_written_prompt, "r").read()
        target_string = target["target_string"]
        dict_compat = {
            "docstring": documentation,
            "example_code": example_code,
            "separator": trigger_to_generate_input,
            "begin": input_hint,
            "hw_prompt": hand_written_prompt,
            "target_api": target_string,
        }
        return dict_compat

    def write_back_file(self, code: str):
        raise NotImplementedError

    # each target defines their way of validating prompts (can overwrite)
    def validate_prompt(self, prompt: str):
        self.logger.log(f"validate_prompt called with prompt: {str(prompt)[:200]}", LEVEL.TRACE)
        start_time = time.time()
        try:
            fos = self.coder.generate(
                prompt,
                batch_size=self.batch_size,
                temperature=self.temperature,
                max_length=self.max_length,
            )
            self.logger.log(f"Generated {len(fos)} code samples for validation. First sample: {str(fos[0])[:200] if fos else 'None'}", LEVEL.VERBOSE)
            unique_set = set()
            score = 0
            for fo in fos:
                code = self.prompt_used["begin"] + "\n" + fo
                self.logger.log(f"Validating code: {str(code)[:200]}", LEVEL.TRACE)
                wb_file = self.write_back_file(code)
                self.logger.log(f"Wrote code to file: {wb_file}", LEVEL.TRACE)
                result, _ = self.validate_individual(wb_file)
                self.logger.log(f"Validation result: {result}", LEVEL.TRACE)
                if (
                    result == FResult.SAFE
                    and self.filter(code)
                    and self.clean_code(code) not in unique_set
                ):
                    unique_set.add(self.clean_code(code))
                    score += 1
            end_time = time.time()
            self.logger.log(f"validate_prompt completed in {end_time - start_time:.2f}s, score: {score}", LEVEL.TRACE)
            return score
        except Exception as e:
            self.logger.log(f"Error during validate_prompt: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise

    # helper for updating
    def filter(self, code: str) -> bool:
        raise NotImplementedError

    # difference between clean and clean_code (honestly just backwards compatibility)
    # but the point is that clean should be applied as soon as generation whereas clean code is used
    # more so for filtering
    def clean(self, code: str) -> str:
        raise NotImplementedError

    def clean_code(self, code: str) -> str:
        raise NotImplementedError

    # validation
    def validate_individual(self, filename) -> tuple[FResult, str]:
        raise NotImplementedError

    def parse_validation_message(self, f_result, message, file_name):
        self.logger.log(f"Validating {file_name} ...", LEVEL.TRACE)
        self.logger.log(f"Validation message: {str(message)[:200]}", LEVEL.VERBOSE)
        if f_result == FResult.SAFE:
            self.logger.log(f"{file_name} is safe", LEVEL.VERBOSE)
        elif f_result == FResult.FAILURE:
            self.logger.log(f"{file_name} failed validation with error message: {message}", LEVEL.VERBOSE)
        elif f_result == FResult.ERROR:
            self.logger.log(f"{file_name} has potential error!\nerror message:\n{message}", LEVEL.VERBOSE)
            self.logger.log(f"{file_name} has potential error!", LEVEL.INFO)
        elif f_result == FResult.TIMED_OUT:
            self.logger.log(f"{file_name} timed out", LEVEL.VERBOSE)

    def validate_all(self):
        self.logger.log(f"validate_all called for folder: {self.folder}", LEVEL.TRACE)
        start_time = time.time()
        try:
            for fuzz_output in track(
                glob.glob(self.folder + "/*.fuzz"),
                description="Validating",
            ):
                self.logger.log(f"Validating fuzz output: {fuzz_output}", LEVEL.TRACE)
                f_result, message = self.validate_individual(fuzz_output)
                self.parse_validation_message(f_result, message, fuzz_output)
            end_time = time.time()
            self.logger.log(f"validate_all completed in {end_time - start_time:.2f}s", LEVEL.TRACE)
        except Exception as e:
            self.logger.log(f"Error during validate_all: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise
