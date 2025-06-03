import glob
import time
from typing import Any, Dict
from rich.progress import track
from logger import GlobberLogger, LEVEL
import traceback
from SUT.utils import FResult, SUTConfig
from coder_LM import BaseCoder

# base class file for target, used for user defined system targets
# the point is to separately define oracles/fuzzing specific functions/and usages
# target should be a stateful objects which has some notion of history (keeping a state of latest prompts)
class BaseSUT(object):
    def __init__(self, sut_config: SUTConfig, target_name: str):
        self.target_name = target_name
        self.sut_config = sut_config
        self.language = sut_config.language
        self.folder = sut_config.folder
        self.timeout = sut_config.timeout
        self.CURRENT_TIME = time.time()
        # model based variables
        self.batch_size = sut_config.batch_size
        self.temperature = sut_config.temperature
        self.max_length = sut_config.max_length
        # loggers
        self.logger = GlobberLogger("fuzzer.log", level=LEVEL.TRACE)
        self.logger.log("BaseSUT initialized with SUTConfig.", LEVEL.INFO)
        self.prompt_used = None

    @staticmethod
    def _create_prompt_from_config(sut_config: SUTConfig) -> Dict:
        """Read the prompt ingredients via a SUTConfig object."""
        documentation, example_code, hand_written_prompt = None, None, None

        if sut_config.path_documentation:
            try:
                documentation = open(sut_config.path_documentation, "r").read()
            except FileNotFoundError:
                print(f"Warning: Documentation file not found: {sut_config.path_documentation}")
        if sut_config.path_example_code:
            try:
                example_code = open(sut_config.path_example_code, "r").read()
            except FileNotFoundError:
                print(f"Warning: Example code file not found: {sut_config.path_example_code}")
        
        trigger_to_generate_input = sut_config.trigger_to_generate_input
        input_hint = sut_config.input_hint
        
        if sut_config.path_hand_written_prompt:
            try:
                hand_written_prompt = open(sut_config.path_hand_written_prompt, "r").read()
            except FileNotFoundError:
                print(f"Warning: Hand-written prompt file not found: {sut_config.path_hand_written_prompt}")
        
        target_string = sut_config.SUT_string
        
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
    
    def wrap_prompt(self, prompt: str) -> str:
        raise NotImplementedError

    def wrap_in_comment(self, prompt: str) -> str:
        raise NotImplementedError

    # each target defines their way of validating prompts (can overwrite)
    def validate_prompt(self, prompt: str, coder: BaseCoder):
        self.logger.log(f"validate_prompt called with prompt: {str(prompt)[:200]}", LEVEL.TRACE)
        start_time = time.time()
        try:
            fos = coder.generate_code(prompt)
            self.logger.log(f"Generated {len(fos)} code samples for validation. First sample: {str(fos[0])[:200] if fos else 'None'}", LEVEL.VERBOSE)
            unique_set = set()
            score = 0
            for fo in fos:
                code = self.prompt_used["begin"] + "\n" + fo
                self.logger.log(f"Validating code: {str(code)[:200]}", LEVEL.TRACE)
                wb_file = self.write_back_file(code)
                self.logger.log(f"Wrote code to file: {wb_file}", LEVEL.TRACE)
                result, _, _ = self.validate_individual(wb_file)
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
    def validate_individual(self, filename) -> tuple[FResult, str, int, float]:
        raise NotImplementedError

    def parse_validation_message(self, f_result, message, file_name):
        self.logger.log(f"Validating {file_name} ...", LEVEL.TRACE)
        self.logger.log(f"Validation message: {str(message)[:200]}", LEVEL.TRACE)
        if f_result == FResult.SAFE:
            self.logger.log(f"{file_name} is safe", LEVEL.TRACE)
        elif f_result == FResult.FAILURE:
            self.logger.log(f"{file_name} failed validation with error message: {message}", LEVEL.TRACE)
        elif f_result == FResult.ERROR:
            self.logger.log(f"{file_name} has potential error!\nerror message:\n{message}", LEVEL.TRACE)
            self.logger.log(f"{file_name} has potential error!", LEVEL.TRACE)
        elif f_result == FResult.TIMED_OUT:
            self.logger.log(f"{file_name} timed out", LEVEL.TRACE)

    def validate_all(self):
        self.logger.log(f"validate_all called for folder: {self.folder}", LEVEL.TRACE)
        start_time = time.time()
        try:
            for fuzz_output in track(
                glob.glob(self.folder + "/*.c"),
                description="Validating",
            ):
                self.logger.log(f"Validating fuzz output: {fuzz_output}", LEVEL.TRACE)
                f_result, message, _ = self.validate_individual(fuzz_output)
                self.parse_validation_message(f_result, message, fuzz_output)
            end_time = time.time()
            self.logger.log(f"validate_all completed in {end_time - start_time:.2f}s", LEVEL.TRACE)
        except Exception as e:
            self.logger.log(f"Error during validate_all: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise
