import glob
import os
import random
import time
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import torch
from rich.progress import track

from GFlowFuzz.coder_LM import build_coder_LM
from GFlowFuzz.util.api_request import create_config, request_engine
from GFlowFuzz.utils import LEVEL, Logger
from GFlowFuzz.instruct_LM import build_instruct_LM_trainer


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
        self.coder_name = kwargs["coder_name"]
        self.coder = None
        # loggers
        self.g_logger = Logger(self.folder, "log_generation.txt", level=kwargs["level"])
        self.v_logger = Logger(self.folder, "log_validation.txt", level=kwargs["level"])
        # main logger for system messages
        self.m_logger = Logger(self.folder, "log.txt")
        # system messages for prompting
        self.SYSTEM_MESSAGE = None
        self.AP_SYSTEM_MESSAGE = "You are an auto-prompting tool"
        self.AP_INSTRUCTION = (
            "Please summarize the above documentation in a concise manner to describe the usage and "
            "functionality of the target "
        )
        # prompt based variables
        self.hw = kwargs["use_hw"]
        self.no_input_prompt = kwargs["no_input_prompt"]
        self.prompt_used = None
        self.prompt = None
        self.initial_prompt = None
        self.prev_example = None
        # prompt strategies
        self.se_prompt = self.wrap_in_comment(
            "Please create a semantically equivalent program to the previous "
            "generation"
        )
        self.m_prompt = self.wrap_in_comment(
            "Please create a mutated program that modifies the previous generation"
        )
        self.c_prompt = self.wrap_in_comment(
            "Please combine the two previous programs into a single program"
        )
        self.p_strategy = kwargs["prompt_strategy"]
        # eos based
        self.special_eos = None
        if "coder_name" in kwargs:
            self.coder_name = kwargs["coder_name"]
        if "target_name" in kwargs:
            self.target_name = kwargs["target_name"]

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
        fos = self.coder.generate(
            prompt,
            batch_size=self.batch_size,
            temperature=self.temperature,
            max_length=self.max_length,
        )
        unique_set = set()
        score = 0
        for fo in fos:
            code = self.prompt_used["begin"] + "\n" + fo
            wb_file = self.write_back_file(code)
            result, _ = self.validate_individual(wb_file)
            if (
                result == FResult.SAFE
                and self.filter(code)
                and self.clean_code(code) not in unique_set
            ):
                unique_set.add(self.clean_code(code))
                score += 1
        return score
    

    # initialize through either some templates or auto-prompting to determine prompts
    def initialize(self):
        self.m_logger.logo(
            "Initializing ... this may take a while ...", level=LEVEL.INFO
        )
        self.m_logger.logo("Loading model ...", level=LEVEL.INFO)
        eos = [
            self.prompt_used["separator"],
            "<eom>",  # for codegen2
            self.se_prompt,
            self.m_prompt,
            self.c_prompt,
        ]
        # if the config_dict is an attribute, add additional eos from config_dict
        # which might be model specific
        if hasattr(self, "config_dict"):
            coder = self.config_dict["coder"]
            coder_name = coder["coder_name"]
            additional_eos = coder.get("additional_eos", [])
            if additional_eos:
                eos = eos + additional_eos
        else:
            coder_name = self.coder_name

        if self.special_eos is not None:
            eos = eos + [self.special_eos]

        self.coder = build_coder_LM(
            eos=eos,
            coder_name=coder_name,
            device=self.device,
            max_length=self.max_length,
        )
        self.m_logger.logo("Model Loaded", level=LEVEL.INFO)
        self.initial_prompt = self.auto_prompt(
            message=self.prompt_used["docstring"],
            hw_prompt=self.prompt_used["hw_prompt"] if self.hw else None,
            hw=self.hw,
            no_input_prompt=self.no_input_prompt,
        )
        self.prompt = self.initial_prompt
        self.m_logger.logo("Done", level=LEVEL.INFO)
    
    def generate_instructions(self, temperature=0.8, num_instructions=3):
        """
        Generate additional instructions based on the current prompt
        
        Args:
            temperature: Temperature to use for instruction generation
            num_instructions: Number of instructions to generate
            
        Returns:
            Updated prompt with additional instructions
        """
        self.m_logger.logo("Generating additional instructions with instruct_LM...", level=LEVEL.INFO)
        
        self.instruction_trainer = build_instruct_LM_trainer(self.instruct_args)
        
        # Generate additional instructions based on the prompt
        additional_instructions = self.instruction_trainer.generate_instructions(
            prompt=self.prompt,
            num_instructions=num_instructions,
            temperature=temperature
        )
        
        # Format and append the additional instructions
        formatted_instructions = self.instruct_args.instruction_separator.join(additional_instructions)
        updated_prompt = self.prompt + f"\n\n{formatted_instructions}"
        
        # Log the instructions for debugging
        self.g_logger.logo(f"Generated additional instructions:\n{formatted_instructions}", level=LEVEL.VERBOSE)
        self.m_logger.logo("Done adding instructions with instruct_LM.", level=LEVEL.INFO)
        
        return updated_prompt

    def generate_code(self) -> List[str]:
        self.g_logger.logo(self.prompt, level=LEVEL.VERBOSE)
        
        return self.coder.generate(
            self.prompt,
            batch_size=self.batch_size,
            temperature=self.temperature,
            max_length=1024,
        )

    # generation
    def generate(self, **kwargs) -> Union[List[str], bool]:
        # Generate instructions and update the prompt before code generation
        self.prompt = self.generate_instructions()
        try:
            fos = self.generate_code()
        except RuntimeError:
            # catch cuda out of memory error.
            self.m_logger.logo("cuda out of memory...", level=LEVEL.INFO)
            del self.coder
            torch.cuda.empty_cache()
            return False
        new_fos = []
        for fo in fos:
            self.g_logger.logo("========== sample =========", level=LEVEL.VERBOSE)
            new_fos.append(self.clean(self.prompt_used["begin"] + "\n" + fo))
            self.g_logger.logo(
                self.clean(self.prompt_used["begin"] + "\n" + fo), level=LEVEL.VERBOSE
            )
            self.g_logger.logo("========== sample =========", level=LEVEL.VERBOSE)
        return new_fos

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
    def validate_individual(self, filename) -> (FResult, str):
        raise NotImplementedError

    def parse_validation_message(self, f_result, message, file_name):
        # TODO: rewrite to include only status in TRACE but full message in VERBOSE
        self.v_logger.logo("Validating {} ...".format(file_name), LEVEL.TRACE)
        if f_result == FResult.SAFE:
            self.v_logger.logo("{} is safe".format(file_name), LEVEL.VERBOSE)
        elif f_result == FResult.FAILURE:
            self.v_logger.logo(
                "{} failed validation with error message: {}".format(
                    file_name, message, LEVEL.VERBOSE
                )
            )
        elif f_result == FResult.ERROR:
            self.v_logger.logo(
                "{} has potential error!\nerror message:\n{}".format(
                    file_name, message, LEVEL.VERBOSE
                )
            )
            self.m_logger.logo(
                "{} has potential error!".format(file_name, message, LEVEL.INFO)
            )
        elif f_result == FResult.TIMED_OUT:
            self.v_logger.logo("{} timed out".format(file_name), LEVEL.VERBOSE)

    def validate_all(self):
        for fuzz_output in track(
            glob.glob(self.folder + "/*.fuzz"),
            description="Validating",
        ):
            f_result, message = self.validate_individual(fuzz_output)
            self.parse_validation_message(f_result, message, fuzz_output)
