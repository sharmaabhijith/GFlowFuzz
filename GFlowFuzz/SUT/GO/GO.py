import subprocess
import time
from typing import List, Union

import torch

from GFlowFuzz.SUT.base_sut import FResult, base_SUT
from GFlowFuzz.SUT.utils import SUTConfig
from GFlowFuzz.utils import LEVEL, comment_remover
from GFlowFuzz.oracle.coverage import CoverageManager, Tool
import pathlib


class GO_SUT(base_SUT):
    def __init__(self, sut_config: SUTConfig):
        super().__init__(sut_config)
        self.prompt_used = self._create_prompt_from_config(sut_config)
        self.special_eos = sut_config.special_eos if sut_config.special_eos is not None else "package main"
        self.coverage_manager = CoverageManager(Tool.GO, pathlib.Path(f"/tmp/out{self.CURRENT_TIME}"))
        self.prev_coverage = 0
        self.lambda_ = sut_config.lambda_hyper
        self.beta1_ = sut_config.beta1_hyper

    def wrap_prompt(self, prompt: str) -> str:
        return (
            f"// {prompt}\n{self.prompt_used['separator']}\n{self.prompt_used['begin']}"
        )

    def wrap_in_comment(self, prompt: str) -> str:
        return f"// {prompt}"

    def filter(self, code: str) -> bool:
        code = code.replace(self.prompt_used["begin"], "").strip()
        code = comment_remover(code)
        if self.prompt_used["target_api"] not in code:
            return False
        return True

    def clean(self, code: str) -> str:
        code = comment_remover(code)
        return code

    def clean_code(self, code: str) -> str:
        code = code.replace(self.prompt_used["begin"], "").strip()
        code = comment_remover(code)
        code = "\n".join([line for line in code.split("\n") if line.strip() != ""])
        return code

    def write_back_file(self, code):
        try:
            with open(
                "/tmp/temp{}.go".format(self.CURRENT_TIME), "w", encoding="utf-8"
            ) as f:
                f.write(code)
        except:
            pass
        return "/tmp/temp{}.go".format(self.CURRENT_TIME)

    def validate_individual(self, filename) -> (FResult, str, float):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                code = f.read()
        except:
            pass
        self.write_back_file(code)
        try:
            exit_code = subprocess.run(
                f"{self.target_name} build -o /tmp/temp{self.CURRENT_TIME} /tmp/temp{self.CURRENT_TIME}.go",
                shell=True,
                capture_output=True,
                encoding="utf-8",
                timeout=5,
                text=True,
            )
        except subprocess.TimeoutExpired as te:
            pname = f"'temp{self.CURRENT_TIME}'"
            subprocess.run(
                ["ps -ef | grep " + pname + " | grep -v grep | awk '{print $2}'"],
                shell=True,
            )
            subprocess.run(
                [
                    "ps -ef | grep "
                    + pname
                    + " | grep -v grep | awk '{print $2}' | xargs -r kill -9"
                ],
                shell=True,
            )  # kill all tests thank you
            return FResult.TIMED_OUT, "go", 0.0
        except UnicodeDecodeError as ue:
            return FResult.FAILURE, "decoding error", 0.0
        if exit_code.returncode == 1:
            fresult = FResult.FAILURE
            msg = exit_code.stderr
        elif exit_code.returncode == 0:
            fresult = FResult.SAFE
            msg = exit_code.stdout
        else:
            fresult = FResult.ERROR
            msg = exit_code.stderr

        self.coverage_manager.run_once()
        new_cov = self.coverage_manager.update_total()
        coverage_diff = new_cov - self.prev_coverage
        bug = 1 if fresult in (FResult.FAILURE, FResult.ERROR) else 0
        reward = coverage_diff + self.lambda_ * new_cov + self.beta1_ * bug
        self.prev_coverage = new_cov

        if fresult == FResult.SAFE:
            return FResult.SAFE, f"its safe\nCoverage: {new_cov}", reward
        elif fresult == FResult.ERROR:
            return FResult.ERROR, f"{msg}\nCoverage: {new_cov}", reward
        elif fresult == FResult.TIMED_OUT:
            return FResult.ERROR, f"timed out\nCoverage: {new_cov}", reward
        elif fresult == FResult.FAILURE:
            return FResult.FAILURE, f"{msg}\nCoverage: {new_cov}", reward
        else:
            return (FResult.TIMED_OUT, f"Coverage: {new_cov}", reward)
