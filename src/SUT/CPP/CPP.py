import re
import subprocess
import time
from typing import List, Union

import torch

from SUT.base_sut import FResult, BaseSUT
from SUT.utils import SUTConfig, comment_remover
from oracle.coverage import CoverageManager, Tool
import pathlib

main_code = """
int main(){
return 0;
}
"""


class CPP_SUT(BaseSUT):
    def __init__(self, sut_config: SUTConfig):
        super().__init__(sut_config)
        self.SYSTEM_MESSAGE = "You are a C++ Fuzzer"
        self.prompt_used = self._create_prompt_from_config(sut_config)
        self.coverage_manager = CoverageManager(Tool.GPP, pathlib.Path(f"/tmp/out{self.CURRENT_TIME}"))
        self.prev_coverage = 0
        self.lambda_ = sut_config.lambda_hyper
        self.beta1_ = sut_config.beta1_hyper

    def write_back_file(self, code):
        try:
            with open(
                "/tmp/temp{}.cpp".format(self.CURRENT_TIME), "w", encoding="utf-8"
            ) as f:
                f.write(code)
        except:
            pass
        return "/tmp/temp{}.cpp".format(self.CURRENT_TIME)

    def wrap_prompt(self, prompt: str) -> str:
        return f"/* {prompt} */\n{self.prompt_used['separator']}\n{self.prompt_used['begin']}"

    def wrap_in_comment(self, prompt: str) -> str:
        return f"/* {prompt} */"

    def filter(self, code) -> bool:
        clean_code = code.replace(self.prompt_used["begin"], "").strip()
        if self.prompt_used["target_api"] not in clean_code:
            return False
        return True

    def clean(self, code: str) -> str:
        code = comment_remover(code)
        return code

    # remove any comments, or blank lines
    def clean_code(self, code: str) -> str:
        code = comment_remover(code)
        code = "\n".join(
            [
                line
                for line in code.split("\n")
                if line.strip() != "" and line.strip() != self.prompt_used["begin"]
            ]
        )
        return code

    def validate_compiler(self, compiler, filename) -> (FResult, str):
        # check without -c option (+ linking)
        try:
            exit_code = subprocess.run(
                f"{compiler} -x c++ -std=c++23 {filename} -o /tmp/out{self.CURRENT_TIME}",
                shell=True,
                capture_output=True,
                encoding="utf-8",
                timeout=5,
                text=True,
            )
        except subprocess.TimeoutExpired as te:
            pname = f"'{filename}'"
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
            return FResult.TIMED_OUT, compiler

        if exit_code.returncode == 1:
            if "undefined reference to `main'" in exit_code.stderr:
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        code = f.read()
                except:
                    pass
                self.write_back_file(code + main_code)
                exit_code = subprocess.run(
                    f"{compiler} -std=c++23 -x c++ /tmp/temp{self.CURRENT_TIME}.cpp -o /tmp/out{self.CURRENT_TIME}",
                    shell=True,
                    capture_output=True,
                    encoding="utf-8",
                    text=True,
                )
                if exit_code.returncode == 0:
                    return FResult.SAFE, "its safe"
            return FResult.FAILURE, exit_code.stderr
        elif exit_code.returncode != 0:
            return FResult.ERROR, exit_code.stderr

        return FResult.SAFE, "its safe"

    def validate_individual(self, filename) -> (FResult, str, float):
        fresult, msg = self.validate_compiler(self.target_name, filename)
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
