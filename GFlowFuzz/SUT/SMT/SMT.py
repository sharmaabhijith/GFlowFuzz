import subprocess
import time
from typing import List, Union

import torch

from GFlowFuzz.SUT.base_sut import FResult, base_SUT
from GFlowFuzz.SUT.utils import SUTConfig
from GFlowFuzz.utils import comment_remover, LEVEL
from GFlowFuzz.oracle.coverage import CoverageManager, Tool
import pathlib


def _check_sat(stdout):
    sat = ""
    for x in stdout.splitlines():
        if "an invalid model was generated" in x.strip():
            sat = "invalid model"
            return sat

    for x in stdout.splitlines():
        if x.strip() == "unknown" or x.strip() == "unsupported":
            sat = "unknown"
            return sat

    for x in stdout.splitlines():
        if x.strip() == "unsat" or x.strip() == "sat":
            sat = x.strip()
            break
    return sat


# why is this needed? because sometimes the error could be suppressed in
# the return code of the smt solver however such error still exists.
def _check_error(stdout):
    error = False
    for x in stdout.splitlines():
        if x.strip().startswith("(error"):
            error = True
            break
    return error


# ignore cvc5 unary minus
# TODO: add additional rewriting rule to fix this
def _check_cvc5_parse_error(stdout):
    error = False
    for x in stdout.splitlines():
        if "Parse Error:" in x.strip():
            error = True
            break
    return error


class SMT_SUT(base_SUT):
    def __init__(self, sut_config: SUTConfig):
        super().__init__(sut_config)
        self.model = None  # to be declared
        self.prompt_used = self._create_prompt_from_config(sut_config)
        self.logger.log(f"Unsupported template or no template for prompt creation: {sut_config.template}", LEVEL.INFO)
        # Use special_eos from sut_config, with a default if not provided
        self.special_eos = sut_config.special_eos if sut_config.special_eos is not None else "#|"
        self.coverage_manager = CoverageManager(Tool.CVC5, pathlib.Path(f"/tmp/out{self.CURRENT_TIME}"))
        self.prev_coverage = 0
        self.lambda_ = sut_config.lambda_hyper
        self.beta1_ = sut_config.beta1_hyper

    def write_back_file(self, code):
        try:
            with open(
                "/tmp/temp{}.smt2".format(self.CURRENT_TIME), "w", encoding="utf-8"
            ) as f:
                f.write(code)
        except:
            pass
        return "/tmp/temp{}.smt2".format(self.CURRENT_TIME)

    def wrap_prompt(self, prompt: str) -> str:
        return (
            f"; {prompt}\n{self.prompt_used['separator']}\n{self.prompt_used['begin']}"
        )

    def wrap_in_comment(self, prompt: str) -> str:
        return f"; {prompt}"

    def filter(self, code) -> bool:
        if "assert" not in code:
            return False
        return True

    def clean(self, code: str) -> str:
        # remove logic set which can lead to parse errors
        # clean_code = "\n".join(
        #     [x for x in code.splitlines() if not x.startswith("(set-logic")]
        # )
        clean_code = comment_remover(code, lang="smt2")
        clean_code = "\n".join(
            [x for x in clean_code.splitlines() if not x.startswith("(set-option :")]
        )
        clean_code = "\n".join(
            [x for x in clean_code.splitlines() if not x.startswith("(get-proof)")]
        )
        return clean_code

    # remove any comments, or blank lines
    def clean_code(self, code: str) -> str:
        clean_code = comment_remover(code, lang="smt2")
        code = "\n".join(
            [
                line
                for line in clean_code.split("\n")
                if line.strip() != "" and line.strip() != self.prompt_used["begin"]
            ]
        )
        return code

    def validate_individual(self, filename) -> (FResult, str, float):
        try:
            cvc_exit_code = subprocess.run(
                f"{self.target_name} -m -i -q --check-models --lang smt2 {filename}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
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
            return FResult.TIMED_OUT, "CVC5 Timed out", 0.0
        except UnicodeDecodeError as ue:
            return FResult.FAILURE, "UnicodeDecodeError", 0.0

        if cvc_exit_code.returncode != 0:
            fresult = FResult.FAILURE
            msg = "CVC5:\n{}".format(
                cvc_exit_code.stdout + cvc_exit_code.stderr,
            )
        else:
            fresult = FResult.SAFE
            msg = "its safe"

        self.coverage_manager.run_once()
        new_cov = self.coverage_manager.update_total()
        coverage_diff = new_cov - self.prev_coverage
        bug = 1 if fresult in (FResult.FAILURE, FResult.ERROR) else 0
        reward = coverage_diff + self.lambda_ * new_cov + self.beta1_ * bug
        self.prev_coverage = new_cov

        if fresult == FResult.SAFE:
            return FResult.SAFE, f"{msg}\nCoverage: {new_cov}", reward
        elif fresult == FResult.ERROR:
            return FResult.ERROR, f"{msg}\nCoverage: {new_cov}", reward
        elif fresult == FResult.TIMED_OUT:
            return FResult.ERROR, f"timed out\nCoverage: {new_cov}", reward
        elif fresult == FResult.FAILURE:
            return FResult.FAILURE, f"{msg}\nCoverage: {new_cov}", reward
        else:
            return (FResult.TIMED_OUT, f"Coverage: {new_cov}", reward)
