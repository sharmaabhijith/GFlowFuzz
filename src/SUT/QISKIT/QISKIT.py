import ast
import os
import re
import subprocess
from enum import Enum
from multiprocessing import Process
from threading import Timer
from typing import List, Tuple, Union
import time
import traceback

from SUT.base_sut import FResult, BaseSUT
from SUT.utils import SUTConfig
from logger import GlobberLogger, LEVEL
from oracle.coverage import CoverageManager, Tool
import pathlib

# create an enum with some code snippets


class Snippet(Enum):
    READ_ANY_QASM_SAME_FOLDER = """
from qiskit import QuantumCircuit
import glob
class CustomFuzzAllException(Exception):
    pass
qasm_files = glob.glob("*.qasm")
for qasm_file in qasm_files:
    try:
        print(f"Importing {qasm_file}")
        qc = QuantumCircuit.from_qasm_file(qasm_file)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"File: {qasm_file}")
        content = open(qasm_file, "r").read()
        print(f"Content: {content}")
        raise CustomFuzzAllException(e)
    """
    CHECK_ANY_CIRCUIT = """
# ==================== ORACLE ====================
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
class CustomFuzzAllException(Exception):
    pass
# get any the global variables (including the circuits)
global_vars = list(globals().keys())
# keep all those that are QuantumCircuit
circuits = [
    globals()[var] for var in global_vars
    if isinstance(globals()[var], QuantumCircuit)
]
try:
    # transpile them
    for circuit in circuits:
        for lvl in range(0, 4):
            res = transpile(circuit, optimization_level=lvl)
            # print(f"Optimization level {lvl} for circuit {circuit.name}")
            # print(res.draw())

    # conert them to qasm and back
    for circuit in circuits:
        # print(f"Converting to qasm and back for circuit {circuit.name}")
        QuantumCircuit().from_qasm_str(circuit.qasm())
except Exception as e:
    raise CustomFuzzAllException(e)
# ==================== ORACLE ====================
"""
    TRANSPILE_QC_OPT_LVL_0 = """
from qiskit.compiler import transpile
qc = transpile(qc, optimization_level=0)
"""
    TRANSPILE_QC_OPT_LVL_1 = """
from qiskit.compiler import transpile
qc = transpile(qc, optimization_level=1)
"""
    TRANSPILE_QC_OPT_LVL_2 = """
from qiskit.compiler import transpile
qc = transpile(qc, optimization_level=2)
"""
    TRANSPILE_QC_OPT_LVL_3 = """
from qiskit.compiler import transpile
qc = transpile(qc, optimization_level=3)
"""


class Qiskit_SUT(BaseSUT):
    def __init__(self, sut_config: SUTConfig):
        super().__init__(sut_config)

        # Qiskit_SUT uses its own logger file, but respect the log level from config
        self.logger = GlobberLogger("fuzzer.log", level=LEVEL.TRACE)
        self.logger.log("Qiskit_SUT initialized with SUTConfig.", LEVEL.INFO)
        self.SYSTEM_MESSAGE = "You are a Qiskit Fuzzer"
        self.prompt_used = self._create_prompt_from_config(sut_config)
        self.logger.log(f"Unsupported template or no template for prompt creation: {sut_config.template}", LEVEL.INFO)
        self.coverage_manager = CoverageManager(Tool.QISKIT, pathlib.Path(f"/tmp/out{self.CURRENT_TIME}"))
        self.prev_coverage = 0
        self.lambda_ = sut_config.lambda_hyper
        self.beta1_ = sut_config.beta1_hyper

    def write_back_file(self, code):
        try:
            with open(f"/tmp/temp{self.CURRENT_TIME}.py", "w", encoding="utf-8") as f:
                f.write(code)
        except Exception:
            pass
        return f"/tmp/temp{self.CURRENT_TIME}.py"

    def wrap_prompt(self, prompt: str) -> str:
        return f"'''{prompt}'''\n{self.prompt_used['separator']}\n{self.prompt_used['begin']}"

    def wrap_in_comment(self, prompt: str) -> str:
        return f'""" {prompt} """'

    def filter(self, code) -> bool:
        clean_code = code.replace(self.prompt_used["begin"], "").strip()
        if self.prompt_used["target_api"] not in clean_code:
            return False
        return True

    def clean(self, code: str) -> str:
        code = self._comment_remover(code)
        return code

    def clean_code(self, code: str) -> str:
        """Remove all comments and empty lines from a string of Python code."""
        code = code.replace(self.prompt_used["begin"], "").strip()
        code = self._comment_remover(code)
        code = "\n".join(
            [
                line
                for line in code.split("\n")
                if line.strip() != "" and line.strip() != self.prompt_used["begin"]
            ]
        )
        return code

    def _comment_remover(self, code: str) -> str:
        """Remove all comments from a string of Python code."""
        # Remove inline comments
        code = re.sub(r"#.*", "", code)
        # Remove block comments
        code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
        return code

    def _validate_static(self, filename) -> Tuple[FResult, str]:
        """Validate the input at the filename path statically (no execution).

        Typically, this is done by checking the return code of the compiler.
        For dynamically typed languages, we could perform both a parser and
        static analysis on the code.
        """

        try:
            content = open(filename, "r", encoding="utf-8").read()
            ast.parse(content)
        except Exception as e:
            return FResult.FAILURE, f"parsing failed {e}"

        return FResult.SAFE, "its safe"

    def _kill_program(self, filename: str) -> None:
        """Kill a program running at the filename path."""
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

    def _remove_partial_lines(self, content: str) -> None:
        """Remove the last line if it is not ending with new line."""
        if not content.endswith("\n"):
            lines = content.split("\n")
            lines = lines[:-1]
            content = "\n".join(lines)
        return content

    def _delete_last_line_inplace(self, filename: str) -> None:
        """Delete the last line of a file."""
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.split("\n")
        lines = lines[:-1]
        content = "\n".join(lines)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    def validate_individual(self, filepath: str) -> Tuple[FResult, str, float]:
        self.logger.log(f"validate_individual called with filepath: {filepath}", LEVEL.TRACE)
        start_time = time.time()
        try:
            self.logger.log("--------------------------", level=LEVEL.VERBOSE)
            parser_result, parser_msg = self._validate_static(filepath)
            self.logger.log(f"Static validation result: {parser_result}, message: {str(parser_msg)[:200]}", LEVEL.TRACE)
            if parser_result != FResult.SAFE:
                self.logger.log("Parser not SAFE, attempting recovery by deleting last line and retrying.", LEVEL.TRACE)
                self._delete_last_line_inplace(filepath)
                parser_result, parser_msg = self._validate_static(filepath)
                self.logger.log(f"Recovery static validation result: {parser_result}, message: {str(parser_msg)[:200]}", LEVEL.TRACE)
                if parser_result != FResult.SAFE:
                    self.logger.log(f"Static validation failed after recovery. Returning. Message: {parser_msg}", LEVEL.INFO)
                    return parser_result, parser_msg, 0.0
            
            # Use oracle_type from sut_config
            oracle = self.sut_config.oracle_type 
            self.logger.log(f"Oracle selected: {oracle}", LEVEL.TRACE)
            if oracle == "crash":
                fresult, msg = self._validate_with_crash_oracle(filepath)
            elif oracle == "diff":
                fresult, msg = self._validate_with_diff_opt_levels(filepath)
            elif oracle == "metamorphic":
                fresult, msg = self._validate_with_QASM_roundtrip(filepath)
            elif oracle == "opt_and_qasm":
                fresult, msg = self._validate_any_circuit(filepath)
            else:
                self.logger.log(f"Unknown oracle: {oracle}, defaulting to crash oracle.", LEVEL.INFO)
                fresult, msg = self._validate_with_crash_oracle(filepath)
            
            self.coverage_manager.run_once()
            new_cov = self.coverage_manager.update_total()
            coverage_diff = new_cov - self.prev_coverage
            bug = 1 if fresult in (FResult.FAILURE, FResult.ERROR) else 0
            reward = coverage_diff + self.lambda_ * new_cov + self.beta1_ * bug
            self.logger.log(f"Coverage: new={new_cov}, prev={self.prev_coverage}, diff={coverage_diff}, bug={bug}, reward={reward}", LEVEL.TRACE)
            self.prev_coverage = new_cov
            if fresult == FResult.SAFE:
                result = (FResult.SAFE, f"{msg}\nCoverage: {new_cov}", reward)
            elif fresult == FResult.ERROR:
                result = (FResult.ERROR, f"{msg}\nCoverage: {new_cov}", reward)
            elif fresult == FResult.TIMED_OUT:
                result = (FResult.ERROR, f"timed out\nCoverage: {new_cov}", reward)
            elif fresult == FResult.FAILURE:
                result = (FResult.FAILURE, f"{msg}\nCoverage: {new_cov}", reward)
            else:
                result = (FResult.TIMED_OUT, f"Coverage: {new_cov}", reward)
            end_time = time.time()
            self.logger.log(f"validate_individual completed in {end_time - start_time:.2f}s, result: {result[0]}, message: {str(result[1])[:200]}, reward: {result[2]}", LEVEL.TRACE)
            return result
        except Exception as e:
            self.logger.log(f"Error during validate_individual: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise

    def _validate_with_diff_opt_levels(self, filepath: str) -> Tuple[FResult, str]:
        self.logger.log(f"_validate_with_diff_opt_levels called with filepath: {filepath}", LEVEL.TRACE)
        start_time = time.time()
        try:
            program_content = open(filepath, "r", encoding="utf-8").read()
            self.logger.log(f"python {filepath}:", level=LEVEL.TRACE)
            self.logger.log("\n" + program_content[:300], level=LEVEL.VERBOSE)
            self.logger.log("-" * 20, level=LEVEL.TRACE)
            if "qc." not in program_content:
                self.logger.log("No circuit `qc.` found in program_content.", LEVEL.TRACE)
                return FResult.FAILURE, "no circuit `qc.` found"
            OPT_LEVELS_SNIPPETS = [
                Snippet.TRANSPILE_QC_OPT_LVL_0,
                Snippet.TRANSPILE_QC_OPT_LVL_3,
            ]
            exit_codes = {}
            for lvl, opt_level_snippet in zip([0, 3], OPT_LEVELS_SNIPPETS):
                exit_codes[opt_level_snippet] = None
                new_filename = f"/tmp/temp{self.CURRENT_TIME}_lvl_{lvl}.py"
                i_content = program_content + "\n" + str(opt_level_snippet.value)
                with open(new_filename, "w", encoding="utf-8") as f:
                    f.write(i_content)
                try:
                    cmd = f"python {new_filename}"
                    self.logger.log(f"Running subprocess: {cmd}", LEVEL.TRACE)
                    exit_code = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        encoding="utf-8",
                        timeout=15,
                        text=True,
                    )
                    exit_codes[opt_level_snippet] = exit_code
                    self.logger.log(f"Execution result: {exit_code}", level=LEVEL.VERBOSE)
                except ValueError as e:
                    self._kill_program(filepath)
                    self.logger.log(f"ValueError during subprocess: {e}", LEVEL.INFO)
                    return FResult.FAILURE, f"ValueError: {str(e)}"
                except subprocess.TimeoutExpired:
                    self._kill_program(filepath)
                    self.logger.log(f"TimeoutExpired for opt level {lvl}", LEVEL.INFO)
                    return FResult.TIMED_OUT, f"timed out for opt level {str(lvl)}"
            for opt_level in OPT_LEVELS_SNIPPETS:
                if exit_codes[opt_level] is None:
                    self.logger.log(f"No exit code found for opt level {opt_level}", LEVEL.INFO)
                    return (
                        FResult.ERROR,
                        f"no exit code found for opt level {str(opt_level)}",
                    )
            if exit_codes[0].stdout != exit_codes[3].stdout:
                self.logger.log("Different outputs for opt levels 0 and 3.", LEVEL.INFO)
                return FResult.ERROR, "different outputs"
            end_time = time.time()
            self.logger.log(f"_validate_with_diff_opt_levels completed in {end_time - start_time:.2f}s", LEVEL.TRACE)
            return FResult.SAFE, "its safe"
        except Exception as e:
            self.logger.log(f"Error during _validate_with_diff_opt_levels: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise

    def _validate_with_crash_oracle(self, filepath: str) -> Tuple[FResult, str]:
        """Check whether the transpiler returns an exception or not.

        If the exception is a TranspilerError, then the program is valid and
        the bug is in the transpiler. If the exception is another one, then
        the program is invalid.
        """
        try:
            cmd = f"python {filepath}"
            exit_code = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                encoding="utf-8",
                timeout=5,
                text=True,
            )
            self.logger.log(f"Execution result: {exit_code}", level=LEVEL.VERBOSE)
            if exit_code.returncode == 0:
                return FResult.SAFE, "its safe"
            else:
                # check if the output contained a TranspilerError
                if "TranspilerError" in exit_code.stderr:
                    return FResult.ERROR, exit_code.stderr
                else:
                    return FResult.FAILURE, "its safe"
        except ValueError as e:
            self._kill_program(filepath)
            return FResult.FAILURE, f"ValueError: {str(e)}"
        except subprocess.TimeoutExpired:
            # kill program
            self._kill_program(filepath)
            return FResult.TIMED_OUT, f"timed out"

    def _validate_with_QASM_roundtrip(self, filepath: str) -> Tuple[FResult, str]:
        """Check if the exported qasm (if any) can be parsed by the QASM parser."""
        # append the snippet to read the qasm files
        program_content = open(filepath, "r", encoding="utf-8").read()
        program_content += "\n" + str(Snippet.READ_ANY_QASM_SAME_FOLDER.value)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(program_content)
            f.close()
        try:
            cmd = f"python {filepath}"
            exit_code = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                encoding="utf-8",
                timeout=5,
                text=True,
            )
            self.logger.log(f"Execution result: {exit_code}", level=LEVEL.VERBOSE)
            if exit_code.returncode == 0:
                return FResult.SAFE, "its safe"
            else:
                if "CustomFuzzAllException" in exit_code.stderr:
                    return FResult.ERROR, "CustomFuzzAllException: POTENTIAL BUG"
                else:
                    return FResult.FAILURE, exit_code.stderr
        except ValueError as e:
            self._kill_program(filepath)
            return FResult.FAILURE, f"ValueError: {str(e)}"
        except subprocess.TimeoutExpired:
            # kill program
            self._kill_program(filepath)
            return FResult.TIMED_OUT, f"timed out"

    def _validate_any_circuit(self, filepath: str) -> Tuple[FResult, str]:
        """Check if any any circuit can be transpiled and converted to qasm.

        To retrieve the circuit in the program we use the global variables.
        """
        program_content = open(filepath, "r", encoding="utf-8").read()
        program_content += "\n" + str(Snippet.CHECK_ANY_CIRCUIT.value)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(program_content)
            f.close()
        try:
            cmd = f"python {filepath}"
            exit_code = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                encoding="utf-8",
                timeout=5,
                text=True,
            )
            self.logger.log(f"Execution result: {exit_code}", level=LEVEL.VERBOSE)
            if exit_code.returncode == 0:
                return FResult.SAFE, "its safe"
            else:
                if "CustomFuzzAllException" in exit_code.stderr:
                    return FResult.ERROR, "CustomFuzzAllException: POTENTIAL BUG"
                else:
                    return FResult.FAILURE, exit_code.stderr
        except ValueError as e:
            self._kill_program(filepath)
            return FResult.FAILURE, f"ValueError: {str(e)}"
        except subprocess.TimeoutExpired:
            # kill program
            self._kill_program(filepath)
            return FResult.TIMED_OUT, f"timed out"
