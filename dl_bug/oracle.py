#!/usr/bin/env python3
"""
PyTorch Fuzzing Reward Model
Combines subprocess-based execution with dense reward modeling for LLM training.
Differentiates between user coding errors and potential PyTorch library bugs.
"""

import ast
import json
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set

class Outcome(Enum):
    SUCCESS = "success"                         # Code executed successfully
    BUG = "bug"                                 # Potential PyTorch library bug
    USER_CODE_ERROR = "user_code_error"         # User coding mistake
    TIMEOUT = "timeout"                         # Execution timeout
    CRASH = "crash"                             # Unexpected system crash

@dataclass
class RewardType:
    outcome: Outcome
    reward_score: float                         # Dense reward (-1.0 to 1.0)
    confidence: float                           # Confidence in classification (0-1)
    returncode: int                             # Process return code
    stderr: str                                 # Error output
    execution_time: float                       # Time taken to execute
    code_quality_score: float                   # Quality of the fuzzing code (0-1)
    exploration_bonus: float                    # Bonus for exploring new APIs (0-1)
    bug_likelihood: float                       # Likelihood this is a real bug (0-1)

class RewardModel:
    """Dense reward model specifically designed for fuzzing"""
    
    def __init__(self):
        # Signal-based crashes (strong indicators of library bugs)
        self.SIG_CRASHES = {-signal.SIGSEGV, -signal.SIGABRT, -signal.SIGILL, -signal.SIGFPE}
        
        # Patterns indicating PyTorch internal failures (HIGH REWARD)
        self.BUG_PATTERNS = [
            re.compile(r"(internal|fatal) assert", re.I),
            re.compile(r"check failed", re.I),
            re.compile(r"CUDA error|device-side assert", re.I),
            re.compile(r"ubsan|asan|msan|tsan", re.I),
            re.compile(r"RuntimeError.*CUDA.*internal", re.I),
            re.compile(r"Assertion.*failed.*torch", re.I),
            re.compile(r"core dumped", re.I),
            re.compile(r"Segmentation fault", re.I),
            re.compile(r"torch.*backend.*error", re.I),
            re.compile(r"c10::Error", re.I),
            re.compile(r"THCudaCheck|CUDA_ERROR", re.I),
        ]
        
        # PyTorch library path detection
        self.PYTORCH_PATH_RE = re.compile(r"site-packages[/\\]torch", re.I)
        
        # User error patterns (NEGATIVE REWARD)
        self.USER_ERROR_PATTERNS = [
            re.compile(r"NameError.*not defined", re.I),
            re.compile(r"TypeError.*takes.*positional argument", re.I),
            re.compile(r"ValueError.*Expected.*got", re.I),
            re.compile(r"AttributeError.*has no attribute", re.I),
            re.compile(r"ImportError|ModuleNotFoundError", re.I),
            re.compile(r"IndentationError|TabError", re.I),
        ]
        
        # Expected PyTorch validation errors (LOW POSITIVE REWARD)
        self.EXPECTED_ERROR_PATTERNS = [
            re.compile(r"RuntimeError.*size mismatch", re.I),
            re.compile(r"RuntimeError.*dimension.*out of range", re.I),
            re.compile(r"ValueError.*Expected tensor", re.I),
            re.compile(r"TypeError.*expected.*Tensor", re.I),
        ]
        
        # Track API exploration for bonus rewards
        self.api_usage_history: Set[str] = set()
        self.rare_combinations: Set[frozenset] = set()
        
        # PyTorch API categories for analysis
        self.pytorch_apis = {
            'tensor_ops': ['tensor', 'zeros', 'ones', 'randn', 'rand', 'empty'],
            'math_ops': ['add', 'sub', 'mul', 'div', 'matmul', 'mm', 'bmm'],
            'nn_ops': ['conv1d', 'conv2d', 'relu', 'sigmoid', 'softmax', 'dropout'],
            'cuda_ops': ['cuda', 'to', 'device'],
            'autograd': ['backward', 'grad', 'requires_grad'],
            'advanced': ['scatter', 'gather', 'index_select', 'masked_select']
        }

    def write_tmp_file(self, code: str) -> Path:
        """Write code to temporary file for subprocess execution"""
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "pytorch_fuzz.py"
        path.write_text(textwrap.dedent(code))
        return path

    def run_subprocess(self, py_file: Path, timeout: int = 10) -> subprocess.CompletedProcess:
        """Execute Python file in subprocess with timeout"""
        return subprocess.run(
            [sys.executable, str(py_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def evaluate_fuzzing_code(self, code: str, timeout: int = 10) -> RewardType:
        """
        Main entry point: evaluate fuzzing code and return dense reward
        """
        start_time = time.time()
        
        # 1. Quick syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            return RewardType(
                outcome=Outcome.USER_CODE_ERROR,
                reward_score=-0.9,
                confidence=0.95,
                returncode=1,
                stderr=str(e),
                execution_time=0.0,
                code_quality_score=0.1,
                exploration_bonus=0.0,
                bug_likelihood=0.0,
            )
        
        # 2. Analyze code quality and exploration bonus
        code_quality = self._analyze_code_quality(code)
        exploration_bonus = self._calculate_exploration_bonus(code)
        
        # 3. Execute in subprocess
        try:
            path = self.write_tmp_file(code)
            result = self.run_subprocess(path, timeout)
            execution_time = time.time() - start_time
            
            return self._classify_execution_result(
                result, code, execution_time, code_quality, exploration_bonus
            )
            
        except subprocess.TimeoutExpired:
            return RewardType(
                outcome=Outcome.TIMEOUT,
                reward_score=-0.2,  # Slight negative for timeout
                confidence=0.8,
                returncode=-1,
                stderr="Execution timed out",
                execution_time=timeout,
                code_quality_score=code_quality,
                exploration_bonus=exploration_bonus,
                bug_likelihood=0.3,  # Could be infinite loop bug
            )
        except Exception as e:
            return RewardType(
                outcome=Outcome.CRASH,
                reward_score=0.0,
                confidence=0.5,
                returncode=-999,
                stderr=str(e),
                execution_time=time.time() - start_time,
                code_quality_score=code_quality,
                exploration_bonus=exploration_bonus,
                bug_likelihood=0.4,
            )

    def _classify_execution_result(self, result: subprocess.CompletedProcess, 
                                 code: str, execution_time: float,
                                 code_quality: float, exploration_bonus: float) -> RewardType:
        """Classify subprocess execution result and assign rewards"""
        
        # SUCCESS CASE
        if result.returncode == 0:
            bug_likelihood = self._analyze_output_for_bugs(result.stdout, result.stderr)
            base_reward = 0.4  # Base reward for successful execution
            if bug_likelihood > 0.6:
                base_reward = 0.8
                outcome = Outcome.BUG
            else:
                outcome = Outcome.SUCCESS
            if execution_time > 5.0:  # Suspiciously slow
                bug_likelihood += 0.1
                base_reward += 0.1
            
            final_reward = np.clip(
                base_reward + exploration_bonus * 0.3 + (code_quality - 0.5) * 0.2,
                -1.0, 1.0
            )
            
            return RewardType(
                outcome=outcome,
                reward_score=final_reward,
                confidence=0.7,
                returncode=0,
                stderr=result.stderr,
                execution_time=execution_time,
                code_quality_score=code_quality,
                exploration_bonus=exploration_bonus,
                bug_likelihood=bug_likelihood,
            )
        if result.returncode in self.SIG_CRASHES:
            return RewardType(
                outcome=Outcome.BUG,
                reward_score=0.95,  # Very high reward for crashes
                confidence=0.9,
                returncode=result.returncode,
                stderr=result.stderr,
                execution_time=execution_time,
                code_quality_score=code_quality,
                exploration_bonus=exploration_bonus,
                bug_likelihood=0.95,
            )
        
        stderr = result.stderr
        is_pytorch_error = self._is_pytorch_library_error(stderr)

        for pattern in self.BUG_PATTERNS:
            if pattern.search(stderr):
                reward = 0.9 if is_pytorch_error else 0.7
                return RewardType(
                    outcome=Outcome.BUG,
                    reward_score=reward,
                    confidence=0.85,
                    returncode=result.returncode,
                    stderr=stderr,
                    execution_time=execution_time,
                    code_quality_score=code_quality,
                    exploration_bonus=exploration_bonus,
                    bug_likelihood=0.85,
                )
        for pattern in self.USER_ERROR_PATTERNS:
            if pattern.search(stderr):
                return RewardType(
                    outcome=Outcome.USER_CODE_ERROR,
                    reward_score=-0.8,
                    confidence=0.9,
                    returncode=result.returncode,
                    stderr=stderr,
                    execution_time=execution_time,
                    code_quality_score=code_quality,
                    exploration_bonus=exploration_bonus,
                    bug_likelihood=0.0,
                )
        for pattern in self.EXPECTED_ERROR_PATTERNS:
            if pattern.search(stderr):
                reward = 0.2 if is_pytorch_error else -0.1
                return RewardType(
                    outcome=Outcome.USER_CODE_ERROR,
                    reward_score=reward,
                    confidence=0.8,
                    returncode=result.returncode,
                    stderr=stderr,
                    execution_time=execution_time,
                    code_quality_score=code_quality,
                    exploration_bonus=exploration_bonus,
                    bug_likelihood=0.1,
                )
        reward = 0.4 if is_pytorch_error else -0.2
        return RewardType(
            outcome=Outcome.BUG if is_pytorch_error else Outcome.USER_CODE_ERROR,
            reward_score=reward,
            confidence=0.5,
            returncode=result.returncode,
            stderr=stderr,
            execution_time=execution_time,
            code_quality_score=code_quality,
            exploration_bonus=exploration_bonus,
            bug_likelihood=0.4 if is_pytorch_error else 0.1,
        )

    def _is_pytorch_library_error(self, stderr: str) -> bool:
        """Check if error originated in PyTorch library code"""
        # Find the deepest (last) "File ..." line in traceback
        last_file_line = None
        for line in reversed(stderr.splitlines()):
            if line.strip().startswith("File"):
                last_file_line = line
                break
        if last_file_line and self.PYTORCH_PATH_RE.search(last_file_line):
            return True
        pytorch_error_indicators = [
            "torch.", "c10::", "THC", "CUDA", "at::Tensor"
        ]
        
        return any(indicator in stderr for indicator in pytorch_error_indicators)

    def _analyze_code_quality(self, code: str) -> float:
        """Analyze quality of PyTorch fuzzing code"""
        try:
            tree = ast.parse(code)
            quality_score = 0.5  # Base score
            has_torch_import = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'torch' in alias.name:
                            has_torch_import = True
                            quality_score += 0.2
                            break
                elif isinstance(node, ast.ImportFrom):
                    if node.module and 'torch' in node.module:
                        has_torch_import = True
                        quality_score += 0.2
            if not has_torch_import:
                quality_score -= 0.3  # Penalty for no PyTorch usage
            pytorch_ops = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'attr'):
                        attr_name = node.func.attr
                        for category, ops in self.pytorch_apis.items():
                            if attr_name in ops:
                                pytorch_ops += 1
                                break
            quality_score += min(0.3, pytorch_ops * 0.05)
            has_tensor_creation = any('tensor' in line.lower() or 'torch.' in line 
                                    for line in code.split('\n'))
            if has_tensor_creation:
                quality_score += 0.1
            if len(code.split('\n')) < 3:
                quality_score -= 0.2
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except SyntaxError:
            return 0.1

    def _calculate_exploration_bonus(self, code: str) -> float:
        """Calculate exploration bonus for API usage"""
        try:
            tree = ast.parse(code)
            current_apis = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'attr'):
                        current_apis.add(node.func.attr)
                    elif hasattr(node.func, 'id'):
                        current_apis.add(node.func.id)
            pytorch_apis_used = set()
            for api in current_apis:
                for category, ops in self.pytorch_apis.items():
                    if api in ops:
                        pytorch_apis_used.add(api)
            new_apis = pytorch_apis_used - self.api_usage_history
            self.api_usage_history.update(pytorch_apis_used)
            exploration_bonus = len(new_apis) * 0.15
            if len(pytorch_apis_used) >= 2:
                combination = frozenset(pytorch_apis_used)
                if combination not in self.rare_combinations:
                    self.rare_combinations.add(combination)
                    exploration_bonus += 0.25
            
            return min(1.0, exploration_bonus)
            
        except:
            return 0.0


    def _analyze_output_for_bugs(self, stdout: str, stderr: str) -> float:
        """Analyze execution output for potential bug indicators"""
        bug_likelihood = 0.0
        combined_output = stdout + stderr
        warning_patterns = [
            r"UserWarning.*might.*incorrect",
            r"Warning.*unexpected.*behavior",
            r"deprecated.*will.*removed",
        ]
        for pattern in warning_patterns:
            if re.search(pattern, combined_output, re.I):
                bug_likelihood += 0.1
        if re.search(r"nan|inf", combined_output, re.I):
            bug_likelihood += 0.3
        if re.search(r"slow|performance|memory", combined_output, re.I):
            bug_likelihood += 0.1
        
        return min(1.0, bug_likelihood)
    