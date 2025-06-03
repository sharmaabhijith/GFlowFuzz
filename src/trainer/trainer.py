"""
Instruction-level Fuzzing Trainer using GFlowNet principles.
This module handles instruction sequence generation and evaluation
instead of token-by-token generation.
"""

import os
import math
import time
import torch
import traceback
import pandas as pd

from rich.traceback import install
install()
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from SUT.base_sut import FResult
from distiller_LM import Distiller, DistillerConfig
from instruct_LM import Sampler, Instructor, InstructorConfig, InstructionBuffer
from coder_LM import Coder, CoderConfig
from SUT import make_SUT, SUTConfig
from oracle import Inspector
from logger import LEVEL
from trainer.utils import TrainerConfig, FuzzerConfig, write_to_file, CompilationRecorder
from trainer.checkpointer import CheckpointManager
from logger import GlobberLogger, LEVEL

class Fuzzer:
    """Class for managing the fuzzing process using GFlowNet principles."""

    def __init__(
        self,
        SUT_config: SUTConfig,
        fuzzer_config: FuzzerConfig,
        distiller_config: DistillerConfig,
        instructor_config: InstructorConfig,
        coder_config: CoderConfig,
        trainer_config: TrainerConfig,
        target_name: str,
        output_folders: dict,
    ):
        """
        Initialize the fuzzer with SUT and configuration parameters.
        
        Args:
            SUT (BaseSUT): The system under test to fuzz.
            fuzzer_config (FuzzerConfig): Configuration for fuzzing process.
            distiller_config (DistillerConfig): Configuration for the distiller module.
            instructor_config (InstructorConfig): Configuration for the instructor module.
            coder_config (CoderConfig): Configuration for the coder module.
            trainer_config (TrainerConfig): Configuration for training parameters.
        """
        # Ensure all modules of the fuzzer are set to the same device
        coder_config.device = trainer_config.device
        instructor_config.device = trainer_config.device
        SUT_config.device = trainer_config.device
        self.SUT = make_SUT(SUT_config, target_name)
        self.number_of_iterations = fuzzer_config.number_of_iterations
        self.total_time = fuzzer_config.total_time
        self.output_folders = output_folders
        self.resume = fuzzer_config.resume
        self.otf = fuzzer_config.otf
        self.max_norm = trainer_config.max_norm
        # Initialize the 3 modules of the framework
        self.coder = Coder(
            coder_config=coder_config, 
            api_driven=coder_config.api_name != "local"
        )
        self.distiller = Distiller(
            distiller_config=distiller_config, 
            coder=self.coder, 
            SUT=self.SUT,
            output_folder=self.output_folders["distilled_prompts"]
        )
        self.instructor = Sampler(
            instructor_config=instructor_config,
            trainer_config=trainer_config
        )
        self.oracle = Inspector(self.SUT)
        # Initialize the fuzzing variables
        self.count = 0
        self.start_time = 0
        self.logger = GlobberLogger("fuzzer.log")
        self.logger.log("Fuzzer initialized.", LEVEL.INFO)
        self.logger.log(f"SUTConfig: {SUT_config}", LEVEL.TRACE)
        self.logger.log(f"FuzzerConfig: {fuzzer_config}", LEVEL.TRACE)
        self.logger.log(f"CoderConfig: {coder_config}", LEVEL.TRACE)
        self.logger.log(f"TrainerConfig: {trainer_config}", LEVEL.TRACE)
        self.logger.log(f"DistillerConfig: {distiller_config}", LEVEL.TRACE)
        self.logger.log(f"InstructorConfig: {instructor_config}", LEVEL.TRACE)
        
        # Initialize compilation recorder
        self.compilation_recorder = CompilationRecorder(
            output_folder=self.output_folders["logs"],
            logger=self.logger
        )

    def __get_resume_count(self) -> int:
        """
        Get the next count for resuming a fuzzing run.
        
        Returns:
            int: The next count for file naming, based on existing output files.
        """
        if not self.resume:
            return 0
            
        try:
            n_existing = [
                int(f.split(".")[0])
                for f in os.listdir(self.output_folders["fuzz_code"])
                if f.endswith(".fuzz")
            ]
            if not n_existing:
                return 0
            n_existing.sort(reverse=True)
            return n_existing[0] + 1
        except Exception as e:
            print(f"Error getting resume count: {e}")
            return 0

    def __train_off_policy(self, batch_size: int = 4, steps: int = 1):
        self.logger.log(f"Starting off-policy training: batch_size={batch_size}, steps={steps}", LEVEL.TRACE)
        for step in range(steps):
            batch = self.ibuffer.sample(batch_size)
            self.logger.log(f"Off-policy step {step}, batch: {str(batch)[:500]}", LEVEL.VERBOSE)
            if not batch:
                self.logger.log("Off-policy batch empty, breaking.", LEVEL.TRACE)
                break
            for t in batch:
                if t["reward"] <= 0:
                    self.logger.log(f"Skipping sample with non-positive reward: {t['reward']}", LEVEL.TRACE)
                    continue
                log_prob_sum = sum(t["f_log_probs"])
                log_z_sum = t["logZ"]
                log_reward = math.log(t["reward"])
                self.logger.log(f"Off-policy sample: log_prob_sum={log_prob_sum}, log_z_sum={log_z_sum}, log_reward={log_reward}", LEVEL.VERBOSE)
                self.instructor.train_step(
                    torch.tensor(log_z_sum, device=self.instructor.device),
                    torch.tensor(log_prob_sum, device=self.instructor.device),
                    torch.tensor(log_reward, device=self.instructor.device)
                )
        self.logger.log("Off-policy training complete.", LEVEL.TRACE)

    def train(self) -> None:
        self.logger.log("Fuzzer training started.", LEVEL.INFO)
        start_time = time.time()
        try:
            self.logger.log("Generating initial prompt using distiller...", LEVEL.TRACE)
            message = self.SUT.prompt_used["docstring"]
            self.initial_prompt = self.distiller.generate_prompt(message)
            self.prompt = self.initial_prompt
            self.start_time = time.time()
            Bug_count = 0
            with Progress(
                TextColumn("Fuzzing • [progress.percentage]{task.percentage:>3.0f}%"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
            ) as p:
                task = p.add_task("Fuzzing", total=self.number_of_iterations)
                self.count = self.__get_resume_count()
                if self.resume and self.count > 0:
                    log = f" (resuming from {self.count})"
                    p.console.print(log)
                    self.logger.log(f"Resuming from count {self.count}", LEVEL.INFO)
                p.update(task, advance=self.count)
                while (
                    self.count < self.number_of_iterations
                    ) and (
                    time.time() - self.start_time < self.total_time * 3600
                    ):
                    iter_start = time.time()
                    self.logger.log(f"Fuzzing iteration {self.count}", LEVEL.INFO)
                    final_prompt, log_probs, log_zs = self.instructor.sample_instruction_sequence(self.prompt)
                    write_to_file(os.path.join(
                        self.output_folders["instruct_prompts"], f"{self.count}.txt"), 
                        final_prompt
                    )
                    self.logger.log(f"Generated code samples:", LEVEL.INFO)
                    fo = self.coder.generate_code(prompt=final_prompt)
                    self.logger.log(f"Evaluating code sample:", LEVEL.INFO)
                    f_result, error, coverage, reward = self.oracle.inspect(
                        fo = fo,
                        output_folder = self.output_folders["fuzz_code"],
                        count = self.count,
                        otf = self.otf,
                    )
                    if f_result == FResult.ERROR:
                        write_to_file(os.path.join(
                            self.output_folders["bugs"], f"{self.count}.txt"), 
                            [final_prompt, fo]
                        )
                        Bug_count += 1
                    self.compilation_recorder.update_record(
                        iteration=self.count,
                        error=error,
                        coverage=coverage,
                        reward=reward
                    )
                    self.instructor.update_strategy(f_result, reward)
                    iter_end = time.time()
                    self.logger.log(f"Iteration {self.count} duration: {iter_end - iter_start:.2f}s", LEVEL.INFO)
                    self.logger.log(f"Compilation output type: {str(error)}", LEVEL.INFO)
                    self.logger.log(f"Updated Bug count: {Bug_count}", LEVEL.INFO)
                    self.count += 1
            end_time = time.time()
            self.logger.log(f"Fuzzer training completed in {end_time - start_time:.2f}s.", LEVEL.INFO)
        except Exception as e:
            self.logger.log(f"Error during training: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise

    def evaluate_all(self) -> None:
        """
        Evaluate all generated outputs against the oracle.

        This method delegates to the SUT's validate_all method to perform
        comprehensive evaluation of all outputs.
        """
        self.SUT.validate_all()




