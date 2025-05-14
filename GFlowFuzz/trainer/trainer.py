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

from rich.traceback import install
install()
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from distiller_LM import Distiller, DistillerConfig
from instruct_LM import Instructor, InstructorConfig, InstructionBuffer
from coder_LM import Coder, CoderConfig
from SUT import make_SUT
from oracle import Inspector

from .utils import TrainerConfig, FuzzerConfig, SUTConfig
from .checkpointer import CheckpointManager
from GFlowFuzz.logger import GlobberLogger, LEVEL


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
    ):
        """
        Initialize the fuzzer with SUT and configuration parameters.
        
        Args:
            SUT (base_SUT): The system under test to fuzz.
            fuzzer_config (FuzzerConfig): Configuration for fuzzing process.
            distiller_config (DistillerConfig): Configuration for the distiller module.
            instructor_config (InstructorConfig): Configuration for the instructor module.
            coder_config (CoderConfig): Configuration for the coder module.
            trainer_config (TrainerConfig): Configuration for training parameters.
        """
        self.SUT = make_SUT(SUT_config)
        self.number_of_iterations = fuzzer_config.number_of_iterations
        self.total_time = fuzzer_config.total_time
        self.output_folder = fuzzer_config.output_folder
        self.resume = fuzzer_config.resume
        self.otf = fuzzer_config.otf
        self.max_norm = trainer_config.max_norm
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        # Initialize the 3 modules of the framework
        self.coder = Coder(coder_config)
        self.distiller = Distiller(distiller_config, self.coder, self.SUT)
        self.instructor = Instructor(instructor_config)
        self.oracle = Inspector(self.SUT)
        self.ibuffer = InstructionBuffer(
            max_size=trainer_config.buffer_size,
            prioritization=trainer_config.prioritization
        )
        # Setup instructor's model, optimizer, scheduler, tokenizer, and projection layer
        self.instructor.setup_model_and_optimizer(trainer_config)
        # Save reference for checkpointing
        self.checkpointer = CheckpointManager(
            save_dir=self.output_folder,
            exp_name="instructor",
            model=self.instructor.model,
            optimizer=self.instructor.optimizer,
            scheduler=self.instructor.scheduler,
            ibuffer=self.ibuffer
        )
        # Initialize the fuzzing variables
        self.count = 0
        self.start_time = 0
        self.logger = GlobberLogger("fuzzer.log", level=LEVEL.TRACE)
        self.logger.log("Fuzzer initialized.", LEVEL.INFO)
        self.logger.log(f"FuzzerConfig: {fuzzer_config}", LEVEL.TRACE)
        self.logger.log(f"TrainerConfig: {trainer_config}", LEVEL.TRACE)
        self.logger.log(f"DistillerConfig: {distiller_config}", LEVEL.TRACE)
        self.logger.log(f"InstructorConfig: {instructor_config}", LEVEL.TRACE)
        self.logger.log(f"CoderConfig: {coder_config}", LEVEL.TRACE)
        self.logger.log(f"Output folder: {self.output_folder}", LEVEL.TRACE)
        self.logger.log(f"Resume: {self.resume}, OTF: {self.otf}", LEVEL.TRACE)
        
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
                for f in os.listdir(self.output_folder)
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
            self.initial_prompt = self.distiller.generate_prompt()
            self.logger.log(f"Initial prompt: {str(self.initial_prompt)[:300]}", LEVEL.VERBOSE)
            self.prompt = self.initial_prompt
            self.start_time = time.time()
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
                    self.logger.log(f"Fuzzing iteration {self.count}", LEVEL.TRACE)
                    self.logger.log(f"Prompt for iteration: {str(self.prompt)[:300]}", LEVEL.VERBOSE)
                    instructions, log_probs, log_zs = self.instructor.generate_instruction_sequence(self.prompt)
                    self.logger.log(f"Instructions: {str(instructions)[:300]}", LEVEL.VERBOSE)
                    self.logger.log(f"Log_probs: {str(log_probs)[:300]}", LEVEL.VERBOSE)
                    self.logger.log(f"Log_zs: {str(log_zs)[:300]}", LEVEL.VERBOSE)
                    fos = self.coder.generate_code(prompt=instructions)
                    self.logger.log(f"Generated code samples: {str(fos)[:500]}", LEVEL.VERBOSE)
                    for fo in fos:
                        self.logger.log(f"Evaluating code sample: {str(fo)[:300]}", LEVEL.TRACE)
                        _, _, reward = self.oracle.inspect(
                            fo = fo,
                            output_folder = self.output_folder,
                            count = self.count,
                            otf = self.otf,
                        )
                        self.logger.log(f"Reward: {reward}", LEVEL.VERBOSE)
                        loss_value = self.instructor.train_step(
                            log_zs, log_probs, reward, self.max_norm
                        )
                        self.logger.log(f"Loss value: {loss_value}", LEVEL.VERBOSE)
                        self.ibuffer.add(
                            states=[],
                            actions=[],
                            instructions=instructions,
                            forward_logprobs=log_probs,
                            backward_logprobs=[],
                            final_reward=reward.item() if hasattr(reward, "item") else reward,
                            logZ=log_zs.item() if hasattr(log_zs, "item") else log_zs
                        )
                        self.logger.log(f"Buffer size after add: {len(self.ibuffer)}", LEVEL.TRACE)
                        self.__train_off_policy(batch_size=2, steps=1)
                        if self.count % 100 == 0:
                            self.logger.log(f"Checkpoint saved at step {self.count}", LEVEL.INFO)
                        iter_end = time.time()
                        self.logger.log(f"Iteration {self.count} duration: {iter_end - iter_start:.2f}s", LEVEL.TRACE)
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




