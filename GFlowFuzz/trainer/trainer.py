"""
Instruction-level Fuzzing Trainer using GFlowNet principles.
This module handles instruction sequence generation and evaluation
instead of token-by-token generation.
"""

import os
import math
import time
import torch

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
from SUT import base_SUT
from oracle import Inspector

from .utils import TrainerConfig, FuzzerConfig
from .checkpointer import CheckpointManager


class Fuzzer:
    """Class for managing the fuzzing process using GFlowNet principles."""

    def __init__(
        self,
        SUT: base_SUT,
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
        self.SUT = SUT
        self.number_of_iterations = fuzzer_config.number_of_iterations
        self.total_time = fuzzer_config.total_time
        self.output_folder = fuzzer_config.output_folder
        self.resume = fuzzer_config.resume
        self.otf = fuzzer_config.otf
        self.max_norm = trainer_config.max_norm
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        # Initialize the 3 modules of the framework
        self.distiller = Distiller(distiller_config)
        self.instructor = Instructor(instructor_config)
        self.coder = Coder(coder_config)
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
        """
        Performs off-policy training steps by sampling from the instruction buffer.

        Args:
            batch_size (int): Number of samples to draw from the buffer per step.
            steps (int): Number of off-policy training steps to perform.
        """
        for _ in range(steps):
            batch = self.ibuffer.sample(batch_size)
            if not batch:
                break
            for t in batch:
                # Skip samples with non-positive reward
                if t["reward"] <= 0:
                    continue
                log_prob_sum = sum(t["f_log_probs"])
                log_z_sum = t["logZ"]
                log_reward = math.log(t["reward"])
                # Perform a training step using the sampled data
                self.instructor.train_step(
                    torch.tensor(log_z_sum, device=self.instructor.device),
                    torch.tensor(log_prob_sum, device=self.instructor.device),
                    torch.tensor(log_reward, device=self.instructor.device)
                )


    def train(self) -> None:
        """
        Run the fuzzing process.

        This method orchestrates the main fuzzing loop: generating prompts, instructions,
        code, evaluating with the oracle, updating the buffer, and performing both
        on-policy and off-policy training steps.
        """
        # Generate auto-prompt from documentation using the distiller
        self.initial_prompt = self.distiller.generate_prompt()
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
            # Resume from previous run if needed
            self.count = self.__get_resume_count()
            if self.resume and self.count > 0:
                log = f" (resuming from {self.count})"
                p.console.print(log)
            p.update(task, advance=self.count)
            # Main fuzzing loop
            while (
                self.count < self.number_of_iterations
                ) and (
                time.time() - self.start_time < self.total_time * 3600
                ):
                # Generate a sequence of instructions from the prompt
                instructions, log_probs, log_zs = self.instructor.generate_instruction_sequence(
                    self.prompt
                )
                # Generate code from the instructions using the coder
                fos = self.coder.generate_code(prompt=instructions)
                for fo in fos:
                    # Evaluate the generated code using the oracle
                    _, _, reward = self.oracle.inspect(
                        fo = fo,
                        output_folder = self.output_folder,
                        count = self.count,
                        otf = self.otf,
                    )
                    # Perform an on-policy training step
                    loss_value = self.instructor.train_step(
                        log_zs, log_probs, reward, self.max_norm
                    )
                    # Add off-policy data to buffer
                    self.ibuffer.add(
                        states=[],
                        actions=[],
                        instructions=instructions,
                        forward_logprobs=log_probs,
                        backward_logprobs=[],
                        final_reward=reward.item() if hasattr(reward, "item") else reward,
                        logZ=log_zs.item() if hasattr(log_zs, "item") else log_zs
                    )
                    # Perform an off-policy update
                    self.__train_off_policy(batch_size=2, steps=1)
                    # Save checkpoint every N steps (e.g., every 100 steps)
                    if self.count % 100 == 0:
                        self.checkpointer.save(self.count)
                    self.count += 1


    
    def evaluate_all(self) -> None:
        """
        Evaluate all generated outputs against the oracle.

        This method delegates to the SUT's validate_all method to perform
        comprehensive evaluation of all outputs.
        """
        self.SUT.validate_all()




