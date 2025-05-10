"""Main script to run the fuzzing process."""

import os
import time
from typing import List, Tuple, Optional, Any

from rich.traceback import install
install()

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from GFlowFuzz.distiller_LM import Distiller, DistillerConfig
from GFlowFuzz.instruct_LM import Instructor
from GFlowFuzz.coder_LM import Coder
from GFlowFuzz.SUT.base_sut import base_SUT
from GFlowFuzz.evaluator import Oracle


class Fuzzer:
    """Class for managing the fuzzing process."""

    def __init__(
        self,
        SUT: base_SUT,
        number_of_iterations: int,
        distiller_config: DistillerConfig,
        instructor_config, # TODO: ADD DATATYPE
        coder_config, # TODO: ADD DATATYPE
        total_time: int,
        output_folder: str,
        resume: bool = False,
        otf: bool = False,
    ):
        """
        Initialize the fuzzer with SUT and configuration parameters.
        
        Args:
            SUT: The SUT to fuzz
            number_of_iterations: Maximum number of fuzzing iterations
            total_time: Maximum fuzzing time in hours
            output_folder: Where to store outputs
            resume: Whether to resume from previous runs
            otf: Whether to validate on the fly
        """
        self.SUT = SUT
        self.number_of_iterations = number_of_iterations
        self.total_time = total_time  # in hours
        self.output_folder = output_folder
        self.resume = resume
        self.otf = otf
        self.count = 0
        self.start_time = 0
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        # initialize the 3 modules of the framework
        self.distiller = Distiller(distiller_config)
        self.instructor = Instructor(instructor_config)
        self.coder = Coder(coder_config)
        self.oracle = Oracle(self.SUT)
    
    def __get_resume_count(self) -> int:
        """
        Get the next count for resuming a fuzzing run.
        
        Returns:
            The next count for file naming
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

    def run(self) -> None:
        """Run the fuzzing process."""
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
                prev = []
                for fo in fos:
                    f_result, content = self.oracle.check(
                        fo = fo,
                        output_folder = self.output_folder,
                        count = self.count,
                        otf = self.otf,
                    )
                    log_reward = self.oracle
                    loss = self.oracle.compute_tb_loss(
                        log_z_sum=log_zs,
                        log_prob_sum=log_probs,
                        log_reward=log_reward,
                    )
                    loss.backward()
    
    def evaluate_all(self) -> None:
        """Evaluate all generated outputs against the oracle."""
        self.SUT.validate_all()



