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

from GFlowFuzz.SUT.base_sut import base_SUT


class Fuzzer:
    """Class for managing the fuzzing process."""

    def __init__(
        self,
        SUT: base_SUT,
        number_of_iterations: int,
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

    def write_to_file(self, content: str, file_name: str) -> None:
        """
        Write content to a file.
        
        Args:
            content: Content to write
            file_name: File path to write to
        """
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Error writing to file {file_name}: {e}")
    
    def get_resume_count(self) -> int:
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

    def should_continue(self) -> bool:
        """
        Check if fuzzing should continue based on iteration count and time.
        
        Returns:
            True if fuzzing should continue, False otherwise
        """
        if self.count >= self.number_of_iterations:
            return False
            
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.total_time * 3600:  # Convert hours to seconds
            return False
            
        return True

    def handle_result(self, fo: str) -> Tuple[Optional[bool], Optional[str]]:
        """
        Handle a single fuzzing result, writing it to file and validating if needed.
        
        Args:
            fo: The fuzzing output
            
        Returns:
            Tuple of (validation_result, message) if otf is True, else (None, None)
        """
        file_name = os.path.join(self.output_folder, f"{self.count}.fuzz")
        self.write_to_file(fo, file_name)
        self.count += 1
        
        if self.otf:
            f_result, message = self.SUT.validate_individual(file_name)
            self.SUT.parse_validation_message(f_result, message, file_name)
            return f_result, fo
        
        return None, None

    def run(self) -> None:
        """Run the fuzzing process."""
        self.SUT.initialize()
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
            self.count = self.get_resume_count()
            if self.resume and self.count > 0:
                log = f" (resuming from {self.count})"
                p.console.print(log)
            
            p.update(task, advance=self.count)
            
            # Main fuzzing loop
            while self.should_continue():
                fos = self.SUT.generate()
                if not fos:
                    self.SUT.initialize()
                    continue
                
                prev = []
                for fo in fos:
                    f_result, content = self.handle_result(fo)
                    p.update(task, advance=1)
                    
                    if self.otf and f_result is not None:
                        prev.append((f_result, content))
                
                if prev:
                    self.SUT.update(prev=prev)
                else:
                    self.SUT.update()
    
    def evaluate_all(self) -> None:
        """Evaluate all generated outputs against the oracle."""
        self.SUT.validate_all()



