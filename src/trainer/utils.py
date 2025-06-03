from dataclasses import dataclass
from typing import Dict, Any, List
import os
import time
import pandas as pd
from logger import GlobberLogger, LEVEL


def write_to_file(file_name: str, content: str | List[str]):
    if isinstance(content, str):
        with open(file_name, "w") as f:
            f.write(content)
    elif isinstance(content, list):
        for c in content:
            with open(file_name, "w") as f:
                f.write(c)

class CompilationRecorder:
    """Class to handle compilation record tracking and CSV file management."""
    
    def __init__(self, output_folder: str, logger: GlobberLogger):
        """
        Initialize the compilation recorder.
        
        Args:
            output_folder: Directory where the CSV file will be saved
            logger: Logger instance for logging updates
        """
        self.logger = logger
        self.compilation_records = pd.DataFrame(columns=[
            'iteration',
            'error',
            'coverage',
            'reward',
            'timestamp'
        ])
        self.compilation_csv_path = os.path.join(output_folder, "compilation_records.csv")
        self.logger.log(f"CompilationRecorder initialized. CSV will be saved at {self.compilation_csv_path}", LEVEL.TRACE)

    def update_record(
        self,
        iteration: int,
        error: str,
        coverage: str,
        reward: float
    ) -> None:
        """
        Update the compilation records with new iteration data and save to CSV.
        
        Args:
            iteration: Current iteration number
            error: Error message or status
            coverage: Coverage information
            reward: Reward value for this iteration
        """
        new_record = pd.DataFrame([{
            'iteration': iteration,
            'error': error,
            'coverage': coverage,
            'reward': reward,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }])
        
        self.compilation_records = pd.concat([self.compilation_records, new_record], ignore_index=True)
        self.compilation_records.to_csv(self.compilation_csv_path, index=False)
        self.logger.log(f"Updated compilation records CSV at {self.compilation_csv_path}", LEVEL.TRACE)

@dataclass
class FuzzerConfig:
    number_of_iterations: int
    total_time: int
    log_level: int
    resume: bool = False
    otf: bool = False

@dataclass
class TrainerConfig:
    """Configuration for the Fuzzing Trainer"""
    device: str
    # Model arguments
    batch_size: int
    sft_ckpt: str
    # Training arguments
    train_steps: int
    grad_acc_steps: int
    lr: float
    max_norm: float
    num_warmup_steps: int
    # LoRA arguments
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    # Buffer arguments
    buffer_size: int
    prioritization: bool
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        # Motivation: This method allows easy conversion of the configuration object
        # into a dictionary format, which is useful for logging or serialization.
        return {k: v for k, v in self.__dict__.items()}
    

