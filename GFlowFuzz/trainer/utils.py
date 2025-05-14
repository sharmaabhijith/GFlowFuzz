from dataclasses import dataclass
from typing import Dict, Any
from SUT import SUTConfig

@dataclass
class FuzzerConfig:
    SUT: SUTConfig
    number_of_iterations: int
    total_time: int
    output_folder: str
    resume: bool = False
    otf: bool = False
    log_level: int

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
    

