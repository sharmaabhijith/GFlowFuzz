from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainerConfig:
    """Configuration for the Fuzzing Trainer"""
    # General arguments
    exp_name: str
    save_dir: str
    wandb_project: str
    prompt_file: str
    few_shot_file: str
    
    # Model arguments
    model_name: str
    sft_ckpt: str
    victim_model: str
    dtype: str
    gpu_memory_utilization: float
    
    # Training arguments
    batch_size: int
    eval_batch_size: int
    train_steps: int
    grad_acc_steps: int
    lr: float
    max_norm: float
    num_warmup_steps: int
    eval_period: int
    
    # LoRA arguments
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    
    # Sampling arguments
    temp_low: float
    temp_high: float
    max_len: int  # Max length of each instruction
    max_instructions: int = 5  # Maximum number of instructions in a sequence
    victim_max_len: int
    min_len: int
    victim_temp: float
    victim_top_p: float
    num_r_samples: int
    
    # Buffer arguments
    metric: str  # "edit" or "cosine"
    buffer_size: int
    prioritization: bool
    compare: str
    
    # Reward arguments
    beta: float
    reward_sched_start: float
    reward_sched_end: float
    reward_sched_horizon: int
    lm_sched_start: float
    lm_sched_end: float
    lm_sched_horizon: int
    
    # Instruction template
    instruction_template: str = "Generate the next instruction:"
    instruction_separator: str = "\n\n"
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        # Motivation: This method allows easy conversion of the configuration object
        # into a dictionary format, which is useful for logging or serialization.
        return {k: v for k, v in self.__dict__.items()}