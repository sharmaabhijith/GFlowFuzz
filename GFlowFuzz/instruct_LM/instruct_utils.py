import os
import json
import random
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from GFlowFuzz.instruct_LM.general_utils import lora_to_base, base_to_lora

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
    
    
class InstructionSequence:
    """
    Represents a sequence of instructions.
    
    Attributes:
        initial_prompt (str): The starting prompt text.
        instructions (List[str]): List of generated instructions.
    """
    
    def __init__(self, initial_prompt: str = ""):
        """
        Initialize the instruction sequence with an optional initial prompt.
        
        Args:
            initial_prompt (str): The starting prompt text. Default is an empty string.
        """
        self.initial_prompt = initial_prompt  # Single string containing the initial prompt.
        self.instructions = []  # List of strings, each representing an instruction.
        
    def add_instruction(self, instruction: str):
        """
        Add an instruction to the sequence.
        
        Args:
            instruction (str): Single instruction text to be added to the sequence.
        """
        self.instructions.append(instruction)  # Append the instruction to the list.
        
    def get_full_text(self, template: str = "", separator: str = "\n") -> str:
        """
        Get the full text of the sequence, including the prompt and all instructions.
        
        Args:
            template (str): Optional template text (not used in current implementation).
            separator (str): String to separate instructions, default is "\n".
            
        Returns:
            str: Concatenated string of the initial prompt and all instructions with separators.
        """
        # Add two newline gap between main prompt and rest of the instructions.
        text = self.initial_prompt + separator # Start with the initial prompt.
        test += "Follow the instructions below closely to generate the code" + separator
        for i, instruction in enumerate(self.instructions):
            text += separator + f"{str(i)} - " + instruction # Append the instruction.
        return text  # Return the full concatenated text.
    
    def get_next_prompt(self, template: str, separator: str = "\n") -> str:
        """
        Get the text to prompt for the next instruction.
        
        Args:
            template (str): Template text to append after the current sequence.
            separator (str): String to separate instructions, default is "\n".
            
        Returns:
            str: Full text with the template appended, ready for the next instruction generation.
        """
        text = self.get_full_text(separator=separator)  # Get the full text of the sequence so far.
        if template:
            text += separator + template  # Append the template with a separator.
        return text  # Return the next prompt text.
    
    def __len__(self) -> int:
        """
        Get the number of instructions in the sequence.
        
        Returns:
            int: The number of instructions in the sequence.
        """
        return len(self.instructions)  # Return the length of the instructions list.
    

class InstructionBuffer:
    """Buffer for storing instruction sequences"""
    
    def __init__(self, max_size, prioritization=False):
        self.max_size = max_size
        self.prioritization = prioritization
        self.sequences = []
        self.lm_rewards = []
        self.c_rewards = []
        self.composite_rewards = []
        self.log_probs = []
        self.log_zs = []
        
    def add(self, sequence, lm_reward, c_reward, composite_reward, log_probs, log_zs):
        """Add a sequence to the buffer"""
        # Motivation: The buffer stores instruction sequences along with their rewards
        # and log probabilities for future sampling and training.
        if len(self.sequences) >= self.max_size:
            # Remove based on priority
            if self.prioritization:
                # Remove lowest reward
                idx = self.composite_rewards.index(min(self.composite_rewards))
                self.sequences.pop(idx)
                self.lm_rewards.pop(idx)
                self.c_rewards.pop(idx)
                self.composite_rewards.pop(idx)
                self.log_probs.pop(idx)
                self.log_zs.pop(idx)
            else:
                # Remove random
                idx = random.randint(0, len(self.sequences) - 1)
                self.sequences.pop(idx)
                self.lm_rewards.pop(idx)
                self.c_rewards.pop(idx)
                self.composite_rewards.pop(idx)
                self.log_probs.pop(idx)
                self.log_zs.pop(idx)
                
        self.sequences.append(sequence)
        self.lm_rewards.append(lm_reward)
        self.c_rewards.append(c_reward)
        self.composite_rewards.append(composite_reward)
        self.log_probs.append(log_probs)
        self.log_zs.append(log_zs)
        
    def sample(self, batch_size):
        """Sample batch_size sequences from buffer"""
        # Motivation: Sampling from the buffer allows reusing past sequences
        # to improve training efficiency and stability.
        if len(self.sequences) == 0:
            return [], [], [], [], [], []
            
        indices = random.sample(range(len(self.sequences)), 
                               min(batch_size, len(self.sequences)))
        
        sampled_sequences = [self.sequences[i] for i in indices]
        sampled_lm_rewards = [self.lm_rewards[i] for i in indices]
        sampled_c_rewards = [self.c_rewards[i] for i in indices]
        sampled_composite_rewards = [self.composite_rewards[i] for i in indices]
        sampled_log_probs = [self.log_probs[i] for i in indices]
        sampled_log_zs = [self.log_zs[i] for i in indices]
        
        return (sampled_sequences, sampled_lm_rewards, sampled_c_rewards, 
                sampled_composite_rewards, sampled_log_probs, sampled_log_zs)
    
    def size(self):
        """Return current buffer size"""
        return len(self.sequences)
    
    def save(self, filename):
        """Save buffer to file"""
        data = {
            "sequences": [(seq.initial_prompt, seq.instructions) for seq in self.sequences],
            "lm_rewards": self.lm_rewards,
            "c_rewards": self.c_rewards,
            "composite_rewards": self.composite_rewards,
            "log_probs": self.log_probs,
            "log_zs": self.log_zs
        }
        
        with open(filename, "w") as f:
            json.dump(data, f)
    
    def load(self, filename):
        """Load buffer from file"""
        if not os.path.exists(filename):
            return
            
        with open(filename, "r") as f:
            data = json.load(f)
        
        self.sequences = []
        for prompt, instructions in data["sequences"]:
            seq = InstructionSequence(prompt)
            for inst in instructions:
                seq.add_instruction(inst)
            self.sequences.append(seq)
            
        self.lm_rewards = data["lm_rewards"]
        self.c_rewards = data["c_rewards"]
        self.composite_rewards = data["composite_rewards"]
        self.log_probs = data["log_probs"]
        self.log_zs = data["log_zs"]