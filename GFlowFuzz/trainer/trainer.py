"""
Instruction-level Fuzzing Trainer using GFlowNet principles.
This module handles instruction sequence generation and evaluation
instead of token-by-token generation.
"""

import logging
import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import wandb
from csv_logger import CsvLogger
from dataset import get_dataloader
from peft import LoraConfig, PeftModel, get_peft_model
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from vllm import LLM, SamplingParams

from GFlowFuzz.instruct_LM.utils import (
    InfIterator, 
    batch_cosine_similarity_kernel, 
    formatted_dict,
)
from GFlowFuzz.instruct_LM.utils import TrainerConfig, InstructionBuffer, InstructionEvaluator
from sampler import InstructionSampler


class Trainer:
    """
    Trainer for generating instruction sequences that maximize reward.
    Uses GFlowNet principles at the instruction sequence level.
    """
    
    def __init__(self, config: TrainerConfig) -> None:
        """
        Initialize the trainer with configuration
        
        Args:
            config: Trainer configuration
        """
        self.config = config
        self._setup_device_and_logging()
        self._setup_model_and_optimizer()
        self._setup_instruction_buffer()
        self._setup_modules()
        
        self.start_step = self._load_checkpoint()
        
    def _setup_device_and_logging(self) -> None:
        """Setup device and logging"""
        self.device = torch.cuda.current_device()
        
        # Initialize wandb
        wandb.init(
            reinit=True, 
            config=self.config.as_dict(),
            project=self.config.wandb_project, 
            name=self.config.exp_name
        )
        
        # Initialize CSV logger
        delimiter = ","
        self.csvlogger = CsvLogger(
            filename=f"logs/{self.config.exp_name}_fuzz.csv",
            delimiter=delimiter,
            level=logging.INFO,
            add_level_nums=None,
            fmt=f'%(asctime)s{delimiter}%(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            header=["date", "sequence", "c_log_reward", "lm_log_reward"]
        )
        
    def _setup_model_and_optimizer(self) -> None:
        """Setup model, tokenizer, optimizer and scheduler"""
        # Motivation: This method initializes the model, tokenizer, optimizer,
        # and learning rate scheduler for training.
        
        # Model dimensions:
        # n_dim: Hidden size of the model, used for the projection layer
        # proj_z: Linear layer with input size [hidden_dim] and output size [1]
        
        # Load model configuration
        config = AutoConfig.from_pretrained(self.config.model_name)
        config.use_cache = True
        
        # Load pre-trained model and apply LoRA
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.sft_ckpt,
            config=config,
            device_map=self.device
        )
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Add projection layer for log_z
        model_config = self.model.config
        if "gpt2" in self.config.model_name:
            n_dim = model_config.n_embd
        else:
            n_dim = model_config.hidden_size
            
        self.model.proj_z = nn.Linear(n_dim, 1).to(self.device)
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.sft_ckpt, padding_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        t_total = self.config.train_steps * self.config.grad_acc_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, self.config.num_warmup_steps, t_total
        )
        
        
    def _setup_instruction_buffer(self) -> None:
        """Initialize instruction buffer"""
        self.ibuffer = InstructionBuffer(
            max_size=self.config.buffer_size,
            prioritization=self.config.prioritization
        )
            
    def _setup_modules(self) -> None:
        """Setup functional modules"""
        self.instruction_sampler = InstructionSampler(self.model, self.tokenizer)
            
    def _load_checkpoint(self) -> int:
        """
        Load checkpoint if available
        
        Returns:
            Starting step number
        """
        output_dir = os.path.join(self.config.save_dir, self.config.exp_name)
        
        if not os.path.exists(output_dir):
            return 1
            
        dirs = sorted(os.listdir(output_dir))
        if len(dirs) == 0:
            return 1
        
        # Find most recent checkpoint
        dirs = [int(x) for x in dirs if x.isdigit()]
        dirs = sorted(dirs, reverse=True)
        ckpt_dir = os.path.join(output_dir, str(dirs[0]))
        
        # Load model
        _model = AutoModelForCausalLM.from_pretrained(self.config.sft_ckpt)
        _model = PeftModel.from_pretrained(_model, ckpt_dir)
        msg = self.model.load_state_dict(_model.state_dict(), strict=False)
        print(msg)
        
        # Load optimizer, scheduler, and projection layer
        ckpt = torch.load(os.path.join(ckpt_dir, "ckpt.pt"))
        self.model.proj_z.load_state_dict(ckpt["proj_z"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        
        # Load buffer
        buffer_path = os.path.join(ckpt_dir, "instruction_buffer.json")
        if os.path.exists(buffer_path):
            self.ibuffer.load(buffer_path)
        
        return ckpt["global_step"] + 1
    
    def _save_checkpoint(self, step: int) -> None:
        """
        Save checkpoint
        
        Args:
            step: Current training step
        """
        output_dir = os.path.join(self.config.save_dir, f"{self.config.exp_name}/{step}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save optimizer, scheduler and projection layer
        ckpt = {
            "global_step": step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "proj_z": self.model.proj_z.state_dict()
        }
        ckpt_file = os.path.join(output_dir, "ckpt.pt")
        torch.save(ckpt, ckpt_file)
        
        # Save buffer
        buffer_path = os.path.join(output_dir, "instruction_buffer.json")
        self.ibuffer.save(buffer_path)
    
        
    def _compute_tb_loss(self, log_z_sum: torch.Tensor, log_prob_sum: torch.Tensor, 
                        log_reward: torch.Tensor) -> torch.Tensor:
        """
        Compute Trajectory Balance loss for instruction sequences
        
        Args:
            log_z_sum: Sum of log Z values for the sequence
            log_prob_sum: Sum of log probabilities for the sequence
            log_reward: Log reward for the sequence
            
        Returns:
            Loss tensor
        """
        # Motivation: The Trajectory Balance loss ensures that the generated
        # sequences align with the reward distribution.
        # Dimensions:
        # log_z_sum: [1]
        # log_prob_sum: [1]
        # log_reward: [1]
        delta = log_z_sum + log_prob_sum - log_reward
        return delta**2


    def train_step(self, log_z_sum: torch.Tensor, log_prob_sum: torch.Tensor, log_reward: torch.Tensor) -> float:
        """
        Performs a single training step (forward + backward) using the provided computations.
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._compute_tb_loss(log_z_sum, log_prob_sum, log_reward)  # Reuse the TB loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm) 
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def train_off_policy(self, batch_size: int = 4, steps: int = 1):
        """
        Performs off-policy training steps by sampling from the instruction buffer.
        """
        for _ in range(steps):
            batch = self.ibuffer.sample(batch_size)
            if not batch:
                break
            for t in batch:
                if t["reward"] <= 0:
                    continue
                log_prob_sum = sum(t["f_log_probs"])
                log_z_sum = t["logZ"]
                log_reward = math.log(t["reward"])
                self.train_step(
                    torch.tensor(log_z_sum, device=self.device),
                    torch.tensor(log_prob_sum, device=self.device),
                    torch.tensor(log_reward, device=self.device)
                )
