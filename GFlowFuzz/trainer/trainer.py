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
    
    @staticmethod
    def _make_prompt(instruction: str) -> str:
        """
        Format prompt for GPT or Dolly models
        
        Args:
            instruction: User instruction
            
        Returns:
            Formatted prompt
        """
        prompt_template = (
            "Below is an instruction that describes a task. Write a response that appropriately "
            "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        )
        return prompt_template.format(instruction=instruction.rstrip())
    
        
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
    
    def _simulate_instruction_experience(self, initial_prompt: str, max_instructions: int, 
                                        temperature: float) -> Dict[str, Any]:
        """
        Generate an instruction sequence and evaluate it
        
        Args:
            initial_prompt: Starting prompt
            max_instructions: Maximum number of instructions
            temperature: Sampling temperature
            
        Returns:
            Dictionary with results
        """
        # Randomly decide how many instructions to generate (at least 1)
        num_instructions = random.randint(1, max_instructions)
        
        # Generate instruction sequence
        sequence, log_probs, log_zs = self.instruction_sampler.generate_instruction_sequence(
            initial_prompt,
            self.config.instruction_template,
            self.config.instruction_separator,
            num_instructions,
            temperature,
            self.config.max_len
        )
        
        # Evaluate the sequence
        lm_reward, c_reward = self.instruction_evaluator.compute_instruction_reward(
            sequence, self.prompt_fn
        )
        
        # Calculate log probability sum and log z sum
        log_prob_sum = sum(log_probs)
        log_z_sum = sum(log_zs)
        
        return {
            "sequence": sequence,
            "lm_reward": lm_reward,
            "c_reward": c_reward,
            "log_prob_sum": log_prob_sum,
            "log_z_sum": log_z_sum,
        }

    def _get_batch_metrics(self, prompt_batch: List[str], step: int, 
                         max_instructions: int, beta: float, train: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process a batch and get training metrics
        
        Args:
            prompt_batch: List of initial prompts
            step: Current step
            max_instructions: Maximum number of instructions per sequence
            beta: Weighting factor for toxicity reward
            train: Whether in training or evaluation mode
            
        Returns:
            Tuple of loss and metrics dictionary
        """
        metrics = {}
        train_test = 'train' if train else 'eval'
        
        all_losses = []
        all_c_rewards = []
        all_lm_rewards = []
        all_log_rewards = []
        all_sequences = []
        
        # Process each prompt in the batch
        for prompt in prompt_batch:
            # Decide whether to sample from buffer or generate new
            if self.ibuffer.size() > 0 and random.random() < 0.5:
                # Sample from buffer
                sequences, lm_rewards, c_rewards, composite_rewards, log_probs, log_zs = self.ibuffer.sample(1)
                
                if sequences:  # Check if sampling returned anything
                    sequence = sequences[0]
                    lm_reward = torch.tensor(lm_rewards[0], device=self.device)
                    c_reward = torch.tensor(c_rewards[0], device=self.device)
                    log_prob_sum = log_probs[0]
                    log_z_sum = log_zs[0]
                else:
                    # Generate new if buffer is empty
                    temp = random.uniform(self.config.temp_low, self.config.temp_high)
                    results = self._simulate_instruction_experience(prompt, max_instructions, temp)
                    sequence = results["sequence"]
                    lm_reward = results["lm_reward"]
                    c_reward = results["c_reward"]
                    log_prob_sum = results["log_prob_sum"]
                    log_z_sum = results["log_z_sum"]
            else:
                # Generate new instruction sequence
                temp = random.uniform(self.config.temp_low, self.config.temp_high)
                results = self._simulate_instruction_experience(prompt, max_instructions, temp)
                sequence = results["sequence"]
                lm_reward = results["lm_reward"]
                c_reward = results["c_reward"]
                log_prob_sum = results["log_prob_sum"]
                log_z_sum = results["log_z_sum"]
            
            # Calculate reward with temperature
            gamma = self._get_temperature(step, mode='lm')
            log_reward = (lm_reward / gamma) + (c_reward / beta)
            
            rew_temp = self._get_temperature(step, mode='total')
            tempered_log_reward = log_reward / rew_temp
            
            # Add to buffer
            self.ibuffer.add(
                sequence, lm_reward.item(), c_reward.item(), log_reward.item(),
                log_prob_sum.item(), log_z_sum.item()
            )
            
            # Log to CSV
            if train and random.random() < 0.1:  # Only log 10% of examples to avoid clutter
                self.csvlogger.info([
                    f'"{sequence.get_full_text()}"',
                    c_reward.item(),
                    lm_reward.item()
                ])
            
            # Compute loss
            loss = self._compute_tb_loss(log_z_sum, log_prob_sum, tempered_log_reward)
            
            all_losses.append(loss)
            all_c_rewards.append(c_reward.item())
            all_lm_rewards.append(lm_reward.item())
            all_log_rewards.append(log_reward.item())
            all_sequences.append(sequence.get_full_text())
        
        # Combine losses
        combined_loss = torch.stack(all_losses).mean()
        
        # Collect metrics
        metrics[f"log_z_sum"] = log_z_sum.item()
        metrics[f"c_log_reward/{train_test}"] = sum(all_c_rewards) / len(all_c_rewards)
        metrics[f"lm_log_reward/{train_test}"] = sum(all_lm_rewards) / len(all_lm_rewards)
        metrics[f"log_reward/{train_test}"] = sum(all_log_rewards) / len(all_log_rewards)
        metrics[f"loss/{train_test}"] = combined_loss.item()
        metrics[f"avg_sequence_length"] = sum(len(s.split()) for s in all_sequences) / len(all_sequences)
        
        return combined_loss, metrics
    
    def train(self) -> None:
        """Run training"""
        # Get initial batch for training
        batch = next(self.train_iter)
        prompts = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        
        # Training loop
        t = tqdm(range(self.start_step, self.config.train_steps+1), 
                 desc="training", dynamic_ncols=True)
                 
        for global_step in t:
            batch_metrics = defaultdict(list)
            
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            # Process batch in gradient accumulation steps
            for _ in range(self.config.grad_acc_steps):
                loss, metrics = self._get_batch_metrics(
                    prompts, global_step, 
                    self.config.max_instructions, self.config.beta
                )
                
                # Collect metrics
                for k, v in metrics.items():
                    if isinstance(v, list):
                        batch_metrics[k].extend(v)
                    else:
                        batch_metrics[k].append(v)
                    
                # Backward pass
                loss = loss / self.config.grad_acc_steps
                loss.backward()
            
            # Apply gradient clipping and update
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            # Log metrics
            batch_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
            wandb.log(batch_metrics, step=global_step)
            
            # Update progress bar
            t.set_description(f"Step {global_step}: {formatted_dict(batch_metrics)}")
            
            # Save checkpoint periodically
            if global_step % self.config.eval_period == 0:
                self._save_checkpoint(global_step)
                
                # Run evaluation
                eval_metrics = self.evaluate()
                wandb.log(eval_metrics, step=global_step)
        
        # Save final checkpoint
        output_dir = os.path.join(self.config.save_dir, self.config.exp_name, "latest")
        self._save_checkpoint(global_step)
        
        # Run final evaluation
        eval_metrics = self.evaluate()
        wandb.log(eval_metrics, step=global_step)
        wandb.finish()