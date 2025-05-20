import torch
from typing import List, Tuple
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
import torch.nn as nn
from dataclasses import asdict
from trainer.utils import TrainerConfig
from instruct_LM.utils import InstructorConfig
from logger import GlobberLogger, LEVEL
import os
import time
import traceback

class InstructionSequence:
    """
    Represents a sequence of instructions.
    
    Attributes:
        initial_prompt (str): The starting prompt text.
        instructions (List[str]): List of generated instructions.
    """
    
    def __init__(self, initial_prompt: str = "", template: dict = None):
        """
        Initialize the instruction sequence with an optional initial prompt.
        
        Args:
            initial_prompt (str): The starting prompt text. Default is an empty string.
        """
        self.initial_prompt = initial_prompt  # Single string containing the initial prompt.
        self.template = template
        self.instructions = []  # List of strings, each representing an instruction.
        
    def add_instruction(self, instruction: str):
        """
        Add an instruction to the sequence.
        
        Args:
            instruction (str): Single instruction text to be added to the sequence.
        """
        self.instructions.append(instruction)  # Append the instruction to the list.
        
    def get_full_text(
        self, 
        separator: str = "\n", 
        instruct_format: str = "llama"
    ) -> str:
        """
        Get the full text of the sequence, including the prompt and all instructions.

        Args:
            separator (str): String to separate instructions, default is "\n".
            instruct_format (str): Format type: "llama", "mistral", or "plain".

        Returns:
            str: Concatenated string of the initial prompt and all instructions with separators.
        """
        
        # Compose base instruction lines
        instruction_lines = [
            f"INITIAL PROMPT: {self.initial_prompt}",
            f"TASK: {self.template['desc']}",
            f"NOTE: {self.template['note']}"
        ]
        instruction_lines.extend(f"{i} - {instr}" for i, instr in enumerate(self.instructions))
        content = separator.join(instruction_lines)
        

        if instruct_format.lower() == "llama":
            content = content + "\n" + self.template["next"]
            return (
                "<|system|>\n"
                f"{self.template['main']}\n"
                "<|user|>\n"
                f"{content}\n"
                "<|assistant|>\n"
            )
        elif instruct_format.lower() == "mistral":
            # Mistral doesn't use separate system/user tags, just [INST] ... [/INST]
            content = content + "\n" + self.template["next"]
            prompt = (
                f"{self.template['main']}\n{content}"
            )
            return f"[INST] {prompt.strip()} [/INST]"
        elif instruct_format.lower() == "plain":
            # No special formatting, just return everything together
            return separator.join([
                f"MAIN: {self.template['main']}",
                content
            ])
        else:
            raise ValueError(f"Unknown instruct_format: {instruct_format}")

    
    def __len__(self) -> int:
        """
        Get the number of instructions in the sequence.
        
        Returns:
            int: The number of instructions in the sequence.
        """
        return len(self.instructions)  # Return the length of the instructions list.
    


class Instructor:
    """Handles instruction-level generation logic and manages its own LLM and training."""

    def __init__(
            self, 
            instructor_config: InstructorConfig,
            trainer_config: TrainerConfig
        ):
        self.engine_name = instructor_config.engine_name
        self.template = asdict(instructor_config.template)
        self.separator = instructor_config.separator
        self.max_instructions = instructor_config.max_instructions
        self.temperature = instructor_config.temperature
        self.max_len = instructor_config.max_len
        self.device = instructor_config.device
        self.__setup_model_and_optimizer(trainer_config)
        self.logger = GlobberLogger("instructor.log", level=LEVEL.INFO)
        self.logger.log("Instructor initialized.", LEVEL.INFO)

    def __setup_model_and_optimizer(self, trainer_config):
        """
        Setup model, tokenizer, optimizer, scheduler, and projection layer for training.
        Args:
            trainer_config: TrainerConfig object with model/training parameters.
        """
        # Load model config (necessary for projection layer Pz)
        config = AutoConfig.from_pretrained(self.engine_name)
        config.use_cache = True
        # Load pre-trained model with full precision
        self.model = AutoModelForCausalLM.from_pretrained(
            self.engine_name,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=True,  # Enable Flash Attention 2 for better performance
            torch_dtype=torch.float16    # Use float16 for better memory efficiency while maintaining precision
        )
        # Setup LoRA configuration for Llama 3
        lora_config = LoraConfig(
            r=trainer_config.lora_r,
            lora_alpha=trainer_config.lora_alpha,
            lora_dropout=trainer_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head"
            ]
        )
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        # Add projection layer for log_z
        model_config = self.model.config
        n_dim = model_config.hidden_size
        self.model.proj_z = nn.Linear(n_dim, 1).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.engine_name,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=trainer_config.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        t_total = trainer_config.train_steps * trainer_config.grad_acc_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=trainer_config.num_warmup_steps,
            num_training_steps=t_total
        )

    def train_step(self, log_z_sum, log_prob_sum, log_reward, max_norm):
        self.logger.log(f"Training step started. log_z_sum: {log_z_sum}, log_prob_sum: {log_prob_sum}, log_reward: {log_reward}, max_norm: {max_norm}", LEVEL.TRACE)
        start_time = time.time()
        try:
            self.model.train()
            self.optimizer.zero_grad()
            loss = (log_z_sum + log_prob_sum - log_reward) ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()
            self.scheduler.step()
            end_time = time.time()
            self.logger.log(f"Loss: {loss.item()} (step duration: {end_time - start_time:.2f}s)", LEVEL.VERBOSE)
            return loss.item()
        except Exception as e:
            self.logger.log(f"Error during training step: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise

    def _avg_pooling(
            self, last_hidden: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
        """
        Average pooling of hidden states using the attention mask
        
        Args:
            last_hidden: Hidden states with shape [batch_size, seq_len, hidden_dim]
            attention_mask: Mask with shape [batch_size, seq_len]
            
        Returns:
            Pooled representation with shape [batch_size, hidden_dim]
        """
        # Motivation: Average pooling is used to summarize the hidden states
        # into a single vector representation for each sequence in the batch.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        denom = torch.clamp(input_mask_expanded.sum(1), min=1)  # Avoid division by zero
        avg_pool = torch.sum(last_hidden * input_mask_expanded, 1) / denom
        return avg_pool

    def generate_instruction(
        self, 
        prompt_text: str,
        temperature: float = 1.0,
        max_len: int = 50,
        stop_sequences: List[str] = None
    ) -> Tuple[str, float, float]:
        self.logger.log(f"generate_instruction called with prompt_text: {str(prompt_text)[:200]}, temperature: {temperature}, max_len: {max_len}, stop_sequences: {stop_sequences}", LEVEL.TRACE)
        try:
            # Tokenize with proper padding
            tokenized = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.model.device)
            prompt_ids = tokenized["input_ids"]
            prompt_attention_mask = tokenized["attention_mask"]
            outputs = self.model(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]
            avg_pool = self._avg_pooling(last_hidden, prompt_attention_mask)
            log_z = self.model.proj_z(avg_pool).squeeze(-1)
            
            gen_output = self.model.generate(
                **tokenized,
                do_sample=True,
                max_new_tokens=max_len,
                temperature=temperature,
                top_p=0.9,  # Add nucleus sampling
                top_k=50,   # Add top-k sampling
                repetition_penalty=1.1,  # Prevent repetition
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = gen_output.sequences[:, prompt_ids.shape[1]:]
            scores = gen_output.scores
            # Handle stop sequences
            stop_positions = []
            if stop_sequences:
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                for stop in stop_sequences:
                    pos = generated_text.find(stop)
                    if pos != -1:
                        stop_positions.append(pos)
            # Calculate log probabilities
            sum_logpf = torch.zeros(1, device=self.model.device)
            for i, score in enumerate(scores):
                if stop_positions and i >= min(stop_positions):
                    break
                log_prob = F.log_softmax(score, dim=-1)
                token_log_prob = torch.gather(
                    log_prob, -1, generated_ids[:, i].unsqueeze(-1)
                ).squeeze(-1)
                sum_logpf += token_log_prob[0]
            # Decode and clean up the generated text
            instruction_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if stop_positions:
                earliest_stop = min(stop_positions)
                instruction_text = instruction_text[:earliest_stop]
            # Clean up any remaining special tokens or formatting
            instruction_text = instruction_text.replace("<|assistant|>", "").replace("<|user|>", "").strip()
            self.logger.log(f"Generated instruction: {str(instruction_text)[:200]}, log_prob: {sum_logpf.item()}, log_z: {log_z.item() if hasattr(log_z, 'item') else log_z}", LEVEL.VERBOSE)
            return instruction_text, sum_logpf, log_z
            
        except Exception as e:
            self.logger.log(f"Error during instruction generation: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise
        
    def generate_instruction_sequence(
        self,
        initial_prompt: str,
    ) -> Tuple['InstructionSequence', List[float], List[float]]:
        self.logger.log(f"generate_instruction_sequence called with initial_prompt: {str(initial_prompt)[:200]}", LEVEL.TRACE)
        start_time = time.time()
        try:
            sequence = InstructionSequence(initial_prompt, self.template)
            log_probs = []
            log_zs = []
            for idx in range(self.max_instructions):
                intermediate_prompt = sequence.get_full_text(self.separator, instruct_format="llama")
                self.logger.log(f"Generating instruction {idx} with prompt: {str(intermediate_prompt)[:200]}", LEVEL.TRACE)
                instruction, log_prob, log_z = self.generate_instruction(
                    intermediate_prompt, 
                    self.temperature, 
                    self.max_len, 
                    [self.separator, self.tokenizer.eos_token]
                )
                sequence.add_instruction(instruction)
                log_probs.append(log_prob)
                log_zs.append(log_z)
                self.logger.log(f"Instruction {idx}: {str(instruction)[:200]}, log_prob: {log_prob.item() if hasattr(log_prob, 'item') else log_prob}, log_z: {log_z.item() if hasattr(log_z, 'item') else log_z}", LEVEL.VERBOSE)
            end_time = time.time()
            final_prompt = sequence.get_full_text(self.separator, instruct_format="plain")
            self.logger.log(f"Instruction sequence generation complete in {end_time - start_time:.2f}s", LEVEL.TRACE)
            return final_prompt, log_probs, log_zs
        except Exception as e:
            self.logger.log(f"Error during instruction sequence generation: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise