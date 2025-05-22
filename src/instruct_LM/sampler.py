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
import time
import traceback
from instruct_LM.instructor import InstructionSequence, Instructor
from coder_LM.base_coder import get_LLM_client


class Sampler:
    """Handles instruction-level generation logic and manages its own LLM and training."""

    def __init__(
            self, 
            instructor_config: InstructorConfig,
            trainer_config: TrainerConfig
        ):
        self.api_name = instructor_config.api_name
        # LLM configs
        self.llm_config = instructor_config.llm_config
        self.engine_name = self.llm_config.engine_name
        self.temperature = self.llm_config.temperature
        self.max_len = self.llm_config.max_tokens
        # Instruction config
        self.template = asdict(instructor_config.template)
        self.separator = instructor_config.separator
        self.max_instructions = instructor_config.max_instructions
        # Setup model and optimizer based on API or local
        if self.api_name!="local":
            self.model, self.tokenizer = None, None
            self.llm_client = get_LLM_client(
                api_name = self.api_name, 
                engine_name = self.engine_name
            )
        else:
            self.model, self.tokenizer = self.__setup_model_and_optimizer(trainer_config)
        # Initialize logger
        self.logger = GlobberLogger("fuzzer.log", level=LEVEL.TRACE)
        self.logger.log("Sampler initialized.", LEVEL.INFO)
        # Initialize instructor
        self.instructor = Instructor(
            api_name=self.api_name,
            model=self.model,
            tokenizer=self.tokenizer,
            llm_client=self.llm_client,
            llm_config=self.llm_config,
            device=instructor_config.device,
        )


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

    def sample_instruction_sequence(
        self,
        initial_prompt: str,
    ) -> Tuple[InstructionSequence, List[float], List[float]]:
        self.logger.log(f"generate_instruction_sequence called with initial_prompt:", LEVEL.TRACE)
        start_time = time.time()
        try:
            sequence = InstructionSequence(
                api_name=self.api_name,
                engine_name=self.engine_name,
                initial_prompt=initial_prompt,
                template=self.template
            )
            log_probs = []
            log_zs = []
            for idx in range(self.max_instructions):
                intermediate_prompt = sequence.get_full_text(self.separator)
                self.logger.log(f"Generating instruction {idx} with existing prompt", LEVEL.TRACE)
                instruction, log_prob, log_z = self.instructor.generate_instruction(intermediate_prompt)
                sequence.add_instruction(instruction)
                log_probs.append(log_prob)
                log_zs.append(log_z)
            end_time = time.time()
            final_prompt = sequence.get_full_text(self.separator, final=True)
            self.logger.log(f"Instruction sequence generation complete in {end_time - start_time:.2f}s", LEVEL.TRACE)
            return final_prompt, log_probs, log_zs

        except Exception as e:
            self.logger.log(f"Error during instruction sequence generation: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise