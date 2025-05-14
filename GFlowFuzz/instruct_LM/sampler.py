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
from .utils import InstructorConfig
from trainer import TrainerConfig

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
    


class Instructor:
    """Handles instruction-level generation logic and manages its own LLM and training."""

    def __init__(self, instructor_config: InstructorConfig):
        self.engine_name = instructor_config.engine_name
        self.tokenizer = instructor_config.tokenizer
        self.instruction_template = instructor_config.instruction_template
        self.separator = instructor_config.separator
        self.max_instructions = instructor_config.max_instructions
        self.temperature = instructor_config.temperature
        self.max_len = instructor_config.max_len
        # Model, optimizer, scheduler, device, and projection layer will be set up later
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None

    def setup_model_and_optimizer(self, trainer_config: TrainerConfig):
        """
        Setup model, tokenizer, optimizer, scheduler, and projection layer for training.
        Args:
            trainer_config: TrainerConfig object with model/training parameters.
        """
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model config
        config = AutoConfig.from_pretrained(trainer_config.model_name)
        config.use_cache = True
        # Load pre-trained model and apply LoRA
        self.model = AutoModelForCausalLM.from_pretrained(
            trainer_config.sft_ckpt,
            config=config,
            device_map=None
        ).to(self.device)
        lora_config = LoraConfig(
            r=trainer_config.lora_r,
            lora_alpha=trainer_config.lora_alpha,
            lora_dropout=trainer_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        # Add projection layer for log_z
        model_config = self.model.config
        if "gpt2" in trainer_config.model_name:
            n_dim = model_config.n_embd
        else:
            n_dim = model_config.hidden_size
        self.model.proj_z = nn.Linear(n_dim, 1).to(self.device)
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            trainer_config.sft_ckpt, padding_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=trainer_config.lr)
        t_total = trainer_config.train_steps * trainer_config.grad_acc_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, trainer_config.num_warmup_steps, t_total
        )

    def train_step(self, log_z_sum, log_prob_sum, log_reward, max_norm):
        """
        Performs a single training step (forward + backward) using the provided computations.
        Args:
            log_z_sum: torch.Tensor
            log_prob_sum: torch.Tensor
            log_reward: torch.Tensor
            max_norm: float, gradient clipping norm
        Returns:
            float: The computed loss value for this step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = (log_z_sum + log_prob_sum - log_reward) ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

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
        """
        Generate a single instruction
        
        Args:
            prompt_text: Text prompt to generate from
            temperature: Sampling temperature
            max_len: Maximum length of generated instruction
            stop_sequences: List of strings that signal end of instruction
            
        Returns:
            Tuple of (generated_instruction, log_prob, log_z)
        """
        # Motivation: This method generates a single instruction based on the given prompt.
        # It also calculates the log probability and log Z for the generated instruction.
        
        # Tokenized input dimensions:
        # prompt_ids: [batch_size, seq_len]
        # prompt_attention_mask: [batch_size, seq_len]

        # Hidden state dimensions:
        # last_hidden: [batch_size, seq_len, hidden_dim]
        # avg_pool: [batch_size, hidden_dim]

        # Generated output dimensions:
        # generated_ids: [batch_size, generated_seq_len]
        # scores: List of tensors, each with shape [batch_size, vocab_size]
        
        # Tokenize prompt
        tokenized = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_ids = tokenized["input_ids"]
        prompt_attention_mask = tokenized["attention_mask"]
        
        outputs = self.model(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True
        )
        
        # Calculate log_z
        last_hidden = outputs.hidden_states[-1]
        avg_pool = self._avg_pooling(last_hidden, prompt_attention_mask)
        log_z = self.model.proj_z(avg_pool).squeeze(-1)
        
        # Generate instruction
        gen_output = self.model.generate(
            **tokenized,
            do_sample=True,
            max_new_tokens=max_len,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Extract generated tokens
        generated_ids = gen_output.sequences[:, prompt_ids.shape[1]:]
        scores = gen_output.scores
        
        # Find stop token positions if any
        stop_positions = []
        if stop_sequences:
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            for stop in stop_sequences:
                pos = generated_text.find(stop)
                if pos != -1:
                    stop_positions.append(pos)
        
        # Calculate log probability
        sum_logpf = torch.zeros(1, device=self.model.device)
        for i, score in enumerate(scores):
            if stop_positions and i >= min(stop_positions):
                break
            log_prob = F.log_softmax(score, dim=-1)
            token_log_prob = torch.gather(
                log_prob, -1, generated_ids[:, i].unsqueeze(-1)
            ).squeeze(-1)
            sum_logpf += token_log_prob[0]
        
        # Extract the instruction text, trimming at the earliest stop sequence
        instruction_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if stop_positions:
            earliest_stop = min(stop_positions)
            instruction_text = instruction_text[:earliest_stop]
        
        return instruction_text, sum_logpf, log_z
        
    def generate_instruction_sequence(
        self,
        initial_prompt: str,
    ) -> Tuple[InstructionSequence, List[float], List[float]]:
        """
        Generate a sequence of instructions
        
        Args:
            initial_prompt: Starting prompt
            instruction_template: Template for requesting next instruction
            separator: String to separate instructions
            num_instructions: Number of instructions to generate
            temperature: Sampling temperature
            max_len: Maximum length of each instruction
            
        Returns:
            Tuple of (InstructionSequence, list of log_probs, list of log_zs)
        """
        sequence = InstructionSequence(initial_prompt)
        log_probs = []
        log_zs = []
        
        for _ in range(self.max_instructions):
            # Build prompt for next instruction
            next_prompt = sequence.get_next_prompt(
                self.instruction_template, self.separator
            )
            
            # Generate the next instruction
            instruction, log_prob, log_z = self.generate_instruction(
                next_prompt, 
                self.temperature, 
                self.max_len, 
                [self.separator, self.tokenizer.eos_token]
            )
            
            # Add instruction to sequence and track probabilities
            sequence.add_instruction(instruction)
            log_probs.append(log_prob)
            log_zs.append(log_z)
        
        return sequence, log_probs, log_zs