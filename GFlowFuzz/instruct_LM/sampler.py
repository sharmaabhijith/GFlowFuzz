from typing import List, Tuple
import torch
import torch.nn.functional as F
from GFlowFuzz.instruct_LM.instruct_utils import InstructionSequence


class InstructionSampler:
    """Handles instruction-level generation logic"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
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
        instruction_template: str,
        separator: str,
        max_instructions: int,
        temperature: float = 1.0,
        max_len: int = 50
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
        
        for _ in range(max_instructions):
            # Build prompt for next instruction
            next_prompt = sequence.get_next_prompt(instruction_template, separator)
            
            # Generate the next instruction
            instruction, log_prob, log_z = self.generate_instruction(
                next_prompt, 
                temperature, 
                max_len, 
                [separator, self.tokenizer.eos_token]
            )
            
            # Add instruction to sequence and track probabilities
            sequence.add_instruction(instruction)
            log_probs.append(log_prob)
            log_zs.append(log_z)
        
        return sequence, log_probs, log_zs