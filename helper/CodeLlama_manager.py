#!/usr/bin/env python3
"""
CodeLlamaManager - A unified class for training and inference with CodeLlama models
Usage examples:

# Training:
manager = CodeLlamaManager(
    model_name="codellama/CodeLlama-13b-Python-hf",
    output_dir="coder-lora"
)
manager.train(
    data_path="datasets/bug_report.jsonl",
    batch_size=1,
    accumulation_steps=8,
    max_steps=3000
)

# Inference with merged model:
manager = CodeLlamaManager(model_path="coder-lora-merged", use_merged=True)
manager.load_for_inference()
response = manager.generate("Write a Python function to reverse a string")

# Inference with LoRA adapter:
manager = CodeLlamaManager(
    model_name="codellama/CodeLlama-13b-Python-hf",
    model_path="coder-lora"
)
manager.load_for_inference()
response = manager.generate("Write a Python function to reverse a string")
"""

import os
import torch
from datasets import disable_progress_bar, load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from typing import Optional, Dict, Any
import json


class CodeLlamaManager:
    """
    Unified class for training and inference with CodeLlama models using LoRA.
    """
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-13b-Python-hf",
        model_path: Optional[str] = None,
        output_dir: str = "coder-lora",
        use_merged: bool = False,
        use_4bit: bool = True
    ):
        """
        Initialize the CodeLlamaManager.
        
        Args:
            model_name: Base model name from HuggingFace
            model_path: Path to trained model/adapter (for inference)
            output_dir: Output directory for training
            use_merged: Whether to use merged model for inference
            use_4bit: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.model_path = model_path
        self.output_dir = output_dir
        self.use_merged = use_merged
        self.use_4bit = use_4bit
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """Get BitsAndBytes configuration for 4-bit quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    def _load_base_model(self, for_training: bool = False) -> None:
        """Load the base model with appropriate configuration."""
        print(f"Loading base model: {self.model_name}")
        
        if self.use_4bit and (for_training or not self.use_merged):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self._get_bnb_config(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _build_datasets(self, data_path: str) -> Dict[str, Any]:
        """Build training and validation datasets from JSONL file."""
        print(f"Loading dataset from {data_path}")
        
        # Load the dataset
        dataset = load_dataset('json', data_files=data_path, split='train')
        
        # Split into train/test
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        return split_dataset
    
    def _tokenize_function(self, examples):
        """Tokenization function for the dataset."""
        if isinstance(examples, dict) and "coder_prompt" in examples:
            # Single example
            text = examples["coder_prompt"] + examples["coder_target"]
        else:
            # Batch of examples
            text = [prompt + target for prompt, target in 
                   zip(examples["coder_prompt"], examples["coder_target"])]
        
        return self.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
    
    def train(
        self,
        data_path: str,
        batch_size: int = 1,
        accumulation_steps: int = 8,
        max_steps: int = 3000,
        learning_rate: float = 2e-4,
        eval_steps: int = 500,
        save_steps: int = 500
    ) -> None:
        """
        Train the model with LoRA.
        
        Args:
            data_path: Path to training data (JSONL format)
            batch_size: Training batch size
            accumulation_steps: Gradient accumulation steps
            max_steps: Maximum training steps
            learning_rate: Learning rate
            eval_steps: Steps between evaluations
            save_steps: Steps between saves
        """
        print("Starting training process...")
        torch.cuda.empty_cache()
        
        # Load base model
        self._load_base_model(for_training=True)
        
        # Attach LoRA
        print("Attaching LoRA adapter...")
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        # Prepare datasets
        split_dataset = self._build_datasets(data_path)
        train_ds = split_dataset["train"].map(
            self._tokenize_function, 
            remove_columns=split_dataset["train"].column_names,
            batched=True
        )
        val_ds = split_dataset["test"].map(
            self._tokenize_function,
            remove_columns=split_dataset["test"].column_names,
            batched=True
        )
        disable_progress_bar()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            max_steps=max_steps,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            report_to="tensorboard",
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        # Train
        print("Starting training...")
        self.trainer.train()
        
        # Save LoRA adapter
        print(f"Saving LoRA adapter to {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Create merged model
        self._create_merged_model()
    
    def _create_merged_model(self) -> None:
        """Create and save merged model without quantization."""
        print("Creating merged model without quantization...")
        merged_dir = f"{self.output_dir}-merged"
        
        # Clear GPU memory
        del self.model
        torch.cuda.empty_cache()
        
        # Load base model WITHOUT quantization for merging
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Load and merge the trained LoRA adapter
        model_with_lora = PeftModel.from_pretrained(base_model, self.output_dir)
        merged_model = model_with_lora.merge_and_unload()
        
        # Save the clean merged model
        merged_model.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        self.tokenizer.save_pretrained(merged_dir)
        print(f"Clean merged model saved to {merged_dir}")
        
        # Clean up
        del base_model, model_with_lora, merged_model
        torch.cuda.empty_cache()
    
    def load_for_inference(self) -> None:
        """Load model for inference."""
        if self.use_merged:
            self._load_merged_model()
        else:
            self._load_model_with_adapter()
        
        print("Model loaded successfully for inference!")
    
    def _load_merged_model(self) -> None:
        """Load merged model for inference."""
        if not self.model_path:
            raise ValueError("model_path must be specified for merged model inference")
        
        print(f"Loading merged model from {self.model_path}")
        
        if self.use_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=self._get_bnb_config(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_model_with_adapter(self) -> None:
        """Load base model with LoRA adapter for inference."""
        if not self.model_path:
            raise ValueError("model_path must be specified for adapter inference")
        
        print(f"Loading base model with LoRA adapter from {self.model_path}")
        
        # Load base model
        if self.use_4bit:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self._get_bnb_config(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response from the model given a prompt.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_for_inference() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens (excluding input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
    
    def interactive_mode(self) -> None:
        """Start interactive inference mode."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_for_inference() first.")
        
        print("Starting interactive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nEnter your prompt: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                response = self.generate(prompt)
                print(f"\nResponse:\n{response}")
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")


# Usage functions
def train_coder_model():
    """Example: Train a coder model"""
    manager = CodeLlamaManager(
        model_name="codellama/CodeLlama-13b-Python-hf",
        output_dir="coder-lora"
    )
    manager.train(
        data_path="datasets/bug_report.jsonl",
        batch_size=1,
        accumulation_steps=8,
        max_steps=3000
    )

def train_instruct_model():
    """Example: Train an instruct model"""
    manager = CodeLlamaManager(
        model_name="codellama/CodeLlama-13b-Instruct-hf",
        output_dir="instruct-lora"
    )
    manager.train(
        data_path="datasets/bug_report.jsonl",
        batch_size=1,
        accumulation_steps=8,
        max_steps=3000
    )

def inference_with_merged():
    """Example: Inference with merged model"""
    manager = CodeLlamaManager(
        model_path="coder-lora-merged",
        use_merged=True,
        use_4bit=False
    )
    manager.load_for_inference()
    
    prompt = """
    ### API: torch.nn
    ### Bug Description: Fails on large input tensors with sparse gradients.
    ### Instructions: Generate minimal PyTorch code to trigger this issue using torch.nn.Linear with sparse gradients.
    ### Code:
    """
    
    response = manager.generate(prompt)
    print(response)

def inference_with_adapter():
    """Example: Inference with LoRA adapter"""
    manager = CodeLlamaManager(
        model_name="codellama/CodeLlama-13b-Python-hf",
        model_path="coder-lora",
        use_4bit=True
    )
    manager.load_for_inference()
    
    # Start interactive mode
    manager.interactive_mode()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CodeLlama Manager")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--model_type", choices=["coder", "instruct"], default="coder")
    parser.add_argument("--model_path", help="Path to model for inference")
    parser.add_argument("--data_path", default="datasets/bug_report.jsonl")
    parser.add_argument("--use_merged", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.model_type == "coder":
            train_coder_model()
        else:
            train_instruct_model()
    else:
        if args.use_merged:
            manager = CodeLlamaManager(
                model_path=args.model_path,
                use_merged=True
            )
        else:
            base_model = ("codellama/CodeLlama-13b-Python-hf" if args.model_type == "coder" 
                         else "codellama/CodeLlama-13b-Instruct-hf")
            manager = CodeLlamaManager(
                model_name=base_model,
                model_path=args.model_path
            )
        
        manager.load_for_inference()
        
        if args.interactive:
            manager.interactive_mode()
        else:
            # Single inference example
            prompt = """
            ### API: torch.nn
            ### Bug Description: Fails on large input tensors with sparse gradients.
            ### Instructions: Generate minimal PyTorch code to trigger this issue using torch.nn.Linear with sparse gradients.
            ### Code:
            """
            response = manager.generate(prompt)
            print(response)