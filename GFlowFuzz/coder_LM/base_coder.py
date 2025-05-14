import os
from typing import List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
)
from coder_LM.utils import EndOfFunctionCriteria, CoderConfig
from GFlowFuzz.logger import GlobberLogger, LEVEL
import time
import traceback


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable warning
EOF_STRINGS = ["<|endoftext|>", "###"]


class BaseCoder:
    def __init__(
        self,
        coder_config: CoderConfig, 
    ):
        self.device = coder_config.device
        self.tokenizer = AutoTokenizer.from_pretrained(coder_config.coder_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            coder_config.coder_name,
            torch_dtype=torch.bfloat16,
        ).to(coder_config.device)
        self.eos = EOF_STRINGS + coder_config.eos
        self.max_length = coder_config.max_length
        self.skip_special_tokens = False
        self.logger = GlobberLogger("coder.log", level=LEVEL.INFO)
        self.logger.log("BaseCoder initialized.", LEVEL.INFO)

    def format_prompt(self, prompt: str) -> str:
        """To be implemented by subclasses if prompt needs special formatting."""
        return prompt

    @torch.inference_mode()
    def generate_code(
        self,
        prompt: str,
        batch_size: int = 10,
        temperature: float = 1.0,
        max_length: int = 512,
    ) -> List[str]:
        self.logger.log(f"Code generation started. prompt: {str(prompt)[:200]}, batch_size: {batch_size}, temperature: {temperature}, max_length: {max_length}", LEVEL.TRACE)
        start_time = time.time()
        try:
            formatted_prompt = self.format_prompt(prompt)
            self.logger.log(f"Formatted prompt: {str(formatted_prompt)[:200]}", LEVEL.VERBOSE)
            input_tokens = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            start_length = input_tokens["input_ids"].shape[1]

            stopping = StoppingCriteriaList([
                EndOfFunctionCriteria(
                    start_length=start_length,
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ])

            raw_outputs = self.model.generate(
                input_tokens["input_ids"],
                max_length=min(self.max_length, start_length + max_length),
                do_sample=True,
                top_p=1.0,
                temperature=max(temperature, 1e-2),
                num_return_sequences=batch_size,
                stopping_criteria=stopping,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            gen_seqs = raw_outputs.sequences[:, start_length:]
            gen_strs = self.tokenizer.batch_decode(gen_seqs, skip_special_tokens=self.skip_special_tokens)

            outputs = []
            for output in gen_strs:
                min_index = min((output.find(eos) for eos in self.eos if eos in output), default=len(output))
                outputs.append(output[:min_index])
            self.logger.log(f"Generated {len(outputs)} code samples. First sample: {str(outputs[0])[:200] if outputs else 'None'}", LEVEL.VERBOSE)
            end_time = time.time()
            self.logger.log(f"Code generation completed in {end_time - start_time:.2f}s", LEVEL.TRACE)
            return outputs
        except Exception as e:
            self.logger.log(f"Error during code generation: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise