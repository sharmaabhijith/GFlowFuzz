from typing import List
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
)
from coder_LM.utils import EndOfFunctionCriteria
from GFlowFuzz.logger import GlobberLogger, LEVEL
import time
import traceback
from utils import CoderConfig
from client_LLM import get_client
from .__init__ import BaseCoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable warning
EOF_STRINGS = ["<|endoftext|>", "###"]



class BaseCoderLocal(BaseCoder):
    def __init__(
        self,
        coder_config: CoderConfig, 
    ):
        self.device = coder_config.device
        self.tokenizer = AutoTokenizer.from_pretrained(coder_config.engine_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            coder_config.engine_name,
            torch_dtype=torch.bfloat16,
            device=self.device,
        )
        self.eos = EOF_STRINGS + coder_config.eos
        self.max_length = coder_config.max_length
        self.skip_special_tokens = False
        self.logger = GlobberLogger("coder.log", level=LEVEL.INFO)
        self.logger.log("BaseCoderLocal initialized.", LEVEL.INFO)

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


class BaseCoderAPI(BaseCoder):
    def __init__(self, coder_config: CoderConfig):
        self.llm_client = get_client(coder_config.llm_config.provider, model=coder_config.llm_config.model)
        self.config = coder_config
        self.logger = GlobberLogger("coder_api.log", level=LEVEL.INFO)
        self.logger.log("BaseCoderAPI initialized.", LEVEL.INFO)

    def format_prompt(self, prompt: str) -> str:
        return prompt

    def generate_code(self, prompt: str, **kwargs) -> List[str]:
        self.logger.log(f"API code generation started. prompt: {str(prompt)[:200]}", LEVEL.TRACE)
        config = {
            "model": self.config.llm_config.model,
            "messages": [{"role": "user", "content": self.format_prompt(prompt)}],
        }
        config.update(kwargs)
        response = self.llm_client.request(config)
        # For OpenAI, response.choices[0].message.content; for DeepInfra, response.json()[...]
        if hasattr(response, 'choices'):
            return [response.choices[0].message.content]
        elif hasattr(response, 'json'):
            data = response.json()
            return [data["choices"][0]["message"]["content"]]
        else:
            return [str(response)]