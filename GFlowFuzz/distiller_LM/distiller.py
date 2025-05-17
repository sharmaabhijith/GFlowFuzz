import os
from typing import List, Dict, Optional
from rich.progress import track
import time
import traceback
from GFlowFuzz.distiller_LM.utils import DistillerConfig
from GFlowFuzz.logger import GlobberLogger, LEVEL
from GFlowFuzz.coder_LM import BaseCoder
from GFlowFuzz.SUT import BaseSUT
from GFlowFuzz.client_LLM import FactoryLLMClient


class Distiller:
    """
    A class for automatically generating and evaluating prompts for code generation tasks.
    """
    
    def __init__(
        self,
        distiller_config: DistillerConfig,
        coder: BaseCoder,
        SUT: BaseSUT,
    ) -> None:
        """
        Initialize the AutoPrompter.
        
        Args:
            folder: Directory path for saving prompts
            logger: Logger instance for output messages
            wrap_prompt_func: Function to wrap prompts in appropriate format
            validate_prompt_func: Function to validate and score prompts
            prompt_components: Dictionary containing prompt components like 'separator', 'begin'
            system_message: System message for auto-prompting
            instruction: Instruction for auto-prompting
        """
        self.folder = distiller_config.folder
        self.logger = GlobberLogger("distiller.log", level=LEVEL.INFO)
        self.logger.log("Distiller initialized.", LEVEL.INFO)
        self.system_message = distiller_config.system_message
        self.instruction = distiller_config.instruction
        self.llm_config = distiller_config.llm_config
        self.engine_name = distiller_config.llm_config.engine_name
        self.coder = coder
        self.SUT = SUT
        # Create prompts directory if it doesn't exist
        os.makedirs(self.folder + "/prompts", exist_ok=True)
    
    def _create_auto_prompt_message(self, message: str) -> List[Dict[str, str]]:
        """
        Create the messages for auto-prompting.
        
        Args:
            message: User message/content to create a prompt from
            
        Returns:
            List of message dictionaries for API request
        """
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": message + "\n" + self.instruction},
        ]
    
    def generate_prompt(
        self,
        message: Optional[str] = None, 
        num_samples: int = 3,
        max_tokens: int = 500
    ) -> str:
        """
        Generate the best prompt based on different strategies and validate them.
        
        Args:
            message: Documentation or content to base the prompt on
            hw_prompt: Hand-written prompt (if using one)
            hw: Whether to use a hand-written prompt
            no_input_prompt: Whether to use a minimal prompt
            engine_name: LLM engine_name to use for auto-prompting
            num_samples: Number of prompt samples to generate
            max_tokens: Maximum tokens for generation
            
        Returns:
            Best prompt based on validation score
        """
        self.logger.log(f"generate_prompt called with message: {str(message)[:200]}, num_samples: {num_samples}, max_tokens: {max_tokens}", LEVEL.TRACE)
        start_time = time.time()
        try:
            if os.path.exists(self.folder + "/prompts/best_prompt.txt"):
                self.logger.log("Use existing prompt ... ", level=LEVEL.INFO)
                with open(
                    self.folder + "/prompts/best_prompt.txt", "r", encoding="utf-8"
                ) as f:
                    best_prompt = f.read()
                self.logger.log(f"Loaded best_prompt.txt: {str(best_prompt)[:300]}", LEVEL.VERBOSE)
                return best_prompt
            self.logger.log("Use auto-prompting prompt ... ", level=LEVEL.INFO)
            config = self.llm_config(
                self._create_auto_prompt_message(message),
                max_tokens=max_tokens,
                temperature=0.0,
                engine_name=self.engine_name,
            )
            self.logger.log(f"OpenAI config for greedy prompt: {str(config)[:300]}", LEVEL.TRACE)
            response = request_engine(config)
            greedy_prompt = self.SUT.wrap_prompt(response.content)
            self.logger.log(f"Greedy prompt: {str(greedy_prompt)[:300]}", LEVEL.VERBOSE)
            with open(
                self.folder + "/prompts/greedy_prompt.txt", "w", encoding="utf-8"
            ) as f:
                f.write(greedy_prompt)   
            best_prompt, best_score = greedy_prompt, self.SUT.validate_prompt(
                greedy_prompt, 
                self.coder
            )
            self.logger.log(f"Greedy prompt score: {best_score}", LEVEL.TRACE)
            with open(self.folder + "/prompts/scores.txt", "a") as f:
                f.write(f"greedy score: {str(best_score)}")
            for i in track(range(num_samples), description="Generating prompts..."):
                self.logger.log(f"Generating sample prompt {i}", LEVEL.TRACE)
                config = self.llm_config(
                    self._create_auto_prompt_message(message),
                    max_tokens=max_tokens,
                    temperature=1.0,
                    engine_name=self.engine_name,
                )
                self.logger.log(f"OpenAI config for sample {i}: {str(config)[:300]}", LEVEL.TRACE)
                response = request_engine(config)
                prompt = self.SUT.wrap_prompt(response.content)
                self.logger.log(f"Sample {i} prompt: {str(prompt)[:300]}", LEVEL.VERBOSE)
                with open(
                    self.folder + "/prompts/prompt_{}.txt".format(i),
                    "w", encoding="utf-8"
                ) as f:
                    f.write(prompt)   
                score = self.SUT.validate_prompt(prompt, self.coder)
                self.logger.log(f"Sample {i} score: {score}", LEVEL.TRACE)
                if score > best_score:
                    best_score = score
                    best_prompt = prompt    
                    self.logger.log(f"Sample {i} is new best prompt.", LEVEL.INFO)
                with open(self.folder + "/prompts/scores.txt", "a") as f:
                    f.write(f"\n{i} prompt score: {str(score)}")
            with open(self.folder + "/prompts/best_prompt.txt", "w", encoding="utf-8") as f:
                f.write(best_prompt)
            end_time = time.time()
            self.logger.log(f"Prompt generation complete in {end_time - start_time:.2f}s. Best score: {best_score}", LEVEL.INFO)
            self.logger.log(f"Best prompt: {str(best_prompt)[:300]}", LEVEL.VERBOSE)
            return best_prompt
        except Exception as e:
            self.logger.log(f"Error during prompt generation: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise