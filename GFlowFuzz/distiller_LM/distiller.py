import os
from typing import List, Dict, Any, Callable, Optional, Tuple
from rich.progress import track

from GFlowFuzz.distiller_LM.utils import OpenAIConfig, DistillerConfig, request_engine
from GFlowFuzz.utils import LEVEL, Logger



class Distiller:
    """
    A class for automatically generating and evaluating prompts for code generation tasks.
    """
    
    def __init__(
        self,
        distiller_config: DistillerConfig,
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
        self.logger = distiller_config.logger
        self.wrap_prompt = distiller_config.wrap_prompt_func
        self.validate_prompt = distiller_config.validate_prompt_func
        self.prompt_components = distiller_config.prompt_components
        self.system_message = distiller_config.system_message
        self.instruction = distiller_config.instruction
        self.openai_config = distiller_config.openai_config
        self.engine_name = distiller_config.openai_config.engine_name
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
        # If we have already done auto-prompting, just return the best prompt
        if os.path.exists(self.folder + "/prompts/best_prompt.txt"):
            self.logger.logo("Use existing prompt ... ", level=LEVEL.INFO)
            with open(
                self.folder + "/prompts/best_prompt.txt", "r", encoding="utf-8"
            ) as f:
                return f.read()
        self.logger.logo("Use auto-prompting prompt ... ", level=LEVEL.INFO)
        # First run with temperature 0.0 to get the greedy prompt
        config = self.openai_config(
            {},
            self._create_auto_prompt_message(message),
            max_tokens=max_tokens,
            temperature=0.0,
            engine_name=self.engine_name,
        )
        response = request_engine(config)
        greedy_prompt = self.wrap_prompt(response.choices[0].message.content)
        with open(
            self.folder + "/prompts/greedy_prompt.txt", "w", encoding="utf-8"
        ) as f:
            f.write(greedy_prompt)   
        # Evaluate the greedy prompt
        best_prompt, best_score = greedy_prompt, self.validate_prompt(greedy_prompt)
        with open(self.folder + "/prompts/scores.txt", "a") as f:
            f.write(f"greedy score: {str(best_score)}")
        # Generate additional prompt samples with temperature 1.0
        for i in track(range(num_samples), description="Generating prompts..."):
            config = self.openai_config(
                {},
                self._create_auto_prompt_message(message),
                max_tokens=max_tokens,
                temperature=1.0,
                engine_name=self.engine_name,
            )
            response = request_engine(config)
            prompt = self.wrap_prompt(response.choices[0].message.content)
            with open(
                self.folder + "/prompts/prompt_{}.txt".format(i),
                "w", encoding="utf-8"
            ) as f:
                f.write(prompt)   
            # Evaluate this prompt
            score = self.validate_prompt(prompt)
            if score > best_score:
                best_score = score
                best_prompt = prompt    
            # Record the score
            with open(self.folder + "/prompts/scores.txt", "a") as f:
                f.write(f"\n{i} prompt score: {str(score)}")
        # Save the best prompt
        with open(self.folder + "/prompts/best_prompt.txt", "w", encoding="utf-8") as f:
            f.write(best_prompt)

        return best_prompt