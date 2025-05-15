from GFlowFuzz.SUT.base_sut import FResult, BaseSUT
from GFlowFuzz.logger import GlobberLogger, LEVEL
import os
import time
import traceback

class Inspector:
    def __init__(self, sut: BaseSUT):
        self.sut = sut
        self.logger = GlobberLogger("inspector.log", level=LEVEL.INFO)
        self.logger.log("Inspector initialized.", LEVEL.INFO)

    def inspect(self, fo: str, output_folder: str, count: int, otf: bool):
        """
        Handle a single fuzzing result, writing it to file and validating if needed.
        
        Args:
            fo: The fuzzing output
            
        Returns:
            Tuple of (validation_result, message, reward) if otf is True, else (None, None, None)
        """
        self.logger.log(f"inspect called with fo: {str(fo)[:200]}, output_folder: {output_folder}, count: {count}, otf: {otf}", LEVEL.TRACE)
        start_time = time.time()
        try:
            if not otf:
                self.logger.log("OTF is False, skipping inspection.", LEVEL.TRACE)
                return None, None, None
            file_name = os.path.join(output_folder, f"{count}.fuzz")
            self.logger.log(f"Inspecting file: {file_name}", LEVEL.TRACE)
            f_result, message, reward = self.sut.validate_individual(file_name)
            self.logger.log(f"Validation result: {f_result}, message: {str(message)[:200]}, reward: {reward}", LEVEL.VERBOSE)
            self.sut.parse_validation_message(f_result, message, file_name)
            end_time = time.time()
            self.logger.log(f"Inspection result: {f_result}, reward: {reward} (duration: {end_time - start_time:.2f}s)", LEVEL.VERBOSE)
            return f_result, fo, reward
        except Exception as e:
            self.logger.log(f"Error during inspection: {e}\n{traceback.format_exc()}", LEVEL.INFO)
            raise
    
    @staticmethod
    def compute_tb_loss(log_z_sum, log_prob_sum, log_reward):
        """
        Compute Trajectory Balance loss for instruction sequences
        
        Args:
            log_z_sum: Sum of log Z values for the sequence
            log_prob_sum: Sum of log probabilities for the sequence
            log_reward: Log reward for the sequence
            
        Returns:
            Loss tensor
        """
        delta = log_z_sum + log_prob_sum - log_reward
        return delta**2