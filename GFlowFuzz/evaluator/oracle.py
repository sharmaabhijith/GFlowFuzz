from GFlowFuzz.SUT.base_sut import FResult, base_SUT
import os

class Oracle:
    def __init__(self, sut: base_SUT):
        self.sut = sut

    def check(self, code: str):
        file_path = self.sut.write_back_file(code)
        result, message = self.sut.validate_individual(file_path)
        return result, message

    def compute_tb_loss(self, log_z_sum, log_prob_sum, log_reward):
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

    def verify(self, fo: str, output_folder: str, count: int, otf: bool):
        """
        Handle a single fuzzing result, writing it to file and validating if needed.
        
        Args:
            fo: The fuzzing output
            
        Returns:
            Tuple of (validation_result, message) if otf is True, else (None, None)
        """
        if not otf:
            return None, None
        file_name = os.path.join(output_folder, f"{count}.fuzz")
        f_result, message = self.sut.validate_individual(file_name)
        self.sut.parse_validation_message(f_result, message, file_name)
        return f_result, fo