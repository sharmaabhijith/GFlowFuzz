import gzip
import heapq
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any

import editdistance
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

@dataclass
class InstructionTemplateConfig:
    main: str
    desc: str
    note: str
    next: str


@dataclass
class InstructorConfig:
    engine_name: str
    template: InstructionTemplateConfig
    separator: str
    max_instructions: int
    temperature: float
    max_len: int
    device: str

def batch_cosine_similarity_kernel(embeddings, batch_size=16):
    num_samples = embeddings.size(0)
    avg_sim = 0.0

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_end = min(i + batch_size, num_samples)
        batch = embeddings[i:batch_end, :]
        with torch.no_grad():
            cos_sim_batch = F.linear(F.normalize(
                batch), F.normalize(embeddings))
        avg_sim += cos_sim_batch.sum().item()

    # Adjust for duplicate pairs and remove diagonal components
    diag = 0.0
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch = embeddings[i:batch_end, :]
        diag += F.cosine_similarity(batch, batch, dim=-1).sum().item()
    avg_sim -= diag

    # Compute average similarity
    avg_sim /= (num_samples * (num_samples - 1))

    return avg_sim


def seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


class InfIterator(object):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)

    def __len__(self):
        return len(self.iterator)


def lora_to_base(model):
    try:
        model.base_model.disable_adapter_layers()
    except:
        print("No adapter layers to disable")
    model.eval()


def base_to_lora(model):
    try:
        model.base_model.enable_adapter_layers()
    except:
        print("No adapter layers to enable")
    model.train()




class InstructionBuffer:
    """
    Stores full trajectories intended for GFlowNet training using
    the Trajectory Balance (TB) objective.
    Each trajectory is represented by:
    - states (List[Any])
    - actions (List[Any])
    - instructions (List[str])
    - forward_logprobs (List[float])
    - backward_logprobs (List[float])
    - final_reward (float)
    - logZ (float)
    - c_reward (float)
    - lm_reward (float)
    - composite_reward (float)
    """

    def __init__(self, capacity: int = 1000, prioritization: bool = False):
        """
        Args:
            capacity: Maximum number of trajectories to store.
            prioritization: Whether to prioritize buffer replacement based on composite_reward.
        """
        self.capacity = capacity
        self.storage = []
        self.pointer = 0
        self.prioritization = prioritization

    def add(
        self,
        states: List[Any],
        actions: List[Any],
        instructions: List[str],
        forward_logprobs: List[float],
        backward_logprobs: List[float],
        final_reward: float,
        logZ: float,
        c_reward: float = 0.0,
        lm_reward: float = 0.0,
        composite_reward: float = 0.0
    ) -> None:
        """
        Add a new trajectory with instructions for off-policy training.
        """
        trajectory = {
            "states": states,
            "actions": actions,
            "instructions": instructions,
            "f_log_probs": forward_logprobs,
            "b_log_probs": backward_logprobs,
            "reward": final_reward,
            "logZ": logZ,
            "c_reward": c_reward,
            "lm_reward": lm_reward,
            "composite_reward": composite_reward
        }
        if len(self.storage) >= self.capacity:
            if self.prioritization:
                # Remove lowest composite reward
                idx = min(range(len(self.storage)), key=lambda i: self.storage[i]["composite_reward"])
                self.storage.pop(idx)
            else:
                # Remove random
                import random
                idx = random.randint(0, len(self.storage) - 1)
                self.storage.pop(idx)
            self.storage.append(trajectory)
        else:
            self.storage.append(trajectory)
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, batch_size: int) -> List[dict]:
        """
        Sample a random batch of trajectories from the buffer.
        Optional: Apply padding to handle variable trajectory lengths if needed.

        Returns:
            A list of trajectory dicts.
        """
        if len(self.storage) == 0:
            return []
        import random
        batch_size = min(batch_size, len(self.storage))
        return random.sample(self.storage, batch_size)

    def size(self) -> int:
        """Number of trajectories currently stored."""
        return len(self.storage)

    def save(self, filename: str) -> None:
        """
        Save buffer to file.
        """
        data = {"trajectories": self.storage}
        with open(filename, "w") as f:
            json.dump(data, f)

    def load(self, filename: str) -> None:
        """
        Load buffer from file.
        """
        if not os.path.exists(filename):
            return
        with open(filename, "r") as f:
            data = json.load(f)
        self.storage = data.get("trajectories", [])
