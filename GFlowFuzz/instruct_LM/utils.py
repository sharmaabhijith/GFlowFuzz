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
class InstructorConfig:
    model: Any
    tokenizer: Any
    instruction_template: str
    separator: str
    max_instructions: int
    temperature: float
    max_len: int


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


@dataclass(order=True)
class TrajectoryWithReward:
    response_ids: list = field(compare=False)
    c_log_reward: float = field(compare=False)
    lm_log_reward: float = field(compare=False)
    log_reward: float = field(compare=True)  # sorting based on this
    decoded_response: str = field(compare=False)
    emb: torch.tensor = field(compare=False)
    ref_reward: float = field(compare=False, init=False)

    def __post_init__(self):
        self.ref_reward = self.log_reward


@dataclass(order=True)
class TrajectoryWithCReward:
    response_ids: list = field(compare=False)
    c_log_reward: float = field(compare=True)  # sorting based on this
    lm_log_reward: float = field(compare=False)
    log_reward: float = field(compare=False)
    decoded_response: str = field(compare=False)
    emb: torch.tensor = field(compare=False)
    ref_reward: float = field(compare=False, init=False)

    def __post_init__(self):
        self.ref_reward = self.c_log_reward


class ReplayBuffer(object):
    def __init__(self,  eos_token_id, max_size=1000, sim_tolerance=0.25, prioritization="c_reward", compare="reward"):
        self.eos_token_id = eos_token_id
        self.max_size = max_size
        self.sim_tolerance = sim_tolerance
        self.buffer = []
        self.response_pool = set()
        self.prioritization = prioritization
        self.compare = compare

        if compare == "c_reward":
            print("comparison with c_reward")
            self.Trajectory = TrajectoryWithCReward
        else:
            print("comparison with total reward")
            self.Trajectory = TrajectoryWithReward

    def size(self):
        return len(self.buffer)

    def add(self, item):
        # check whether the item has been already added before.
        if item.decoded_response in self.response_pool:
            return
        tokens = [x for x in item.response_ids.tolist() if x !=
                  self.eos_token_id]
        # find examples that are similar to the item and replace it with new one if new one has higher reward
        for buffer_item in self.buffer:
            existing_tokens = [
                x for x in buffer_item.response_ids.tolist() if x != self.eos_token_id]
            if editdistance.eval(tokens, existing_tokens) < (len(tokens) + len(existing_tokens)) * self.sim_tolerance:
                if buffer_item.ref_reward >= item.ref_reward:
                    return
                else:
                    # remove the old item
                    self.response_pool.discard(buffer_item.decoded_response)
                    self.buffer.remove(buffer_item)
                    heapq.heapify(self.buffer)

                    # add new item
                    self.response_pool.add(item.decoded_response)
                    heapq.heappush(self.buffer, item)

                    if len(self.buffer) != len(self.response_pool):
                        self.response_pool = set(
                            [x.decoded_response for x in self.buffer])
                    return

        self.response_pool.add(item.decoded_response)

        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, item)
        else:
            popped = heapq.heappushpop(self.buffer, item)
            try:
                self.response_pool.remove(popped.decoded_response)
            except KeyError:
                self.response_pool = set(
                    [x.decoded_response for x in self.buffer])

    def add_batch(self, responses, decoded_responses, res_embs, c_log_rewards, lm_log_rewards, log_rewards):
        # move tensors to cpu
        responses = responses.cpu()
        res_embs = res_embs.cpu()

        pad_mask = (responses == self.eos_token_id).cumsum(1) > 1
        response_lengths = torch.sum((~pad_mask).long(), 1)

        for i in range(log_rewards.size(0)):
            response_len = response_lengths[i].item()
            # responses is padded with right-side
            response_id = responses[i, :response_len]

            c_log_reward = c_log_rewards[i].item()
            lm_log_reward = lm_log_rewards[i].item()
            log_reward = log_rewards[i].item()

            decoded_response = decoded_responses[i]
            emb = res_embs[i]
            # add new item
            item = self.Trajectory(
                response_id,
                c_log_reward,
                lm_log_reward,
                log_reward,
                decoded_response,
                emb)

            self.add(item)

    def sample(self, num_samples):
        if self.prioritization == "reward":
            priorities = [item.log_reward for item in self.buffer]
            priorities = np.array(priorities)
            priorities = priorities - np.max(priorities)
            priorities = np.exp(priorities)
            prob = priorities / np.sum(priorities)

        elif self.prioritization == "c_reward":
            priorities = [item.c_log_reward for item in self.buffer]
            priorities = np.array(priorities)
            priorities = priorities - np.max(priorities)
            priorities = np.exp(priorities)
            prob = priorities / np.sum(priorities)

        elif self.prioritization == "uniform":
            prob = np.ones(len(self.buffer)) / len(self.buffer)

        idx = np.random.choice(
            len(self.buffer), num_samples, p=prob, replace=False)

        # right-side padding
        response_ids = [self.buffer[i].response_ids for i in idx]
        response_mask = [torch.ones_like(x) for x in response_ids]

        response_ids = pad_sequence(
            response_ids, batch_first=True, padding_value=self.eos_token_id)
        response_mask = pad_sequence(
            response_mask, batch_first=True, padding_value=0)

        response_batch = {"input_ids": response_ids,
                          "attention_mask": response_mask}

        c_log_rewards = torch.tensor(
            [self.buffer[i].c_log_reward for i in idx])
        lm_log_rewards = torch.tensor(
            [self.buffer[i].lm_log_reward for i in idx])
        log_rewards = torch.tensor([self.buffer[i].log_reward for i in idx])

        reward_batch = {"c_log_reward": c_log_rewards,
                        "lm_log_reward": lm_log_rewards,
                        "log_reward": log_rewards}

        return response_batch, reward_batch

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with gzip.open(path, "rb") as f:
            self.buffer = pickle.load(f)
        heapq.heapify(self.buffer)


class CosineRelayBuffer(ReplayBuffer):
    def __init__(self, eos_token_id, max_size=1000, sim_tolerance=0.4, prioritization="c_reward", compare="reward"):
        super().__init__(eos_token_id, max_size, sim_tolerance, prioritization, compare)

    def add(self, item):
        # check whether the item has been already added before.
        if item.decoded_response in self.response_pool:
            return

        if len(self.buffer) > 0:
            buffer_embs = torch.stack(
                [item.emb for item in self.buffer], dim=0)  # [b,d]
            # find examples that are similar to the item and replace it with new one if new one has higher reward
            query = item.emb.unsqueeze(0)  # [1,d]
            cos_sims = F.cosine_similarity(query, buffer_embs, dim=1)
            max_id = torch.argmax(cos_sims, dim=0)
            max_sim = cos_sims[max_id].item()

            if max_sim > self.sim_tolerance:
                buffer_item = self.buffer[max_id]
                if buffer_item.ref_reward >= item.ref_reward:
                    return
                else:
                    self.response_pool.discard(buffer_item.decoded_response)
                    self.buffer.remove(buffer_item)
                    heapq.heapify(self.buffer)

                    # add new item
                    self.response_pool.add(item.decoded_response)
                    heapq.heappush(self.buffer, item)

                    if len(self.buffer) != len(self.response_pool):
                        self.response_pool = set(
                            [x.decoded_response for x in self.buffer])
                    return

        self.response_pool.add(item.decoded_response)

        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, item)
        else:
            popped = heapq.heappushpop(self.buffer, item)
            try:
                self.response_pool.remove(popped.decoded_response)
            except KeyError:
                self.response_pool = set(
                    [x.decoded_response for x in self.buffer])
