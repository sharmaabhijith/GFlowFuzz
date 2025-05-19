import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from typing import Optional

class CheckpointManager:
    def __init__(self, save_dir: str, exp_name: str, model, optimizer, scheduler, ibuffer):
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ibuffer = ibuffer

    def load(self) -> int:
        output_dir = os.path.join(self.save_dir, self.exp_name)

        if not os.path.exists(output_dir):
            return 1

        dirs = sorted(os.listdir(output_dir))
        if len(dirs) == 0:
            return 1

        dirs = [int(x) for x in dirs if x.isdigit()]
        dirs = sorted(dirs, reverse=True)
        ckpt_dir = os.path.join(output_dir, str(dirs[0]))

        # Load model
        _model = AutoModelForCausalLM.from_pretrained(self.config.sft_ckpt)
        _model = PeftModel.from_pretrained(_model, ckpt_dir)
        msg = self.model.load_state_dict(_model.state_dict(), strict=False)
        print(msg)

        # Load optimizer, scheduler, and projection layer
        ckpt = torch.load(os.path.join(ckpt_dir, "ckpt.pt"))
        self.model.proj_z.load_state_dict(ckpt["proj_z"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        # Load buffer
        buffer_path = os.path.join(ckpt_dir, "instruction_buffer.json")
        if os.path.exists(buffer_path):
            self.ibuffer.load(buffer_path)

        return ckpt["global_step"] + 1

    def save(self, step: int) -> None:
        output_dir = os.path.join(self.save_dir, f"{self.exp_name}/{step}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save optimizer, scheduler and projection layer
        ckpt = {
            "global_step": step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "proj_z": self.model.proj_z.state_dict()
        }
        ckpt_file = os.path.join(output_dir, "ckpt.pt")
        torch.save(ckpt, ckpt_file)

        # Save buffer
        buffer_path = os.path.join(output_dir, "instruction_buffer.json")
        self.ibuffer.save(buffer_path)