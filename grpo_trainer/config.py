from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class GRPOConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    learning_rate: float = 2e-7
    batch_size: int = 1
    max_length: int = 20
    num_turns: int = 5
    device: Optional[str] = None
    kl_coeff: float = 0
    branches: int = 4
    temperature: float = 0.8
    top_p: float = 0.85
    gradient_checkpointing: bool = True
    fp16: bool = False

    device: str = "cuda"  # Auto-selects GPUs
    multi_gpu: bool = True  # Enable multi-GPU training
    gradient_accumulation_steps: int = 2  # Better memory utilization

    def __post_init__(self):
        """Validate configuration and set default device"""
        if self.device is None:
            self.device = "cuda" # if torch.cuda.is_available() else "cpu"
            # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.branches < 1:
            raise ValueError("branches must be at least 1")
        if not 0 < self.temperature <= 1.0:
            raise ValueError("temperature must be between 0 and 1")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")