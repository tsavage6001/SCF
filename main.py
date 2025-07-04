#!/usr/bin/env python3

# === 🔧 Patch for PyTorch DTensor save issue ===
try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError, ModuleNotFoundError):
    class DTensor:
        pass
    globals()["DTensor"] = DTensor

import transformers.modeling_utils as _mu
if not hasattr(_mu, "DTensor"):
    _mu.DTensor = DTensor
# === End Patch ===

import os
import pandas as pd
from transformers import AutoTokenizer
from grpo_trainer import (
    GRPOConfig,
    GRPOTrainer,
    print_heading
)

def main():
    print_heading("GRPO Medical Dialogue Trainer")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Configuration
    config = GRPOConfig(
        branches=4,
        num_turns=5,
        max_length=20,
        fp16=False
    )

    # Load medical cases
    medical_data = pd.read_csv("training_cases.csv")
    print(medical_data.head())

    trainer = GRPOTrainer(tokenizer, config, medical_data)
    save_path = "./saved_model3"

    for i, row in enumerate(medical_data.itertuples(), 1):
        print_heading(f"Processing case: {str(row.case)[:50]}...")
        print(i)
        trainer.train_on_conversation_tree(row.case)

        # ✅ Save every 5 actual training steps, not cases
        if trainer.training_step_count > 0 and trainer.training_step_count % 1000 == 0:
            print(f"🔄 Saving checkpoint after {trainer.training_step_count} training steps to: {save_path}")
            trainer.save_model_locally(save_path)

    # Final save if not already saved at exact step
    if trainer.training_step_count % 2000 != 0:
        print(f"🔄 Final model save to: {save_path}")
        trainer.save_model_locally(save_path)

if __name__ == "__main__":
    main()
