
# ============================================================
#  🔧  Hot-patch (fix for PyTorch DTensor save bug on some wheels)
# ============================================================
try:
    from torch.distributed.tensor import DTensor          # noqa: F401
except (ImportError, AttributeError, ModuleNotFoundError):
    class DTensor:                                        # pragma: no cover
        pass
    import transformers.modeling_utils as _mu
    _mu.DTensor = DTensor                                 # monkey-patch
# ------------------------------------------------------------

import os
import shutil
import pandas as pd
from transformers import AutoTokenizer

from grpo_trainer import (
    GRPOConfig,
    GRPOTrainer,
    print_heading,            # you already export this helper
)

# ------------------------------------------------------------
#  🚀  Main entry-point
# ------------------------------------------------------------
def main() -> None:
    print_heading("GRPO Medical Dialogue Trainer")

    # ----- tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ----- config ------------------------------------------------------------
    config = GRPOConfig(
        branches        = 1,      # ONE doctor branch per turn
        runs_per_case   = 4,     # ⇠ 10 independent conversations per case
        num_turns       = 5,
        max_length      = 20,
        fp16            = False,
        # any other overrides you want … (lr, kl_coeff, etc.)
    )

    # ----- data --------------------------------------------------------------
    medical_data = pd.read_csv("training_cases.csv")
    print(medical_data.head())

    # ----- trainer -----------------------------------------------------------
    trainer   = GRPOTrainer(tokenizer, config, medical_data)
    save_path = "./saved_model_kl_0_batch4"

    # ----- training loop -----------------------------------------------------
    successful_train_steps = 0  # <-- counter for batches that actually update model
    for idx, row in enumerate(medical_data.itertuples(), start=1):
        print_heading(f"Processing case {idx}: {row.case[:50]} …")
        stats = trainer.train_on_case(row.case)

        if stats:  # ← Only increment/save if a real update happened
            successful_train_steps += 1

            # checkpoint every 1000 updates (not all rows)
            if successful_train_steps % 1000 == 0:
                print(f"💾  Saving checkpoint at {successful_train_steps} updates → {save_path}")
                _overwrite_dir(save_path)
                trainer.doctor_model.save_pretrained(save_path, safe_serialization=False)
                tokenizer.save_pretrained(save_path)

    # ----- final save --------------------------------------------------------
    print(f"✅  Finished. Saving final model → {save_path}")
    _overwrite_dir(save_path)
    trainer.doctor_model.save_pretrained(save_path, safe_serialization=False)
    tokenizer.save_pretrained(save_path)


# ------------------------------------------------------------
#  helper: clear / recreate a directory
# ------------------------------------------------------------
def _overwrite_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
