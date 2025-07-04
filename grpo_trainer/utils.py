import warnings
import gc
import torch
from transformers import logging

def setup_environment():
    """Configure environment settings and warnings"""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.set_verbosity_error()  # Only show errors for transformers

    # Configure torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

def cleanup_memory():
    """Clean up GPU memory and perform garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def print_heading(text: str, width: int = 50):
    """Print a formatted heading"""
    print(f"\n{'=' * width}")
    print(f"{text.center(width)}")
    print(f"{'=' * width}")