import os
from functools import lru_cache


def prepare_transformers():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_JAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


@lru_cache(maxsize=1)
def get_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


prepare_transformers()
