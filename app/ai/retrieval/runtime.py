import os

from app.ai.runtime import get_device as get_device


def prepare_transformers():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_JAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


prepare_transformers()
