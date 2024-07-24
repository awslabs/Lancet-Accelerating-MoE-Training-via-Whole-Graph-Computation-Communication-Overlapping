"""Utilities."""
from time import time

from .logger import get_logger

logger = get_logger("Util")  # pylint: disable=invalid-name


def func_timer(prefix=None):
    """A decorator to time the execution time of a function."""

    def _do_timer(func):
        msg = func.__name__ if not prefix else prefix

        def _wrap_timer(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            end = time()
            logger.info("%s...%.2fs", msg, end - start)
            return result

        return _wrap_timer

    return _do_timer

def fix_seed(seed):
    """Fix the seed for reproducibility."""
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)