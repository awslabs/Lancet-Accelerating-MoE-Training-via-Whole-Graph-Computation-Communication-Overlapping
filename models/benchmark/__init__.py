"""Benchmark models."""
import importlib.util

from .registry import get_model_bencher, list_model_names, reg_model

if importlib.util.find_spec("raf") is not None:
    from . import raf
    from .raf.raf_bencher import RAFBencher

if importlib.util.find_spec("torch") is not None:
    from . import pytorch
    from .pytorch.torch_bencher import TorchBencher
