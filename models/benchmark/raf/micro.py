"""Register models for microbenchmark from PyTorch."""
# pylint: disable=missing-function-docstring, missing-class-docstring, unused-argument
# pylint: disable=invalid-name, too-many-locals, attribute-defined-outside-init

import raf
from raf.testing import one_hot_torch, randn_torch
from raf.model.nn import GELU

from .raf_bencher import RAFBencher
from ..logger import get_logger
from ..registry import reg_model

logger = get_logger("MNM-MicroBenchmark")

bert_batch_matmul_workloads = [
    [128, 3072, 768],
    [128, 2304, 768],
    [128, 768, 768],
    [128, 768, 3072],
    [128, 768, 2304]
]

bert_matmul_tn_scaled_add_workloads = [
    [768, 768, 2048],
    [3072, 768, 2048],
    [768, 3072, 2048],
]

@reg_model("raf")
def batch_matmul_add_per_channel(batch_size, shape, dtype, include_orig_model=False):
    m, n, k = shape
    m_b, _ = randn_torch((n,), dtype=dtype)
    m_w, _ = randn_torch((1, n, k), dtype=dtype)

    class MNMModel(raf.Model):
        def build(self):
            self.w = m_w
            self.b = m_b

        @raf.model.trace
        def forward(self, x):
            x = raf.batch_matmul_nt(x, self.w)
            x = raf.add(x, self.b)
            return x

    m_model = MNMModel()
    m_model.infer_mode()
    input_shape = (batch_size, m, k)
    m_x, _ = randn_torch(input_shape, dtype=dtype)
    m_dy, _ = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    m_ytrue, _ = one_hot_torch(batch_size=batch_size, num_classes=10)
    return RAFBencher(m_model, input_shape, [m_x], m_dy, m_ytrue, ref_bencher=None)


@reg_model("raf")
def batch_matmul_add_per_channel_gelu(batch_size, shape, dtype, include_orig_model=False):
    m, n, k = shape
    m_b, _ = randn_torch((n,), dtype=dtype)
    m_w, _ = randn_torch((1, n, k), dtype=dtype)

    class MNMModel(raf.Model):
        def build(self):
            self.w = m_w
            self.b = m_b
            self.gelu = GELU()

        @raf.model.trace
        def forward(self, x):
            x = raf.batch_matmul_nt(x, self.w)
            x = raf.add(x, self.b)
            x = self.gelu(x)
            return x

    m_model = MNMModel()
    m_model.infer_mode()
    input_shape = (batch_size, m, k)
    m_x, _ = randn_torch(input_shape, dtype=dtype)
    m_dy, _ = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    m_ytrue, _ = one_hot_torch(batch_size=batch_size, num_classes=10)
    m_model.to(dtype=dtype)
    return RAFBencher(m_model, input_shape, [m_x], m_dy, m_ytrue, ref_bencher=None)


@reg_model("raf")
def matmul_tn_scaled_add(batch_size, shape, dtype, include_orig_model=False):
    m, n, k = shape
    # matmul_tn((k, m), (k, n)), outputs (m, n)

    m_w, _ = randn_torch((k, n), dtype=dtype)
    m_beta = raf.array(0.01, dtype=dtype)
    m_bias, _ = randn_torch((m, n), dtype=dtype)

    class MNMModel(raf.Model):
        def build(self):
            self.w = m_w
            self.beta = m_beta
            self.bias = m_bias

        @raf.model.trace
        def forward(self, x):
            a = raf.matmul_tn(x, self.w)
            b = raf.multiply(self.beta, self.bias)
            return raf.add(a, b)

    m_model = MNMModel()
    m_model.infer_mode()
    input_shape = (k, m)
    m_x, _ = randn_torch(input_shape, dtype=dtype)
    m_dy, _ = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    return RAFBencher(m_model, input_shape, [m_x], m_dy, None, ref_bencher=None)
