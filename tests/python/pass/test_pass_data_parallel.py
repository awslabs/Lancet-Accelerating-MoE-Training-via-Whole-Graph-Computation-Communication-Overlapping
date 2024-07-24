# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import pytest
import numpy as np

import raf
from raf import distributed as dist
from raf.testing import randn, run_infer_type
from raf._ffi.pass_ import InferType, AutoDiff, AutoDataParallel
from raf.ir import RAFSequential
import tvm
from tvm import relay
from raf.model import Conv2d, Linear, BatchNorm
from raf import distributed as dist
from raf._ffi.pass_ import AutoDataParallel, AutoDiff, SimplifyExpr, InferType
from raf.ir import RAFSequential
from raf._core.ir_ext import extended_var

class LetBuilder:
    """Helper class to build nested let expression in python"""
    def __init__(self):
        self.vars = []
        self.values = []

    def append(self, value):
        """Append a expression into let list, returns the created var"""
        self.vars.append(extended_var(name_hint=""))
        self.values.append(value)
        return self.vars[-1]

    def ret(self, body):
        """Construct the expression using `body` as return value"""
        for var, value in reversed(list(zip(self.vars, self.values))):
            body = tvm.relay.Let(var, value, body)
        self.vars.clear()
        self.values.clear()
        return body


def one_hot(batch_size, num_classes, device="cuda"):
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = raf.array(targets, device=device)
    assert list(m_x.shape) == [
        batch_size,
    ]
    return m_x


def randn(shape, *, device="cuda", dtype="float32", std=1.0, mean=0.0,
          requires_grad=False, positive=False):
    if positive:
        x = np.abs(np.random.randn(*shape)) * std + mean
    else:
        x = np.random.randn(*shape) * std + mean
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = raf.array(x, device=device)
    if requires_grad:
        m_x.requires_grad = True
    return m_x


class RAFTest(raf.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                            bias=False)
        self.bn1 = BatchNorm(6)
        self.linear1 = Linear((input_shape // 2) ** 2 * 6,
                              num_classes)
    # pylint: enable=attribute-defined-outside-init

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss

    @raf.model.trace
    def forward_infer(self, x):
        out = self.bn1(self.conv1(x))
        out = raf.sigmoid(out)
        out = raf.avg_pool2d(out, (2, 2), (2, 2))
        out = raf.batch_flatten(out)
        out = self.linear1(out)
        return out

# pylint: disable=unused-variable
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "config",
    [
        (2, 2, 2),
    ],
)
def test_dp(config):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True
    dctx.force_sync_after_comm = force_sync_after_comm
    device = f"cuda({dctx.local_rank})"
    const, _ = randn([config[0], config[1]], device=device)
    nccl_version = raf.build.with_nccl()

    class TestModel(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.c = const

        # pylint: enable=attribute-defined-outside-init

        @raf.model.trace
        def forward(self, x, y_true):
            y_pred = self.forward_infer(x)
            loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
            return loss

        @raf.model.trace
        def forward_infer(self, x):
            out = raf.matmul(x, self.c)
            return out

    def expected():
        shape = [config[0], config[0]]

        # Params
        x = relay.var("x", relay.TensorType(shape))
        c = relay.var("c", relay.TensorType(shape))
        y_true = relay.var(
            "y_true",
            relay.TensorType(
                [
                    2,
                ],
                dtype="int64",
            ),
        )

        # Forward IR components
        expr_a1 = raf.ir.op.matmul(x, c)
        var_a1 = relay.var("a1")

        expr_a2 = raf.ir.op.nll_loss(y_true, var_a1)
        var_a2 = relay.var("a2")

        # Backward IR components
        dy = relay.var(
            "dy",
            relay.TensorType(
                [
                    1,
                ],
                dtype="float32",
            ),
        )
        var_closure = relay.var("closure")

        expr_x1 = raf.ir.op.nll_loss_dpred(dy, y_true, var_a1)
        var_x0 = relay.var("x0")

        expr_x2 = raf.ir.op.matmul_nt(var_x0, c)
        var_x1 = relay.var("x1")

        op_matmul_nt = raf._ffi.op.GetOp('raf.op.matmul_nt')
        expr_x2 = tvm.relay.Call(op_matmul_nt, [var_x0, c])
        var_x1 = ll.append(expr_x2)

        expr_x3 = raf.ir.op.matmul_tn(x, var_x0)
        var_x2 = relay.var("x2")

        op__allreduce = raf._ffi.op.GetOp('raf.op._allreduce')
        expr_g = tvm.relay.Call(op__allreduce, [var_t])
        var_g = ll.append(expr_g)

        expr_x4 = raf.ir.op.zeros_like(y_true)
        var_x3 = relay.var("x3")

        op_matmul_tn = raf._ffi.op.GetOp('raf.op.matmul_tn')
        expr_x3 = tvm.relay.Call(op_matmul_tn, [x, var_x0])
        var_x2 = ll.append(expr_x3)

        if nccl_version > 21000:
            expr_g = raf.ir.op._allreduce(allreduce_in, "avg")
            var_g = relay.var("g")

            expr_g1 = raf.ir.op._allreduce(allreduce_in1, "avg")
            var_g1 = relay.var("g1")

            expr_g2 = raf.ir.op._allreduce(allreduce_in2, "avg")
            var_g2 = relay.var("g2")
        else:
            fdeno = raf.ir.const(float(dctx.size), dtype="float32")
            ideno = raf.ir.const(dctx.size, dtype="int64")

            expr_g = raf.ir.op._allreduce(allreduce_in)
            var_g_sum = relay.var("g_sum")
            expr_avg = raf.ir.op.divide(var_g_sum, fdeno)
            var_g = relay.var("g")

            expr_g1 = raf.ir.op._allreduce(allreduce_in1)
            var_g1_sum = relay.var("g_sum1")
            expr_avg1 = raf.ir.op.divide(var_g1_sum, fdeno)
            var_g1 = relay.var("g1")

            expr_g2 = raf.ir.op._allreduce(allreduce_in2)
            var_g2_sum = relay.var("g_sum2")
            expr_avg2 = raf.ir.op.divide(var_g2_sum, ideno)
            var_g2 = relay.var("g2")

        expr_x5 = relay.Tuple([var_g, var_g2, var_g1])
        var_x5 = relay.var("x5")

        # Forward IR components
        expr_ret = relay.Tuple([var_a2, var_closure])
        var_ret = relay.var("ret")
        if nccl_version >= 21000:
            # Construct Backward IR as a closure
            let8 = relay.Let(var_x5, expr_x5, var_x5)
            let7 = relay.Let(var_g2, expr_g2, let8)
            let_ad2 = relay.Let(allreduce_in2, expr_t2, let7)
            let6 = relay.Let(var_x3, expr_x4, let_ad2)
            let5 = relay.Let(var_g1, expr_g1, let6)
            let_ad1 = relay.Let(allreduce_in1, expr_t1, let5)
            let4 = relay.Let(var_x2, expr_x3, let_ad1)
            let3 = relay.Let(var_g, expr_g, let4)
            let_ad = relay.Let(allreduce_in, expr_t, let3)
            let2 = relay.Let(var_x1, expr_x2, let_ad)
            let1 = relay.Let(var_x0, expr_x1, let2)
            closure_func = relay.Function([dy], let1)
        else:
            expr_x5 = tvm.relay.Tuple([var_g_tgi, var_g_tgi2, var_g_tgi1])
            var_x5 = ll.append(expr_x5)

        let1 = ll.ret(var_x5)

        # Forward IR components
        expr_ret = tvm.relay.Tuple([var_a2, var_closure])
        var_ret = tvm.relay.var('ret')

        # Construct Backward IR as a closure
        closure_func = tvm.relay.Function([dy], let1)

        # Construct Forward IR
        let10 = relay.Let(var_ret, expr_ret, var_ret)
        let0 = relay.Let(var_closure, closure_func, let10)

        let_1 = relay.Let(var_a2, expr_a2, let0)
        let_2 = relay.Let(var_a1, expr_a1, let_1)

        return relay.Function([x, y_true, c], let_2)

    m_model = TestModel()
    m_model.to(device=device)
    m_model.train_mode()

    m_x = randn([config[0], config[0]], device=device, requires_grad=True)
    m_y = one_hot(batch_size=config[0], num_classes=config[1], device=device)
    m_x.requires_grad = True
    m_y.requires_grad = True

    record = m_model._internal(m_x, m_y)
    mod_before = record.mod
    passes = [
        InferType(),
        AutoDiff(record.requires_grads),
        InferType(),
        AutoDataParallel(),
        InferType(),
    ]
    seq = RAFSequential(passes)
    mod = seq(mod_before)
    func_after = mod["main"]
    func_expected = expected()
    expected_mod = tvm.IRModule()
    expected_mod['main'] = func_expected
    expected_mod = SimplifyExpr()(expected_mod)
    text = func_after.astext()
    assert "raf.op._allreduce" in text
    assert tvm.ir.structural_equal(func_after, func_expected)
    dctx.enable_data_parallel = False


if __name__ == "__main__":
    pytest.main([__file__])