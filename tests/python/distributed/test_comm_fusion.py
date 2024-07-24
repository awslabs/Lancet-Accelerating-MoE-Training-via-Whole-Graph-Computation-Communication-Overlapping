# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import tvm
import raf
from raf.distributed import LancetScheduleSimulator
from raf.model import Conv2d, Linear, BatchNorm
from raf.testing import (
    one_hot_torch,
    randn_torch,
)
from raf._ffi.pass_ import DataParallelSchedule

os.environ["ZERO_OPT_LEVEL"] = "2"
os.environ["RANK_SIZE"] = "4"
os.environ["RANK"] = "2"
# os.environ["SIMULATION_DEBUG_PREFIX"] = "/models/profiles/test_comm_fusion"
os.environ["DEBUG_DUMP_PROFILE_PREFIX"] = "/models/profiles/test_comm_fusion"
os.environ["DISABLE_PARTITION"] = "1"
os.environ["DISABLE_FUSION"] = "0"
# os.environ["DEBUG_SCHEDULE_FIXED_PARAMS"] = "4.84295e-05;0.000697595;0.000168263;0.000468476"
# os.environ["DEBUG_CM_FIXED_PARAMS"] = "50;50000"
os.environ["PROFILE_SETTING"] = "2;5"

class RAFTest(raf.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=True)
        self.bn1 = BatchNorm(6)
        self.linear1 = Linear((input_shape // 2) ** 2 * 6, num_classes)

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

def test_state_partition():
    """Note that this test only verifies the IR with SGD without checking the correctness.
    Accordingly, this test does not require multiple devices.
    """
    shape, n_classes = 28, 10
    batch_size = 7
    m_model = RAFTest(shape, 10)
    m_model.train_mode()
    m_optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)[0]

    device = "cuda"
    m_x, _ = randn_torch([batch_size, 3, shape, shape], requires_grad=True, device=device)
    m_dy, _ = randn_torch((), std=0.0, mean=1.0, device=device, requires_grad=False)
    m_ytrue, _ = one_hot_torch(batch_size=batch_size, num_classes=n_classes, device=device)
    args = [m_dy, m_x, m_ytrue]

    record = m_optimizer._internal(*args)
    mod = record.mod
    config = {
        "raf.dp_schedule.use_profile": True,
    }
    with tvm.transform.PassContext(
            opt_level=3,
            config=config,
        ):
        # profile the expr
        mod = DataParallelSchedule(1)(mod)

    sim = LancetScheduleSimulator()
    expr = sim.load_profile("/models/profiles/test_comm_fusion")
    print("Input expr:")
    print(raf.ir.AsText(expr))
    opt_expr = sim.run(expr, "FIFO", 1)
    print("Opt expr: ")
    print(raf.ir.AsText(opt_expr))


if __name__ == "__main__":
    test_state_partition()
