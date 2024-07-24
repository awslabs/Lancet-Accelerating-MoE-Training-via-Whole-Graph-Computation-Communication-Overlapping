# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name,protected-access,too-many-locals,attribute-defined-outside-init
import numpy as np
import torch
import raf
import tvm
from raf._core.device import Device
from raf._ffi.pass_ import InferType, AutoDiff, LiftBranchBody, RevInlineAppliedBackward
from raf._ffi.pass_ import LambdaLift, FlattenClosure
from raf.ir import RAFSequential
from raf.model.trace import _get_func_inputs
from raf.testing import randn, randn_torch, one_hot_torch
from raf.testing.utils import run_trace_record
from raf.optim.sgd import with_sgd
from raf.distributed import get_context
from raf._core.executor import VMExecutor

def ad_passes(mod, requires_grads=None):
    if requires_grads is None:
        requires_grads = []
    seq = RAFSequential(
        [
            InferType(),
            LambdaLift(),
            InferType(),
            FlattenClosure(),
            InferType(),
            LiftBranchBody(),
            InferType(),
            AutoDiff(requires_grads),
            InferType(),
        ]
    )
    return seq(mod)

def test_simple(device="cuda(0)", pt_device="cuda:0"):

    get_context().overlap_comm_forward = True

    shape = (2048, 2048)

    class Model(raf.Model):
        def build(self, y, z):
            self.w0 = y
            self.w1 = z

        @raf.model.trace
        def forward(self, x, y_true):  # pylint: disable=no-self-use
            x = raf.relu(x)
            x = raf.matmul_nt(x, self.w0)
            x = raf.relu(x)
            x = raf.matmul_nt(x, self.w1)
            loss = raf.nll_loss(y_true=y_true, y_pred=x)
            return loss
    
    class PTModel(torch.nn.Module):
        def __init__(self, y, z):
            super().__init__()
            self.w0 = torch.nn.Parameter(y, requires_grad=True)
            self.w1 = torch.nn.Parameter(z, requires_grad=True)

        def forward(self, x, y_true):
            x = torch.relu(x)
            x = torch.matmul(x, self.w0.t())
            x = torch.relu(x)
            x = torch.matmul(x, self.w1.t())
            loss = torch.nn.functional.nll_loss(x, y_true)
            return loss


    m_y, t_y = randn(shape, device=device, requires_grad=True)
    m_z, t_z = randn(shape, device=device, requires_grad=True)
    m_y.requires_grad = True
    m_z.requires_grad = True
    t_y = torch.tensor(t_y, device=pt_device)
    t_z = torch.tensor(t_z, device=pt_device)

    m_model = Model(m_y, m_z)
    t_model = PTModel(t_y, t_z)
    m_model.to(device=device)
    t_model.to(device=pt_device)
    m_model.train_mode()
    t_model.train()

    m_model, m_fw_model = with_sgd()(m_model)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)

    m_x, t_x = randn(shape, device=device, requires_grad=True)
    t_x = torch.tensor(t_x, device=pt_device)

    m_ytrue, t_ytrue = one_hot_torch(batch_size=shape[0], num_classes=shape[1], device=device)
    m_x.requires_grad = True
    m_ytrue.requires_grad = True

    m_dy, _ = randn_torch((1,), std=0.0, mean=1.0, requires_grad=False)

    ad_record = m_model._internal(m_dy, m_x, m_ytrue)
    ad_mod = ad_record.mod

    bw_mod = InferType()(RevInlineAppliedBackward()(InferType()(ad_mod)))

    fw_record = m_fw_model._internal(m_dy, m_x, m_ytrue)
    fw_mod = fw_record.mod

    print(bw_mod)
    print("<<<<<<<<<<<<<<<<<<<<<<<<")
    print(fw_mod)

    raf.utils.memory_profiler.reset()
    raf.utils.memory_profiler.start()

    # first half cycle
    # loss, intermediate results
    m_loss, a4, a3, a1 = run_trace_record(fw_record, device, [m_dy, m_x, m_ytrue])
    t_loss = t_model(t_x, t_ytrue)
    print("initial:", m_loss.numpy(), t_loss.detach().cpu().numpy())

    a4i = raf.array(())
    a3i = raf.array(())
    a1i = raf.array(())

    a4i.update_value(a4)
    a3i.update_value(a3)
    a1i.update_value(a1)

    vm_inputs = _get_func_inputs(ad_record, [m_dy, m_x, m_ytrue], {}, get_handle=False)

    vm_inputs.append(a4i)
    vm_inputs.append(a3i)
    vm_inputs.append(a1i)

    t_loss = t_model(t_x, t_ytrue)

    with Device(device):
        with tvm.transform.PassContext(
            opt_level=3,
            disabled_pass = ["AutoSchedulerLayoutRewrite"],
        ):
            vm = VMExecutor(bw_mod, device, dryrun=False)
            vm_exec = vm.make_executor()
            for i in range(10):
                vm_out = vm_exec(*vm_inputs)
                t_loss.backward()
                t_optimizer.step()
                t_optimizer.zero_grad()
                t_loss = t_model(t_x, t_ytrue)
                print(f"iter {i}: m_loss:", vm_out[0].numpy(), "t_loss:", t_loss.detach().cpu().numpy())
                _, a4, a3, a1 = vm_out
                a4i.update_value(a4)
                a3i.update_value(a3)
                a1i.update_value(a1)
            vm_out = vm_exec(*vm_inputs)
            t_loss.backward()
            t_optimizer.step()
            t_optimizer.zero_grad()
            t_loss = t_model(t_x, t_ytrue)
            print(f"Final : m_loss:", vm_out[0].numpy(), "t_loss:", t_loss.detach().cpu().numpy())
    raf.utils.memory_profiler.stop()
    ret_map = raf.utils.memory_profiler.get_max_memory_info(raf.Device(device))
    mem_trace = raf.utils.memory_profiler.get_memory_trace(raf.Device(device))
    print("Peak memory:", ret_map['max_used'].value, "MBs.")
    print(mem_trace)

test_simple()