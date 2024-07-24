# pylint: disable=protected-access
from re import L, S
from threading import local
import random
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import raf
from raf.model import Conv2d, Linear
from raf.optim.sgd import with_sgd
from raf.testing import check, to_torch_dev, t2m_param, run_vm_model, with_seed
from raf.distributed import get_context
from raf.optim.optim import with_autodiff

def randn_torch_w_slice(shape, *, num_slices=1, device="cpu", dtype="float32", requires_grad=False, mean=0.0, std=1.0,
                positive=False):
    """Helper function to generate a pair of raf and torch arrays"""
    x = np.random.randn(*shape) * std + mean
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)

    if shape:
        assert num_slices > 0 and shape[0] % num_slices == 0
        slice_size = int(shape[0] / num_slices)
        m_xs = []
        for slice_idx in range(num_slices):
            m_xs.append(raf.array(n_x[slice_size*slice_idx:slice_size*(slice_idx+1)].copy(), device=device))
            m_xs[slice_idx].requires_grad = requires_grad
            assert list(m_xs[slice_idx].shape) == [int(shape[0] / num_slices)]+list(shape[1:])
    else:
        assert num_slices == 1
        m_xs = [raf.array(n_x, device=device)]
        m_xs[0].requires_grad = requires_grad
    t_x = torch.tensor(n_x, requires_grad=requires_grad, device=to_torch_dev(device))  # pylint: disable=not-callable
    return m_xs, t_x

def one_hot_torch_w_slice(batch_size, num_classes, num_slices=1, device="cpu"):
    """Helper function to generate one hot tensors in raf and torch"""
    targets = np.random.randint(0, num_classes, size=batch_size)
    assert num_slices > 0 and batch_size % num_slices == 0
    slice_size = int(batch_size / num_slices)
    m_xs = []
    for slice_idx in range(num_slices):
        m_xs.append(raf.array(targets[slice_size*slice_idx:slice_size*(slice_idx+1)], device=device))
        assert list(m_xs[slice_idx].shape) == [int(batch_size / num_slices)]
    t_x = torch.tensor(targets, requires_grad=False, device=to_torch_dev(device))  # pylint: disable=not-callable
    assert list(t_x.shape) == [batch_size]
    return m_xs, t_x

def get_dy(device="cpu"):
    m_dys, t_dy = randn_torch_w_slice((), num_slices=1, device=device, mean=1, std=0)
    return m_dys[0], t_dy
    

class TorchLeNet(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5,
                               padding=2,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               bias=False)
        self.linear1 = nn.Linear(((input_shape // 2 - 4) // 2) ** 2 * 16,
                                 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x, y_true): # pylint: disable=arguments-differ
        y_pred = self.forward_infer(x)
        y_pred = F.log_softmax(y_pred, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss

    def forward_infer(self, x):
        out = self.conv1(x)
        out = torch.sigmoid(out) # pylint: disable=no-member
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = self.conv2(out)
        out = torch.sigmoid(out) # pylint: disable=no-member
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1) # pylint: disable=no-member
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

class RAFLeNet(raf.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                            bias=False)
        self.conv2 = Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            bias=False)
        self.linear1 = Linear(((input_shape // 2 - 4) // 2) ** 2 * 16,
                              120)
        self.linear2 = Linear(120, 84)
        self.linear3 = Linear(84, num_classes)

    # pylint: enable=attribute-defined-outside-init

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss

    @raf.model.trace
    def forward_infer(self, x):
        out = self.conv1(x)
        out = raf.sigmoid(out)
        out = raf.avg_pool2d(out, (2, 2), (2, 2))
        out = self.conv2(out)
        out = raf.sigmoid(out)
        out = raf.avg_pool2d(out, (2, 2), (2, 2))
        out = raf.batch_flatten(out)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

@pytest.mark.skip()
@with_seed(42)
def test_lenet(config, use_sgd):
    dctx = get_context()
    dctx.enable_data_parallel = True
    local_rank = dctx.local_rank
    device = "cuda({})".format(local_rank)

    assert local_rank < 2 and local_rank >= 0
    assert dctx.local_size == dctx.size
    assert dctx.local_size == 2

    t_model = TorchLeNet(*(config[1:-1]))
    t_model.to(device=to_torch_dev(device))
    m_model = RAFLeNet(*(config[1:-1]))
    m_model.to(device=device)
    m_model.conv1.w = t2m_param(t_model.conv1.weight, device=device)
    m_model.conv2.w = t2m_param(t_model.conv2.weight, device=device)
    m_model.linear1.w = t2m_param(t_model.linear1.weight, device=device)
    m_model.linear1.b = t2m_param(t_model.linear1.bias, device=device)
    m_model.linear2.w = t2m_param(t_model.linear2.weight, device=device)
    m_model.linear2.b = t2m_param(t_model.linear2.bias, device=device)
    m_model.linear3.w = t2m_param(t_model.linear3.weight, device=device)
    m_model.linear3.b = t2m_param(t_model.linear3.bias, device=device)

    print("### Switch to training mode")
    m_model.train_mode()
    t_model.train()

    if not use_sgd:
        m_model = with_autodiff(m_model)
    else:
        m_optimizer = with_sgd(learning_rate=0.1, momentum=0.01)(m_model)[0]
        t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)

    if use_sgd:
        t_optimizer.zero_grad()

    if use_sgd:
        t_optimizer.zero_grad()
        t_optimizer.step()
    
    if not use_sgd:
        # generate input for two devices
        m_xs, t_x = randn_torch_w_slice([config[0], 3, config[1], config[1]], num_slices=2, requires_grad=True, device=device)
        m_ys, t_y = one_hot_torch_w_slice(batch_size=config[0], num_classes=config[2], num_slices=2, device=device)
        m_dy, _ = get_dy(device=device)

        m_x = m_xs[local_rank]
        m_y = m_ys[local_rank]

        t_loss = t_model(t_x, t_y)
        t_loss.backward()
        _, gradients = run_vm_model(m_model, device, [m_dy, m_x, m_y], fuse_level=1)
        _, _, conv1_grad, conv2_grad, linear1b_grad, linear1w_grad, linear2b_grad, linear2w_grad, linear3b_grad, linear3w_grad = gradients

        check(conv1_grad.numpy(), t_model.conv1.weight.grad, rtol=1e-4, atol=1e-4)
        check(conv2_grad.numpy(), t_model.conv2.weight.grad, rtol=1e-4, atol=1e-4)
        check(linear1w_grad.numpy(), t_model.linear1.weight.grad, rtol=1e-4, atol=1e-4)
        check(linear1b_grad.numpy(), t_model.linear1.bias.grad, rtol=1e-4, atol=1e-4)
        check(linear2w_grad.numpy(), t_model.linear2.weight.grad, rtol=1e-4, atol=1e-4)
        check(linear2b_grad.numpy(), t_model.linear2.bias.grad, rtol=1e-4, atol=1e-4)
        check(linear3w_grad.numpy(), t_model.linear3.weight.grad, rtol=1e-4, atol=1e-4)
        check(linear3b_grad.numpy(), t_model.linear3.bias.grad, rtol=1e-4, atol=1e-4)
    else:
        for _ in range(config[3]):
            # generate input for two devices
            m_xs, t_x = randn_torch_w_slice([config[0], 3, config[1], config[1]], num_slices=2, requires_grad=True, device=device)
            m_ys, t_y = one_hot_torch_w_slice(batch_size=config[0], num_classes=config[2], num_slices=2, device=device)
            m_dy, _ = get_dy(device=device)

            m_x = m_xs[local_rank]
            m_y = m_ys[local_rank]

            _ = run_vm_model(m_optimizer, device, [m_dy, m_x, m_y], fuse_level=1)

            t_optimizer.zero_grad()
            t_loss = t_model(t_x, t_y)
            t_loss.backward()
            t_optimizer.step()

            check(m_model.conv1.w, t_model.conv1.weight, rtol=1e-4, atol=1e-4)
            check(m_model.conv2.w, t_model.conv2.weight, rtol=1e-4, atol=1e-4)
            check(m_model.linear1.w, t_model.linear1.weight, rtol=1e-4, atol=1e-4)
            check(m_model.linear1.b, t_model.linear1.bias, rtol=1e-4, atol=1e-4)
            check(m_model.linear2.w, t_model.linear2.weight, rtol=1e-4, atol=1e-4)
            check(m_model.linear2.b, t_model.linear2.bias, rtol=1e-4, atol=1e-4)
            check(m_model.linear3.w, t_model.linear3.weight, rtol=1e-4, atol=1e-4)
            check(m_model.linear3.b, t_model.linear3.bias, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    if raf.build.with_distributed():
        for config in [((10, 224, 1000, 5), False),
                       ((10, 224, 1000, 5), True),
                       ((10, 32, 100, 5), False),
                       ((10, 32, 10, 5), True),
                       ((10, 28, 10, 5), False),
                       ((10, 28, 10, 5), True)]:
            test_lenet(*config)
        raf.distributed.RemoveCommunicator()
