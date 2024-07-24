import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import torch.nn as nn
import sys
import argparse
import math

import raf
from raf.model import Conv2d, Linear, BatchNorm
from raf.random.nn import kaiming_uniform
from raf.random.np import zeros_, ones_, normal

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=100, shuffle=False)

def t2m_param(param, ctx="cuda"):
    return raf.ndarray(param.detach().numpy(), ctx=ctx)  # pylint: disable=unexpected-keyword-arg

def one_hot(target, n_class):
    label = np.eye(n_class)[target]
    return label

# pylint: disable=C0103
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class MNMReLU6(raf.Model):
    def build(self):
        pass
    
    def forward(self, x):
        return raf.clip(raf.relu(x), 0, 6)

class MNM_Adaptive_Avg_Pool2d(raf.Model):
    def build(self, x, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.stride = (math.floor(x[2]/self.output_size[0]), math.floor(x[3]/self.output_size[1]))
        self.kernel_size = (x[2] - (self.output_size[0] - 1) * self.stride[0],
                            x[3] - (self.output_size[1] - 1) * self.stride[1])
    def forward(self, x):  # pylint: disable=no-self-use
        return raf.avg_pool2d(x, kernel=self.kernel_size, stride=self.stride)

class MNMConvBNReLU(raf.Model):
    def build(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        # BatchNorm2D
        current_batchnorm = BatchNorm(out_planes)
        current_batchnorm.w = ones_(current_batchnorm.w.shape)
        current_batchnorm.b = zeros_(current_batchnorm.b.shape)
        # Conv2D
        current_conv2d = Conv2d(in_planes, out_planes, kernel_size, stride,
                                padding, groups=groups, bias=False)
        current_conv2d.w = kaiming_uniform(current_conv2d.w.shape, mode='fan_out')
        if current_conv2d.b_shape is not None:
            current_conv2d.b = zeros_(current_conv2d.b_shape)
        self.conv1 = current_conv2d
        self.batchnorm = current_batchnorm
        self.relu6 = MNMReLU6()
        self.features = [self.conv1,
                         self.batchnorm,
                         self.relu6]
        self.layers = raf.model.Sequential(*self.features)

    def forward(self, x):
        return self.layers(x)


class MNMInvertedResidual(raf.Model):
    def build(self, inp, oup, stride, expand_ratio):
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            # pw
            layers.append(MNMConvBNReLU(inp, hidden_dim, kernel_size=1))
        # BatchNorm2D
        self.current_batchnorm = BatchNorm(oup)
        self.current_batchnorm.w = ones_(self.current_batchnorm.w.shape)
        self.current_batchnorm.b = zeros_(self.current_batchnorm.b.shape)

        # Conv2D
        self.current_conv2d = Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.current_conv2d.w = kaiming_uniform(self.current_conv2d.w.shape, mode='fan_out')
        if self.current_conv2d.b_shape is not None:
            self.current_conv2d.b = zeros_(self.current_conv2d.b_shape)

        layers.extend([
            # dw
            MNMConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            self.current_conv2d,
            self.current_batchnorm,
        ])
        self.layers = layers
        self.conv = raf.model.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return raf.add(x, self.conv(x))
        return self.conv(x)

class MNMMobileNetV2(raf.Model):
    def build(self,
              num_data=0,
              x_shape=None,
              num_classes=1000,
              width_mult=1.0,
              inverted_residual_setting=None,
              round_nearest=8,
              block=None):
        self.num_data = num_data
        self.aap2d = MNM_Adaptive_Avg_Pool2d(x_shape, 1)
        if block is None:
            block = MNMInvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [MNMConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # build last several layers
        features.append(MNMConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it Sequential
        self.features = raf.model.Sequential(*features)

        # building classifier
        L = Linear(self.last_channel, num_classes)
        L.w = normal(0, 0.01, (L.w).shape)
        L.b = zeros_((L.b).shape)
        self.L = L
        self.classifier = raf.model.Sequential(
            L,
        )

    def store_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    @raf.model.trace
    def forward_infer(self, x):
        x = self.features(x)
        x = self.aap2d(x)
        x = raf.reshape(x, shape=(self.batch_size, 1280))
        x = self.classifier(x)
        return x

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss


data_shape = train_loader.dataset[0][0].detach().numpy().shape
data_shape = (100, *data_shape)

ctx="cuda"
isTrain = True
m_model = MNMMobileNetV2(100, data_shape, num_classes=10)
m_model.to(ctx=ctx)
m_param_dict = m_model.state()
m_optimizer = raf.optim.SGD(m_param_dict.values(), 0.001)
m_model.train_mode()

for epoch in range(50):
    print(epoch, "   =====>")
    m_model.train_mode()
    Loss = 0
    train_correct = 0
    total = 0
    for batch_num, (data, target) in enumerate(train_loader):
        m_data = raf.ndarray(data.detach().numpy(), dtype="float32", ctx=ctx)
        numpy_data = data.detach().numpy()
        m_data.requires_grad = True
        target1 = one_hot(target.detach().numpy(), 10)
        m_target = raf.ndarray(target1.astype(np.float32), dtype="float32", ctx=ctx)
        m_model.store_batch_size(numpy_data.shape[0])
        m_loss = m_model(m_data, m_target)
        y_pred = m_model.forward_infer(m_data)
        y_pred = np.argmax((y_pred).asnumpy(), 1)
        Loss += m_loss.asnumpy()[0]
        y_true = target.detach().numpy()
        total += y_true.shape[0]
        train_correct += np.sum(y_pred == y_true)
        m_loss.backward()
        m_optimizer.step()
    print("train ====>")
    print("train acc: ", train_correct / total)
    print("train loss: ", Loss)
    print("test ====>")
    Loss = 0
    test_correct = 0
    total = 0
    m_model.infer_mode()
    for batch_num, (data, target) in enumerate(test_loader):
        m_data = raf.ndarray(data.detach().numpy(), dtype="float32", ctx=ctx)
        target1 = one_hot(target.detach().numpy(), 10)
        m_target = raf.ndarray(target1.astype(np.float32), dtype="float32", ctx=ctx)
        y_pred = m_model.forward_infer(m_data)
        y_pred = np.argmax((y_pred).asnumpy(), 1)
        y_true = target.detach().numpy()
        total += y_true.shape[0]
        test_correct += np.sum(y_pred == y_true)
    print("test acc: ", test_correct / total)
    test_acc = test_correct / total 