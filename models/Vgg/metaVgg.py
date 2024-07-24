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
import time

import raf
from raf.model import Conv2d, Linear, BatchNorm
from torchVgg import VGG

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

class MNMVgg11(raf.Model):
    def build(self):
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(128)
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm(256)
        self.conv4 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = BatchNorm(256)
        self.conv5 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = BatchNorm(512)
        self.conv6 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = BatchNorm(512)
        self.conv7 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = BatchNorm(512)
        self.conv8 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = BatchNorm(512)
        self.classifier = Linear(512, 10)

    @raf.model.trace
    def forward_infer(self, x):
        x = raf.relu(self.bn1(self.conv1(x)))
        x = raf.max_pool2d(x, (2, 2), (2, 2))
        x = raf.relu(self.bn2(self.conv2(x)))
        x = raf.max_pool2d(x, (2, 2), (2, 2))
        x = raf.relu(self.bn3(self.conv3(x)))
        x = raf.relu(self.bn4(self.conv4(x)))
        x = raf.max_pool2d(x, (2, 2), (2, 2))
        x = raf.relu(self.bn5(self.conv5(x)))
        x = raf.relu(self.bn6(self.conv6(x)))
        x = raf.max_pool2d(x, (2, 2), (2, 2))
        x = raf.relu(self.bn7(self.conv7(x)))
        x = raf.relu(self.bn8(self.conv8(x)))
        x = raf.max_pool2d(x, (2, 2), (2, 2))
        x = raf.avg_pool2d(x, (1, 1), (1, 1))
        x = raf.batch_flatten(x)
        x = self.classifier(x)
        return x

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss

def one_hot(target, n_class):
    label = np.eye(n_class)[target]
    return label

def t2m_param(param, ctx="cuda"):
    return raf.ndarray(param.cpu().detach().numpy(), ctx=ctx)


ctx="cuda"
isTrain = True
t_model = VGG().to(torch.device('cuda'))
m_model = MNMVgg11()
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
        m_data.requires_grad = True
        target1 = one_hot(target.detach().numpy(), 10)
        m_target = raf.ndarray(target1.astype(np.float32), dtype="float32", ctx=ctx)
        m_loss = m_model(m_data, m_target)
        Loss += m_loss.asnumpy()[0]
        m_loss.backward()
        m_optimizer.step()
    print("loss: ", Loss)
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
    print("acc: ", test_correct / total)
    test_acc = test_correct / total