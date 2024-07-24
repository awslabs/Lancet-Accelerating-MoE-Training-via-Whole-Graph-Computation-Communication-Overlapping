import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import raf
from raf.model import BatchNorm, Conv2d, Linear, Sequential
from raf.testing import check, one_hot_torch, t2m_param, get_device_list, randn_torch
from raf.testing.utils import get_param, set_param


# map from m_model params to t_model params
resnet18_param_map = {
    "bn1.b": "bn1.bias",
    "bn1.running_mean": "bn1.running_mean",
    "bn1.running_var": "bn1.running_var",
    "bn1.w": "bn1.weight",
    "conv1.w": "conv1.weight",
    "fc1.b": "fc1.bias",
    "fc1.w": "fc1.weight",
    "layer1.seq_0.bn1.b": "layer1.0.bn1.bias",
    "layer1.seq_0.bn1.running_mean": "layer1.0.bn1.running_mean",
    "layer1.seq_0.bn1.running_var": "layer1.0.bn1.running_var",
    "layer1.seq_0.bn1.w": "layer1.0.bn1.weight",
    "layer1.seq_0.bn2.b": "layer1.0.bn2.bias",
    "layer1.seq_0.bn2.running_mean": "layer1.0.bn2.running_mean",
    "layer1.seq_0.bn2.running_var": "layer1.0.bn2.running_var",
    "layer1.seq_0.bn2.w": "layer1.0.bn2.weight",
    "layer1.seq_0.conv1.w": "layer1.0.conv1.weight",
    "layer1.seq_0.conv2.w": "layer1.0.conv2.weight",
    "layer1.seq_1.bn1.b": "layer1.1.bn1.bias",
    "layer1.seq_1.bn1.running_mean": "layer1.1.bn1.running_mean",
    "layer1.seq_1.bn1.running_var": "layer1.1.bn1.running_var",
    "layer1.seq_1.bn1.w": "layer1.1.bn1.weight",
    "layer1.seq_1.bn2.b": "layer1.1.bn2.bias",
    "layer1.seq_1.bn2.running_mean": "layer1.1.bn2.running_mean",
    "layer1.seq_1.bn2.running_var": "layer1.1.bn2.running_var",
    "layer1.seq_1.bn2.w": "layer1.1.bn2.weight",
    "layer1.seq_1.conv1.w": "layer1.1.conv1.weight",
    "layer1.seq_1.conv2.w": "layer1.1.conv2.weight",
    "layer2.seq_0.bn1.b": "layer2.0.bn1.bias",
    "layer2.seq_0.bn1.running_mean": "layer2.0.bn1.running_mean",
    "layer2.seq_0.bn1.running_var": "layer2.0.bn1.running_var",
    "layer2.seq_0.bn1.w": "layer2.0.bn1.weight",
    "layer2.seq_0.bn2.b": "layer2.0.bn2.bias",
    "layer2.seq_0.bn2.running_mean": "layer2.0.bn2.running_mean",
    "layer2.seq_0.bn2.running_var": "layer2.0.bn2.running_var",
    "layer2.seq_0.bn2.w": "layer2.0.bn2.weight",
    "layer2.seq_0.conv1.w": "layer2.0.conv1.weight",
    "layer2.seq_0.conv2.w": "layer2.0.conv2.weight",
    "layer2.seq_0.downsample.seq_0.w": "layer2.0.downsample.0.weight",
    "layer2.seq_0.downsample.seq_1.b": "layer2.0.downsample.1.bias",
    "layer2.seq_0.downsample.seq_1.running_mean": "layer2.0.downsample.1.running_mean",
    "layer2.seq_0.downsample.seq_1.running_var": "layer2.0.downsample.1.running_var",
    "layer2.seq_0.downsample.seq_1.w": "layer2.0.downsample.1.weight",
    "layer2.seq_1.bn1.b": "layer2.1.bn1.bias",
    "layer2.seq_1.bn1.running_mean": "layer2.1.bn1.running_mean",
    "layer2.seq_1.bn1.running_var": "layer2.1.bn1.running_var",
    "layer2.seq_1.bn1.w": "layer2.1.bn1.weight",
    "layer2.seq_1.bn2.b": "layer2.1.bn2.bias",
    "layer2.seq_1.bn2.running_mean": "layer2.1.bn2.running_mean",
    "layer2.seq_1.bn2.running_var": "layer2.1.bn2.running_var",
    "layer2.seq_1.bn2.w": "layer2.1.bn2.weight",
    "layer2.seq_1.conv1.w": "layer2.1.conv1.weight",
    "layer2.seq_1.conv2.w": "layer2.1.conv2.weight",
    "layer3.seq_0.bn1.b": "layer3.0.bn1.bias",
    "layer3.seq_0.bn1.running_mean": "layer3.0.bn1.running_mean",
    "layer3.seq_0.bn1.running_var": "layer3.0.bn1.running_var",
    "layer3.seq_0.bn1.w": "layer3.0.bn1.weight",
    "layer3.seq_0.bn2.b": "layer3.0.bn2.bias",
    "layer3.seq_0.bn2.running_mean": "layer3.0.bn2.running_mean",
    "layer3.seq_0.bn2.running_var": "layer3.0.bn2.running_var",
    "layer3.seq_0.bn2.w": "layer3.0.bn2.weight",
    "layer3.seq_0.conv1.w": "layer3.0.conv1.weight",
    "layer3.seq_0.conv2.w": "layer3.0.conv2.weight",
    "layer3.seq_0.downsample.seq_0.w": "layer3.0.downsample.0.weight",
    "layer3.seq_0.downsample.seq_1.b": "layer3.0.downsample.1.bias",
    "layer3.seq_0.downsample.seq_1.running_mean": "layer3.0.downsample.1.running_mean",
    "layer3.seq_0.downsample.seq_1.running_var": "layer3.0.downsample.1.running_var",
    "layer3.seq_0.downsample.seq_1.w": "layer3.0.downsample.1.weight",
    "layer3.seq_1.bn1.b": "layer3.1.bn1.bias",
    "layer3.seq_1.bn1.running_mean": "layer3.1.bn1.running_mean",
    "layer3.seq_1.bn1.running_var": "layer3.1.bn1.running_var",
    "layer3.seq_1.bn1.w": "layer3.1.bn1.weight",
    "layer3.seq_1.bn2.b": "layer3.1.bn2.bias",
    "layer3.seq_1.bn2.running_mean": "layer3.1.bn2.running_mean",
    "layer3.seq_1.bn2.running_var": "layer3.1.bn2.running_var",
    "layer3.seq_1.bn2.w": "layer3.1.bn2.weight",
    "layer3.seq_1.conv1.w": "layer3.1.conv1.weight",
    "layer3.seq_1.conv2.w": "layer3.1.conv2.weight",
    "layer4.seq_0.bn1.b": "layer4.0.bn1.bias",
    "layer4.seq_0.bn1.running_mean": "layer4.0.bn1.running_mean",
    "layer4.seq_0.bn1.running_var": "layer4.0.bn1.running_var",
    "layer4.seq_0.bn1.w": "layer4.0.bn1.weight",
    "layer4.seq_0.bn2.b": "layer4.0.bn2.bias",
    "layer4.seq_0.bn2.running_mean": "layer4.0.bn2.running_mean",
    "layer4.seq_0.bn2.running_var": "layer4.0.bn2.running_var",
    "layer4.seq_0.bn2.w": "layer4.0.bn2.weight",
    "layer4.seq_0.conv1.w": "layer4.0.conv1.weight",
    "layer4.seq_0.conv2.w": "layer4.0.conv2.weight",
    "layer4.seq_0.downsample.seq_0.w": "layer4.0.downsample.0.weight",
    "layer4.seq_0.downsample.seq_1.b": "layer4.0.downsample.1.bias",
    "layer4.seq_0.downsample.seq_1.running_mean": "layer4.0.downsample.1.running_mean",
    "layer4.seq_0.downsample.seq_1.running_var": "layer4.0.downsample.1.running_var",
    "layer4.seq_0.downsample.seq_1.w": "layer4.0.downsample.1.weight",
    "layer4.seq_1.bn1.b": "layer4.1.bn1.bias",
    "layer4.seq_1.bn1.running_mean": "layer4.1.bn1.running_mean",
    "layer4.seq_1.bn1.running_var": "layer4.1.bn1.running_var",
    "layer4.seq_1.bn1.w": "layer4.1.bn1.weight",
    "layer4.seq_1.bn2.b": "layer4.1.bn2.bias",
    "layer4.seq_1.bn2.running_mean": "layer4.1.bn2.running_mean",
    "layer4.seq_1.bn2.running_var": "layer4.1.bn2.running_var",
    "layer4.seq_1.bn2.w": "layer4.1.bn2.weight",
    "layer4.seq_1.conv1.w": "layer4.1.conv1.weight",
    "layer4.seq_1.conv2.w": "layer4.1.conv2.weight",
}

resnet50_param_map = {
    "conv1.w": "conv1.weight",
    "bn1.w": "bn1.weight",
    "bn1.b": "bn1.bias",
    "bn1.running_mean": "bn1.running_mean",
    "bn1.running_var": "bn1.running_var",
    "layer1.seq_0.conv1.w": "layer1.0.conv1.weight",
    "layer1.seq_0.bn1.w": "layer1.0.bn1.weight",
    "layer1.seq_0.bn1.b": "layer1.0.bn1.bias",
    "layer1.seq_0.bn1.running_mean": "layer1.0.bn1.running_mean",
    "layer1.seq_0.bn1.running_var": "layer1.0.bn1.running_var",
    "layer1.seq_0.conv2.w": "layer1.0.conv2.weight",
    "layer1.seq_0.bn2.w": "layer1.0.bn2.weight",
    "layer1.seq_0.bn2.b": "layer1.0.bn2.bias",
    "layer1.seq_0.bn2.running_mean": "layer1.0.bn2.running_mean",
    "layer1.seq_0.bn2.running_var": "layer1.0.bn2.running_var",
    "layer1.seq_0.conv3.w": "layer1.0.conv3.weight",
    "layer1.seq_0.bn3.w": "layer1.0.bn3.weight",
    "layer1.seq_0.bn3.b": "layer1.0.bn3.bias",
    "layer1.seq_0.bn3.running_mean": "layer1.0.bn3.running_mean",
    "layer1.seq_0.bn3.running_var": "layer1.0.bn3.running_var",
    "layer1.seq_0.downsample.seq_0.w": "layer1.0.downsample.0.weight",
    "layer1.seq_0.downsample.seq_1.w": "layer1.0.downsample.1.weight",
    "layer1.seq_0.downsample.seq_1.b": "layer1.0.downsample.1.bias",
    "layer1.seq_0.downsample.seq_1.running_mean": "layer1.0.downsample.1.running_mean",
    "layer1.seq_0.downsample.seq_1.running_var": "layer1.0.downsample.1.running_var",
    "layer1.seq_1.conv1.w": "layer1.1.conv1.weight",
    "layer1.seq_1.bn1.w": "layer1.1.bn1.weight",
    "layer1.seq_1.bn1.b": "layer1.1.bn1.bias",
    "layer1.seq_1.bn1.running_mean": "layer1.1.bn1.running_mean",
    "layer1.seq_1.bn1.running_var": "layer1.1.bn1.running_var",
    "layer1.seq_1.conv2.w": "layer1.1.conv2.weight",
    "layer1.seq_1.bn2.w": "layer1.1.bn2.weight",
    "layer1.seq_1.bn2.b": "layer1.1.bn2.bias",
    "layer1.seq_1.bn2.running_mean": "layer1.1.bn2.running_mean",
    "layer1.seq_1.bn2.running_var": "layer1.1.bn2.running_var",
    "layer1.seq_1.conv3.w": "layer1.1.conv3.weight",
    "layer1.seq_1.bn3.w": "layer1.1.bn3.weight",
    "layer1.seq_1.bn3.b": "layer1.1.bn3.bias",
    "layer1.seq_1.bn3.running_mean": "layer1.1.bn3.running_mean",
    "layer1.seq_1.bn3.running_var": "layer1.1.bn3.running_var",
    "layer1.seq_2.conv1.w": "layer1.2.conv1.weight",
    "layer1.seq_2.bn1.w": "layer1.2.bn1.weight",
    "layer1.seq_2.bn1.b": "layer1.2.bn1.bias",
    "layer1.seq_2.bn1.running_mean": "layer1.2.bn1.running_mean",
    "layer1.seq_2.bn1.running_var": "layer1.2.bn1.running_var",
    "layer1.seq_2.conv2.w": "layer1.2.conv2.weight",
    "layer1.seq_2.bn2.w": "layer1.2.bn2.weight",
    "layer1.seq_2.bn2.b": "layer1.2.bn2.bias",
    "layer1.seq_2.bn2.running_mean": "layer1.2.bn2.running_mean",
    "layer1.seq_2.bn2.running_var": "layer1.2.bn2.running_var",
    "layer1.seq_2.conv3.w": "layer1.2.conv3.weight",
    "layer1.seq_2.bn3.w": "layer1.2.bn3.weight",
    "layer1.seq_2.bn3.b": "layer1.2.bn3.bias",
    "layer1.seq_2.bn3.running_mean": "layer1.2.bn3.running_mean",
    "layer1.seq_2.bn3.running_var": "layer1.2.bn3.running_var",
    "layer2.seq_0.conv1.w": "layer2.0.conv1.weight",
    "layer2.seq_0.bn1.w": "layer2.0.bn1.weight",
    "layer2.seq_0.bn1.b": "layer2.0.bn1.bias",
    "layer2.seq_0.bn1.running_mean": "layer2.0.bn1.running_mean",
    "layer2.seq_0.bn1.running_var": "layer2.0.bn1.running_var",
    "layer2.seq_0.conv2.w": "layer2.0.conv2.weight",
    "layer2.seq_0.bn2.w": "layer2.0.bn2.weight",
    "layer2.seq_0.bn2.b": "layer2.0.bn2.bias",
    "layer2.seq_0.bn2.running_mean": "layer2.0.bn2.running_mean",
    "layer2.seq_0.bn2.running_var": "layer2.0.bn2.running_var",
    "layer2.seq_0.conv3.w": "layer2.0.conv3.weight",
    "layer2.seq_0.bn3.w": "layer2.0.bn3.weight",
    "layer2.seq_0.bn3.b": "layer2.0.bn3.bias",
    "layer2.seq_0.bn3.running_mean": "layer2.0.bn3.running_mean",
    "layer2.seq_0.bn3.running_var": "layer2.0.bn3.running_var",
    "layer2.seq_0.downsample.seq_0.w": "layer2.0.downsample.0.weight",
    "layer2.seq_0.downsample.seq_1.w": "layer2.0.downsample.1.weight",
    "layer2.seq_0.downsample.seq_1.b": "layer2.0.downsample.1.bias",
    "layer2.seq_0.downsample.seq_1.running_mean": "layer2.0.downsample.1.running_mean",
    "layer2.seq_0.downsample.seq_1.running_var": "layer2.0.downsample.1.running_var",
    "layer2.seq_1.conv1.w": "layer2.1.conv1.weight",
    "layer2.seq_1.bn1.w": "layer2.1.bn1.weight",
    "layer2.seq_1.bn1.b": "layer2.1.bn1.bias",
    "layer2.seq_1.bn1.running_mean": "layer2.1.bn1.running_mean",
    "layer2.seq_1.bn1.running_var": "layer2.1.bn1.running_var",
    "layer2.seq_1.conv2.w": "layer2.1.conv2.weight",
    "layer2.seq_1.bn2.w": "layer2.1.bn2.weight",
    "layer2.seq_1.bn2.b": "layer2.1.bn2.bias",
    "layer2.seq_1.bn2.running_mean": "layer2.1.bn2.running_mean",
    "layer2.seq_1.bn2.running_var": "layer2.1.bn2.running_var",
    "layer2.seq_1.conv3.w": "layer2.1.conv3.weight",
    "layer2.seq_1.bn3.w": "layer2.1.bn3.weight",
    "layer2.seq_1.bn3.b": "layer2.1.bn3.bias",
    "layer2.seq_1.bn3.running_mean": "layer2.1.bn3.running_mean",
    "layer2.seq_1.bn3.running_var": "layer2.1.bn3.running_var",
    "layer2.seq_2.conv1.w": "layer2.2.conv1.weight",
    "layer2.seq_2.bn1.w": "layer2.2.bn1.weight",
    "layer2.seq_2.bn1.b": "layer2.2.bn1.bias",
    "layer2.seq_2.bn1.running_mean": "layer2.2.bn1.running_mean",
    "layer2.seq_2.bn1.running_var": "layer2.2.bn1.running_var",
    "layer2.seq_2.conv2.w": "layer2.2.conv2.weight",
    "layer2.seq_2.bn2.w": "layer2.2.bn2.weight",
    "layer2.seq_2.bn2.b": "layer2.2.bn2.bias",
    "layer2.seq_2.bn2.running_mean": "layer2.2.bn2.running_mean",
    "layer2.seq_2.bn2.running_var": "layer2.2.bn2.running_var",
    "layer2.seq_2.conv3.w": "layer2.2.conv3.weight",
    "layer2.seq_2.bn3.w": "layer2.2.bn3.weight",
    "layer2.seq_2.bn3.b": "layer2.2.bn3.bias",
    "layer2.seq_2.bn3.running_mean": "layer2.2.bn3.running_mean",
    "layer2.seq_2.bn3.running_var": "layer2.2.bn3.running_var",
    "layer2.seq_3.conv1.w": "layer2.3.conv1.weight",
    "layer2.seq_3.bn1.w": "layer2.3.bn1.weight",
    "layer2.seq_3.bn1.b": "layer2.3.bn1.bias",
    "layer2.seq_3.bn1.running_mean": "layer2.3.bn1.running_mean",
    "layer2.seq_3.bn1.running_var": "layer2.3.bn1.running_var",
    "layer2.seq_3.conv2.w": "layer2.3.conv2.weight",
    "layer2.seq_3.bn2.w": "layer2.3.bn2.weight",
    "layer2.seq_3.bn2.b": "layer2.3.bn2.bias",
    "layer2.seq_3.bn2.running_mean": "layer2.3.bn2.running_mean",
    "layer2.seq_3.bn2.running_var": "layer2.3.bn2.running_var",
    "layer2.seq_3.conv3.w": "layer2.3.conv3.weight",
    "layer2.seq_3.bn3.w": "layer2.3.bn3.weight",
    "layer2.seq_3.bn3.b": "layer2.3.bn3.bias",
    "layer2.seq_3.bn3.running_mean": "layer2.3.bn3.running_mean",
    "layer2.seq_3.bn3.running_var": "layer2.3.bn3.running_var",
    "layer3.seq_0.conv1.w": "layer3.0.conv1.weight",
    "layer3.seq_0.bn1.w": "layer3.0.bn1.weight",
    "layer3.seq_0.bn1.b": "layer3.0.bn1.bias",
    "layer3.seq_0.bn1.running_mean": "layer3.0.bn1.running_mean",
    "layer3.seq_0.bn1.running_var": "layer3.0.bn1.running_var",
    "layer3.seq_0.conv2.w": "layer3.0.conv2.weight",
    "layer3.seq_0.bn2.w": "layer3.0.bn2.weight",
    "layer3.seq_0.bn2.b": "layer3.0.bn2.bias",
    "layer3.seq_0.bn2.running_mean": "layer3.0.bn2.running_mean",
    "layer3.seq_0.bn2.running_var": "layer3.0.bn2.running_var",
    "layer3.seq_0.conv3.w": "layer3.0.conv3.weight",
    "layer3.seq_0.bn3.w": "layer3.0.bn3.weight",
    "layer3.seq_0.bn3.b": "layer3.0.bn3.bias",
    "layer3.seq_0.bn3.running_mean": "layer3.0.bn3.running_mean",
    "layer3.seq_0.bn3.running_var": "layer3.0.bn3.running_var",
    "layer3.seq_0.downsample.seq_0.w": "layer3.0.downsample.0.weight",
    "layer3.seq_0.downsample.seq_1.w": "layer3.0.downsample.1.weight",
    "layer3.seq_0.downsample.seq_1.b": "layer3.0.downsample.1.bias",
    "layer3.seq_0.downsample.seq_1.running_mean": "layer3.0.downsample.1.running_mean",
    "layer3.seq_0.downsample.seq_1.running_var": "layer3.0.downsample.1.running_var",
    "layer3.seq_1.conv1.w": "layer3.1.conv1.weight",
    "layer3.seq_1.bn1.w": "layer3.1.bn1.weight",
    "layer3.seq_1.bn1.b": "layer3.1.bn1.bias",
    "layer3.seq_1.bn1.running_mean": "layer3.1.bn1.running_mean",
    "layer3.seq_1.bn1.running_var": "layer3.1.bn1.running_var",
    "layer3.seq_1.conv2.w": "layer3.1.conv2.weight",
    "layer3.seq_1.bn2.w": "layer3.1.bn2.weight",
    "layer3.seq_1.bn2.b": "layer3.1.bn2.bias",
    "layer3.seq_1.bn2.running_mean": "layer3.1.bn2.running_mean",
    "layer3.seq_1.bn2.running_var": "layer3.1.bn2.running_var",
    "layer3.seq_1.conv3.w": "layer3.1.conv3.weight",
    "layer3.seq_1.bn3.w": "layer3.1.bn3.weight",
    "layer3.seq_1.bn3.b": "layer3.1.bn3.bias",
    "layer3.seq_1.bn3.running_mean": "layer3.1.bn3.running_mean",
    "layer3.seq_1.bn3.running_var": "layer3.1.bn3.running_var",
    "layer3.seq_2.conv1.w": "layer3.2.conv1.weight",
    "layer3.seq_2.bn1.w": "layer3.2.bn1.weight",
    "layer3.seq_2.bn1.b": "layer3.2.bn1.bias",
    "layer3.seq_2.bn1.running_mean": "layer3.2.bn1.running_mean",
    "layer3.seq_2.bn1.running_var": "layer3.2.bn1.running_var",
    "layer3.seq_2.conv2.w": "layer3.2.conv2.weight",
    "layer3.seq_2.bn2.w": "layer3.2.bn2.weight",
    "layer3.seq_2.bn2.b": "layer3.2.bn2.bias",
    "layer3.seq_2.bn2.running_mean": "layer3.2.bn2.running_mean",
    "layer3.seq_2.bn2.running_var": "layer3.2.bn2.running_var",
    "layer3.seq_2.conv3.w": "layer3.2.conv3.weight",
    "layer3.seq_2.bn3.w": "layer3.2.bn3.weight",
    "layer3.seq_2.bn3.b": "layer3.2.bn3.bias",
    "layer3.seq_2.bn3.running_mean": "layer3.2.bn3.running_mean",
    "layer3.seq_2.bn3.running_var": "layer3.2.bn3.running_var",
    "layer3.seq_3.conv1.w": "layer3.3.conv1.weight",
    "layer3.seq_3.bn1.w": "layer3.3.bn1.weight",
    "layer3.seq_3.bn1.b": "layer3.3.bn1.bias",
    "layer3.seq_3.bn1.running_mean": "layer3.3.bn1.running_mean",
    "layer3.seq_3.bn1.running_var": "layer3.3.bn1.running_var",
    "layer3.seq_3.conv2.w": "layer3.3.conv2.weight",
    "layer3.seq_3.bn2.w": "layer3.3.bn2.weight",
    "layer3.seq_3.bn2.b": "layer3.3.bn2.bias",
    "layer3.seq_3.bn2.running_mean": "layer3.3.bn2.running_mean",
    "layer3.seq_3.bn2.running_var": "layer3.3.bn2.running_var",
    "layer3.seq_3.conv3.w": "layer3.3.conv3.weight",
    "layer3.seq_3.bn3.w": "layer3.3.bn3.weight",
    "layer3.seq_3.bn3.b": "layer3.3.bn3.bias",
    "layer3.seq_3.bn3.running_mean": "layer3.3.bn3.running_mean",
    "layer3.seq_3.bn3.running_var": "layer3.3.bn3.running_var",
    "layer3.seq_4.conv1.w": "layer3.4.conv1.weight",
    "layer3.seq_4.bn1.w": "layer3.4.bn1.weight",
    "layer3.seq_4.bn1.b": "layer3.4.bn1.bias",
    "layer3.seq_4.bn1.running_mean": "layer3.4.bn1.running_mean",
    "layer3.seq_4.bn1.running_var": "layer3.4.bn1.running_var",
    "layer3.seq_4.conv2.w": "layer3.4.conv2.weight",
    "layer3.seq_4.bn2.w": "layer3.4.bn2.weight",
    "layer3.seq_4.bn2.b": "layer3.4.bn2.bias",
    "layer3.seq_4.bn2.running_mean": "layer3.4.bn2.running_mean",
    "layer3.seq_4.bn2.running_var": "layer3.4.bn2.running_var",
    "layer3.seq_4.conv3.w": "layer3.4.conv3.weight",
    "layer3.seq_4.bn3.w": "layer3.4.bn3.weight",
    "layer3.seq_4.bn3.b": "layer3.4.bn3.bias",
    "layer3.seq_4.bn3.running_mean": "layer3.4.bn3.running_mean",
    "layer3.seq_4.bn3.running_var": "layer3.4.bn3.running_var",
    "layer3.seq_5.conv1.w": "layer3.5.conv1.weight",
    "layer3.seq_5.bn1.w": "layer3.5.bn1.weight",
    "layer3.seq_5.bn1.b": "layer3.5.bn1.bias",
    "layer3.seq_5.bn1.running_mean": "layer3.5.bn1.running_mean",
    "layer3.seq_5.bn1.running_var": "layer3.5.bn1.running_var",
    "layer3.seq_5.conv2.w": "layer3.5.conv2.weight",
    "layer3.seq_5.bn2.w": "layer3.5.bn2.weight",
    "layer3.seq_5.bn2.b": "layer3.5.bn2.bias",
    "layer3.seq_5.bn2.running_mean": "layer3.5.bn2.running_mean",
    "layer3.seq_5.bn2.running_var": "layer3.5.bn2.running_var",
    "layer3.seq_5.conv3.w": "layer3.5.conv3.weight",
    "layer3.seq_5.bn3.w": "layer3.5.bn3.weight",
    "layer3.seq_5.bn3.b": "layer3.5.bn3.bias",
    "layer3.seq_5.bn3.running_mean": "layer3.5.bn3.running_mean",
    "layer3.seq_5.bn3.running_var": "layer3.5.bn3.running_var",
    "layer4.seq_0.conv1.w": "layer4.0.conv1.weight",
    "layer4.seq_0.bn1.w": "layer4.0.bn1.weight",
    "layer4.seq_0.bn1.b": "layer4.0.bn1.bias",
    "layer4.seq_0.bn1.running_mean": "layer4.0.bn1.running_mean",
    "layer4.seq_0.bn1.running_var": "layer4.0.bn1.running_var",
    "layer4.seq_0.conv2.w": "layer4.0.conv2.weight",
    "layer4.seq_0.bn2.w": "layer4.0.bn2.weight",
    "layer4.seq_0.bn2.b": "layer4.0.bn2.bias",
    "layer4.seq_0.bn2.running_mean": "layer4.0.bn2.running_mean",
    "layer4.seq_0.bn2.running_var": "layer4.0.bn2.running_var",
    "layer4.seq_0.conv3.w": "layer4.0.conv3.weight",
    "layer4.seq_0.bn3.w": "layer4.0.bn3.weight",
    "layer4.seq_0.bn3.b": "layer4.0.bn3.bias",
    "layer4.seq_0.bn3.running_mean": "layer4.0.bn3.running_mean",
    "layer4.seq_0.bn3.running_var": "layer4.0.bn3.running_var",
    "layer4.seq_0.downsample.seq_0.w": "layer4.0.downsample.0.weight",
    "layer4.seq_0.downsample.seq_1.w": "layer4.0.downsample.1.weight",
    "layer4.seq_0.downsample.seq_1.b": "layer4.0.downsample.1.bias",
    "layer4.seq_0.downsample.seq_1.running_mean": "layer4.0.downsample.1.running_mean",
    "layer4.seq_0.downsample.seq_1.running_var": "layer4.0.downsample.1.running_var",
    "layer4.seq_1.conv1.w": "layer4.1.conv1.weight",
    "layer4.seq_1.bn1.w": "layer4.1.bn1.weight",
    "layer4.seq_1.bn1.b": "layer4.1.bn1.bias",
    "layer4.seq_1.bn1.running_mean": "layer4.1.bn1.running_mean",
    "layer4.seq_1.bn1.running_var": "layer4.1.bn1.running_var",
    "layer4.seq_1.conv2.w": "layer4.1.conv2.weight",
    "layer4.seq_1.bn2.w": "layer4.1.bn2.weight",
    "layer4.seq_1.bn2.b": "layer4.1.bn2.bias",
    "layer4.seq_1.bn2.running_mean": "layer4.1.bn2.running_mean",
    "layer4.seq_1.bn2.running_var": "layer4.1.bn2.running_var",
    "layer4.seq_1.conv3.w": "layer4.1.conv3.weight",
    "layer4.seq_1.bn3.w": "layer4.1.bn3.weight",
    "layer4.seq_1.bn3.b": "layer4.1.bn3.bias",
    "layer4.seq_1.bn3.running_mean": "layer4.1.bn3.running_mean",
    "layer4.seq_1.bn3.running_var": "layer4.1.bn3.running_var",
    "layer4.seq_2.conv1.w": "layer4.2.conv1.weight",
    "layer4.seq_2.bn1.w": "layer4.2.bn1.weight",
    "layer4.seq_2.bn1.b": "layer4.2.bn1.bias",
    "layer4.seq_2.bn1.running_mean": "layer4.2.bn1.running_mean",
    "layer4.seq_2.bn1.running_var": "layer4.2.bn1.running_var",
    "layer4.seq_2.conv2.w": "layer4.2.conv2.weight",
    "layer4.seq_2.bn2.w": "layer4.2.bn2.weight",
    "layer4.seq_2.bn2.b": "layer4.2.bn2.bias",
    "layer4.seq_2.bn2.running_mean": "layer4.2.bn2.running_mean",
    "layer4.seq_2.bn2.running_var": "layer4.2.bn2.running_var",
    "layer4.seq_2.conv3.w": "layer4.2.conv3.weight",
    "layer4.seq_2.bn3.w": "layer4.2.bn3.weight",
    "layer4.seq_2.bn3.b": "layer4.2.bn3.bias",
    "layer4.seq_2.bn3.running_mean": "layer4.2.bn3.running_mean",
    "layer4.seq_2.bn3.running_var": "layer4.2.bn3.running_var",
    "fc1.w": "fc1.weight",
    "fc1.b": "fc1.bias",
}

class TorchBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(TorchBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class TorchBottleneck(nn.Module):
    # pylint: disable=missing-function-docstring, abstract-method
    """torch BottleNeck"""
    expansion = 4
    def __init__(self, inplanes, planes, stride):
        super(TorchBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding=1,
                               groups=1,
                               dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or inplanes != planes * TorchBottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes * TorchBottleneck.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * TorchBottleneck.expansion))
        else:
            self.downsample = None

    def forward(self, x):  # pylint: disable=arguments-differ
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class TorchResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(TorchResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y_true):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        y_pred = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss


def _torch_resnet(block, layers, **kwargs):
    model = TorchResNet(block, layers, **kwargs)
    return model


def torch_resnet18(**kwargs):
    return _torch_resnet(TorchBasicBlock, [2, 2, 2, 2], **kwargs)


def torch_resnet50(**kwargs):
    return _torch_resnet(TorchBottleneck, [3, 4, 6, 3], **kwargs)


class MNMBasicBlock(raf.Model):
    expansion = 1

    def build(self, inplanes, planes, stride=1):
        self.conv1 = Conv2d(inplanes,
                            planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv2d(planes,
                            planes,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
        self.bn2 = BatchNorm(planes)
        self.stride = stride
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = Sequential(
                Conv2d(inplanes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm(self.expansion*planes)
            )
        else:
            self.downsample = Sequential()

    @raf.model.trace
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = raf.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out = raf.add(out, identity)
        out = raf.relu(out)
        return out


class MNMBottleneck(raf.Model):
    """ BottleNeck"""
    # pylint: disable=missing-function-docstring
    expansion = 4

    def build(self, inplanes, planes, stride):
        self.conv1 = Conv2d(inplanes,
                            planes,
                            kernel_size=1,
                            stride=1,
                            bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv2d(planes,
                            planes,
                            kernel_size=3,
                            stride=stride,
                            bias=False,
                            padding=1,
                            groups=1,
                            dilation=1)
        self.bn2 = BatchNorm(planes)
        self.conv3 = Conv2d(planes,
                            planes * self.expansion,
                            kernel_size=1,
                            stride=1,
                            bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        if stride != 1 or inplanes != planes * MNMBottleneck.expansion:
            self.downsample = Sequential(
                Conv2d(inplanes,
                       planes * MNMBottleneck.expansion,
                       kernel_size=1,
                       stride=stride,
                       bias=False),
                BatchNorm(planes * MNMBottleneck.expansion))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = raf.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = raf.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = raf.add(out, identity)
        out = raf.relu(out)
        return out


class MNMResNet(raf.Model):
    """ ResNet"""
    # pylint: disable=missing-function-docstring, too-many-instance-attributes

    def build(self, block, num_blocks, num_classes=1000):
        self.num_blocks = num_blocks
        self.inplanes = 64
        self.conv1 = Conv2d(3,
                            self.inplanes,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
        self.bn1 = BatchNorm(self.inplanes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc1 = Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return Sequential(*layers)

    @raf.model.trace
    def forward_infer(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = raf.relu(x)
        x = raf.max_pool2d(x, kernel=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = raf.adaptive_avg_pool2d(x, (1, 1))
        x = raf.batch_flatten(x)
        x = self.fc1(x)
        return x

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss


def _raf_resnet(block, layers, **kwargs):
    model = MNMResNet(block, layers, **kwargs)
    return model


def raf_resnet18(**kwargs):
    return _raf_resnet(MNMBasicBlock, [2, 2, 2, 2], **kwargs)


def raf_resnet50(**kwargs):
    return _raf_resnet(MNMBottleneck, [3, 4, 6, 3], **kwargs)


def init(m_model, t_model, param_map, device="cuda"):
    for m_param, t_param in param_map.items():
        set_param(m_model, m_param, t2m_param(t_model.state_dict()[t_param], device=device))


def check_params(m_model, t_model, params, atol=1e-5, rtol=1e-5):
    for m_param, t_param in params.items():
        m_value = m_model.state()[m_param]
        t_value = t_model.state_dict()[t_param]
        check(m_value, t_value, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", get_device_list())
def test_r18_v1_cifar(device):
    t_model = torch_resnet18(num_classes=10)
    m_model = raf_resnet18(num_classes=10)
    init(m_model, t_model, resnet18_param_map, device)
    m_x, t_x = randn_torch([2, 3, 32, 32], requires_grad=True, device=device)
    m_y, t_y = one_hot_torch(batch_size=2, num_classes=10, device=device)
    m_model.train_mode()
    m_model.to(device=device)
    m_loss = m_model(m_x, m_y)
    t_model.train()
    t_model.to(device)
    t_loss = t_model(t_x, t_y)
    check(m_loss, t_loss, atol=1e-2, rtol=1e-2)
    check_params(m_model, t_model, atol=1e-2, rtol=1e-2, params=resnet18_param_map)

# TODO(@XIAO-XIA): complete the train test after merging SGD and Fusion.


@pytest.mark.parametrize("device", get_device_list())
def test_r50_v1(device):
    t_model = torch_resnet50(num_classes=1000)
    m_model = raf_resnet50(num_classes=1000)
    init(m_model, t_model, resnet50_param_map, device)
    m_x, t_x = randn_torch([1, 3, 224, 224], requires_grad=True, device=device)
    m_y, t_y = one_hot_torch(batch_size=1, num_classes=1000, device=device)
    m_model.train_mode()
    m_model.to(device=device)
    m_loss = m_model(m_x, m_y)
    t_model.train()
    t_model.to(device)
    t_loss = t_model(t_x, t_y)
    check(m_loss, t_loss, atol=1e-2, rtol=1e-2)
    check_params(m_model, t_model, atol=1e-2, rtol=1e-2, params=resnet50_param_map)


if __name__ == "__main__":
    pytest.main([__file__])
