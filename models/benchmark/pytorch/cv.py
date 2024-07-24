"""Register computer vision models from PyTorch."""
# pylint: disable=not-callable, missing-function-docstring, unused-argument
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from ..logger import get_logger
from ..registry import reg_model
from .torch_bencher import TorchBencher
from .utils import one_hot_torch, randn_torch, to_torch_dev, init_deepspeed_moe_pt, ParallelModelLoader
from ..utils import fix_seed

logger = get_logger("PyTorch-CV")  # pylint: disable=invalid-name

def image_classify_common(
    model_name, batch_size, image_size, dtype="float32", include_orig_model=False
):
    """The utility of processing image classification models.

    Parameters
    ----------
    model_name: str
         The model name in torchvision.

    batch_size: int
        Batch size.

    image_size: Optional[Tuple[int, int]]
        The image size. If not present, (224, 224) will be used.

    dtype: str
        The data type. Default is float32.

    include_orig_model: bool
        Whether to include the original model as the reference.

    Returns
    -------
    mod_n_shape: Tuple[raf.Model, Tuple[int, int, int, int]]
        The converted model and input shape.
    """
    fix_seed(42)
    image_size = image_size if image_size is not None else (224, 224)
    input_shape = (batch_size, 3, *image_size)
    with ParallelModelLoader():
        t_model = getattr(torchvision.models, model_name)(pretrained=True)
    t_x = randn_torch(input_shape, dtype=dtype)
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_dy = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=1000)

    bencher = TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue)
    torch.cuda.empty_cache()
    return bencher


@reg_model("torch")
def test_fc(batch_size, image_size, dtype="float32", include_orig_model=False):
    fix_seed(42)
    class FC(torch.nn.Module):
        """Single FC model."""

        def __init__(self, image_size):
            super().__init__()
            self.fc1 = torch.nn.Linear(image_size, 10)

        def forward(self, x):
            return self.fc1(x)

    image_size = image_size[0] * image_size[1] if image_size is not None else 28 * 28
    t_model = FC(image_size)
    input_shape = (batch_size, image_size)

    t_x = randn_torch(input_shape, dtype=dtype)
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_dy = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=10)

    bencher = TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue)
    torch.cuda.empty_cache()
    return bencher


@reg_model("torch")
def lenet5(batch_size, image_size, dtype="float32", include_orig_model=False):
    fix_seed(42)
    class LeNet5(torch.nn.Module):
        """LetNet-5 implementation in PyTorch.
        Source: https://github.com/bentrevett/pytorch-image-classification/blob/master/2_lenet.ipynb
        """

        def __init__(self, image_size):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
            self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
            size = ((image_size - 4) // 2 - 4) // 2
            self.fc_1 = torch.nn.Linear(16 * size * size, 120)
            self.fc_2 = torch.nn.Linear(120, 84)
            self.fc_3 = torch.nn.Linear(84, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=2)
            x = torch.nn.functional.relu(x)
            x = self.conv2(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=2)
            x = torch.nn.functional.relu(x)
            x = x.view(x.shape[0], -1)
            x = self.fc_1(x)
            x = torch.nn.functional.relu(x)
            x = self.fc_2(x)
            x = torch.nn.functional.relu(x)
            x = self.fc_3(x)
            return x

    image_size = image_size if image_size is not None else (28, 28)
    t_model = LeNet5(image_size[0])
    input_shape = (batch_size, 3, *image_size)

    t_x = randn_torch(input_shape, dtype=dtype)
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_dy = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=10)

    bencher = TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue)
    torch.cuda.empty_cache()
    return bencher


@reg_model("torch")
def resnet18(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("resnet18", batch_size, image_size, dtype, include_orig_model)


@reg_model("torch")
def resnet50(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("resnet50", batch_size, image_size, dtype, include_orig_model)


@reg_model("torch")
def vgg11(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("vgg11", batch_size, image_size, dtype, include_orig_model)

@reg_model("torch")
def vgg16(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("vgg16", batch_size, image_size, dtype, include_orig_model)

@reg_model("torch")
def mobilenet_v2(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("mobilenet_v2", batch_size, image_size, dtype, include_orig_model)


@reg_model("torch")
def squeezenet1_0(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("squeezenet1_0", batch_size, image_size, dtype, include_orig_model)


@reg_model("torch")
def densenet121(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("densenet121", batch_size, image_size, dtype, include_orig_model)


@reg_model("torch")
def maskrcnn_resnet50(batch_size, image_size, dtype="float32", include_orig_model=False):
    fix_seed(42)
    image_size = image_size if image_size is not None else (300, 300)
    input_shape = (batch_size, 3, *image_size)
    with ParallelModelLoader():
        t_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_x = randn_torch(input_shape, dtype=dtype)
    t_dy = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=10)

    bencher = (
        TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue)
    )
    torch.cuda.empty_cache()
    return bencher

@reg_model("torch")
def resnext101(batch_size, image_size, dtype, include_orig_model):
    fix_seed(42)
    image_size = image_size if image_size is not None else (224, 224)
    input_shape = (batch_size, 3, *image_size)
    with ParallelModelLoader():
        t_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
    t_x = randn_torch(input_shape, dtype=dtype)
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_dy = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=1000)

    bencher = TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue)
    torch.cuda.empty_cache()
    return bencher

@reg_model("torch")
def resnext50(batch_size, image_size, dtype, include_orig_model):
    fix_seed(42)
    image_size = image_size if image_size is not None else (224, 224)
    input_shape = (batch_size, 3, *image_size)
    with ParallelModelLoader():
        t_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    t_x = randn_torch(input_shape, dtype=dtype)
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_dy = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False)
    t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=1000)

    bencher = TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue)
    torch.cuda.empty_cache()
    return bencher

@reg_model("torch")
def test_cv_moe(batch_size, image_size, dtype="float32", include_orig_model=False, num_experts=None, device="cuda"):
    fix_seed(42)
    # init ds moe environment
    import deepspeed
    num_experts, world_size = init_deepspeed_moe_pt(num_experts=num_experts)

    image_size = image_size if image_size is not None else (32, 32)
    if image_size != (32, 32):
        raise ValueError("test_moe only supports input image size of 32x32.")
    input_shape = (batch_size, 3, *image_size)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 84)
            self.fc3 = deepspeed.moe.layer.MoE(
                hidden_size=84,
                expert=self.fc3,
                num_experts=num_experts,
                k=1,
                min_capacity=4,
                use_rts=False)
            self.fc4 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x, _, _ = self.fc3(x)
            x = self.fc4(x)
            return x

    t_model = Net()
    t_model = t_model.to(to_torch_dev(device))
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_x = randn_torch(input_shape, dtype=dtype).to(to_torch_dev(device))
    t_dy = randn_torch((), std=0.0, mean=1.0, dtype=dtype, requires_grad=False).to(to_torch_dev(device))
    t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=10).to(to_torch_dev(device))
    bencher = (
        TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue)
    )
    torch.cuda.empty_cache()
    return bencher
