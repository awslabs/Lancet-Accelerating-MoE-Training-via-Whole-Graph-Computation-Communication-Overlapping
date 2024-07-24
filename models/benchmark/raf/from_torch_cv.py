"""Register models that are converted from torchvision."""
# pylint: disable=protected-access, missing-function-docstring
import raf
from raf.frontend import from_pytorch

from .raf_bencher import RAFBencher
from .utils import patch_deepspeed_moe_conversion_map
from ..logger import get_logger
from ..registry import reg_model, get_model_bencher

logger = get_logger("RAF-TorchCV")  # pylint: disable=invalid-name


def image_classify_common(
    model_name, batch_size, image_size, dtype="float32", include_orig_model=False, **kwargs
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
    device = kwargs.get("device", "cpu")
    ref_bencher = get_model_bencher("torch", model_name, batch_size, image_size, dtype, **kwargs)

    t_model = ref_bencher.model
    input_shape = ref_bencher.input_shape

    if "moe" in model_name:
        custom_convert_map = patch_deepspeed_moe_conversion_map()
    else:
        custom_convert_map = None
    try:
        m_model = from_pytorch(t_model, {"input0": ((input_shape, dtype))}, 
                               device=device, custom_convert_map=custom_convert_map)
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError("Failed to convert model to RAF: %s" % (str(err)))

    # Making RAF buffers
    m_x = raf.array(ref_bencher.args[0].cpu().numpy(), device=device)
    m_dy = raf.array(ref_bencher.dy.cpu().numpy(), device=device)
    m_ytrue = raf.array(ref_bencher.y_true.cpu().numpy(), device=device)

    if not include_orig_model:
        del ref_bencher.model
        del ref_bencher.args
        del ref_bencher.dy
        del ref_bencher.y_true
        ref_bencher = None
    return RAFBencher(m_model, input_shape, [m_x], m_dy, m_ytrue, ref_bencher=ref_bencher)


@reg_model("raf")
def test_fc(batch_size, image_size, dtype="float32", include_orig_model=False):
    return image_classify_common("test_fc", batch_size, image_size, dtype, include_orig_model)


@reg_model("raf")
def lenet5(batch_size, image_size, dtype="float32", include_orig_model=False):
    return image_classify_common("lenet5", batch_size, image_size, dtype, include_orig_model)


@reg_model("raf")
def resnet18(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("resnet18", batch_size, image_size, dtype, include_orig_model)


@reg_model("raf")
def resnet50(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("resnet50", batch_size, image_size, dtype, include_orig_model)


@reg_model("raf")
def vgg11(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("vgg11", batch_size, image_size, dtype, include_orig_model)

@reg_model("raf")
def vgg16(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("vgg16", batch_size, image_size, dtype, include_orig_model)

@reg_model("raf")
def mobilenet_v2(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("mobilenet_v2", batch_size, image_size, dtype, include_orig_model)


@reg_model("raf")
def squeezenet1_0(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("squeezenet1_0", batch_size, image_size, dtype, include_orig_model)


@reg_model("raf")
def densenet121(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("densenet121", batch_size, image_size, dtype, include_orig_model)


@reg_model("raf")
def maskrcnn_resnet50(batch_size, image_size, dtype="float32", include_orig_model=False):
    return image_classify_common(
        "maskrcnn_resnet50", batch_size, image_size, dtype, include_orig_model
    )

@reg_model("raf")
def resnext101(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("resnext101", batch_size, image_size, dtype, include_orig_model)

@reg_model("raf")
def resnext50(batch_size, image_size, dtype, include_orig_model):
    return image_classify_common("resnext50", batch_size, image_size, dtype, include_orig_model)

@reg_model("raf")
def test_cv_moe(batch_size, image_size, dtype="float32", include_orig_model=False, num_experts=None, device="cuda"):
    return image_classify_common(
        "test_cv_moe", batch_size, image_size, dtype, include_orig_model, device=device, num_experts=num_experts
    )