"""Register models that are converted from Huggingface transformers."""
# pylint: disable=protected-access, missing-function-docstring
import raf
from raf.frontend import from_pytorch

from raf._ffi.pass_ import InferType, ExprAppend, ExtractBinding
from raf._core.module import IRModule
from raf._core.ndarray import Symbol
from raf.frontend.model import FrameworkModel
from tvm import relay

from .raf_bencher import RAFBencher
from .utils import patch_deepspeed_moe_conversion_map
from ..logger import get_logger
from ..registry import reg_model, get_model_bencher
from ..dataset import load_wikitext2, transform_batch

logger = get_logger("RAF-TorchNLP")  # pylint: disable=invalid-name


def get_raf_func_output_var(func):
    """A helper function to get the output Var of the given function."""
    body = func.body
    while not isinstance(body, relay.Var):
        if isinstance(body, relay.Let):
            body = body.body
        else:
            raise NotImplementedError("Not supported type: ", type(body))
    return body


def transformer_common(model_name, batch_size, seq_length, dtype, include_orig_model, **kwargs):
    """The utility of processing transformer models.

    Parameters
    ----------
    model_name: str
        The model name.

    batch_size: int
        Batch size.

    seq_length: Optional[int]
        The sequence length. If None, 128 will be used.

    dtype: str
        The data type. Default is float32.

    include_orig_model: bool
        Whether to include the original model as the reference.

    Returns
    -------
    mod_n_shape: Tuple[raf.Model, Tuple[int, int]]
        The converted model and input shape.
    """
    device = kwargs.get("device", "cpu")
    input_is_float = kwargs.pop("input_is_float", False)

    if "moe" in model_name:
        kwargs["moe_framework"] = 'raf'
    
    gate_type = kwargs.get("moe_gate_type", "switch")

    ref_bencher = get_model_bencher("torch", model_name, batch_size, seq_length, dtype, **kwargs)

    t_model = ref_bencher.model

    input_shape = ref_bencher.input_shape
    reshape_output = ref_bencher.kwargs.get("reshape_output", None)

    if "moe" in model_name:
        custom_convert_map = patch_deepspeed_moe_conversion_map(gate_type=gate_type)
    else:
        custom_convert_map = None

    try:
        ref_dataloader = ref_bencher.dataloader
        ref_batch = next(iter(ref_dataloader))
        inputs, labels = transform_batch(ref_batch)
        m_x = raf.ndarray(inputs.numpy(), device=device)
        m_model = from_pytorch(t_model, {"input_ids": (input_shape, "float32" if input_is_float else "int64")},
                               device=device, custom_convert_map=custom_convert_map)
        record = m_model._internal(m_x)
        mod = record.mod
        mod = InferType()(mod)
        func = mod["main"]
        ret_var = get_raf_func_output_var(func)
        if isinstance(ret_var.checked_type, relay.TupleType):
            ret = Symbol.from_expr(ret_var)
            ret = ret[0]
            ret = ExtractBinding(ret._Symbol__handle, [ret_var])
            new_body = ExprAppend(func.body, ret)
            new_func = relay.Function(func.params, new_body)
            new_mod = IRModule.from_expr(new_func)
            m_model = FrameworkModel(
                new_mod,
                new_mod,
                m_model._FrameworkModel__arg_params,
                m_model._FrameworkModel__aux_params,
                param_attrs = m_model._FrameworkModel__param_attrs
            )
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError("Failed to convert model to RAF: %s" % (str(err)))

    m_dy = raf.array(ref_bencher.dy.cpu().numpy(), device=device)
    # m_ytrue = raf.array(ref_bencher.y_true.cpu().numpy(), device=device)

    if not include_orig_model:
        del ref_bencher.model
        # del ref_bencher.args
        del ref_bencher.dy
        # del ref_bencher.y_true
        ref_bencher = None

    return RAFBencher(
        m_model,
        input_shape,
        load_wikitext2(batch_size, seq_length),
        m_dy,
        ref_bencher=ref_bencher,
        reshape_output=reshape_output,
        dtype=dtype,
    )


@reg_model("raf")
def bert_base_classify(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common(
        "bert_base_classify", batch_size, seq_length, dtype, include_orig_model
    )

@reg_model("raf")
def bert_base_classify_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, moe_expert_interval=2):
    return transformer_common("bert_base_classify_moe", batch_size, seq_length, dtype, include_orig_model, 
                              device=device, num_experts=num_experts, moe_expert_interval=moe_expert_interval)

@reg_model("raf")
def bert_test_classify(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common(
        "bert_test_classify", batch_size, seq_length, dtype, include_orig_model
    )


@reg_model("raf")
def bert_large_classify(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common(
        "bert_large_classify", batch_size, seq_length, dtype, include_orig_model
    )

@reg_model("raf")
def bert_large_test_classify(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common(
        "bert_large_test_classify", batch_size, seq_length, dtype, include_orig_model
    )

@reg_model("raf")
def gpt2_classify(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("gpt2_classify", batch_size, seq_length, dtype, include_orig_model)

@reg_model("raf")
def gpt2_test_classify(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("gpt2_test_classify", batch_size, seq_length, dtype, include_orig_model)


@reg_model("raf")
def bert_base_mlm(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("bert_base_mlm", batch_size, seq_length, dtype, include_orig_model)


@reg_model("raf")
def bert_large_mlm(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("bert_large_mlm", batch_size, seq_length, dtype, include_orig_model)


@reg_model("raf")
def bert_test_mlm(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("bert_test_mlm", batch_size, seq_length, dtype, include_orig_model)


@reg_model("raf")
def gpt2(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("gpt2", batch_size, seq_length, dtype, include_orig_model)

@reg_model("raf")
def gpt2_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", 
             num_experts=None, moe_expert_interval=2, num_layers=12, d_model=768, n_head=12, 
             check_correctness=False, moe_gate_type="switch"):
    return transformer_common("gpt2_moe", batch_size, seq_length, dtype, include_orig_model, 
                              device=device, num_experts=num_experts, moe_expert_interval=moe_expert_interval,
                              num_layers=num_layers, d_model=d_model, n_head=n_head, check_correctness=check_correctness,
                              moe_gate_type=moe_gate_type)

@reg_model("raf")
def vit(batch_size, seq_length, dtype, include_orig_model, device="cuda"):
    return transformer_common("vit", batch_size, seq_length, dtype, include_orig_model, input_is_float=True)

@reg_model("raf")
def vit_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, moe_expert_interval=2):
    return transformer_common("vit_moe", batch_size, seq_length, dtype, include_orig_model, 
                                device=device, num_experts=num_experts, moe_expert_interval=moe_expert_interval, 
                                input_is_float=True)

@reg_model("raf")
def vit_test_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, moe_expert_interval=2):
    return transformer_common("vit_test_moe", batch_size, seq_length, dtype, include_orig_model, 
                                device=device, num_experts=num_experts, moe_expert_interval=moe_expert_interval, 
                                input_is_float=True)

@reg_model("raf")
def gpt2_test(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("gpt2_test", batch_size, seq_length, dtype, include_orig_model)

@reg_model("raf")
def gpt2_test_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, moe_expert_interval=1, check_correctness=False, **kwargs):
    return transformer_common("gpt2_test_moe", batch_size, seq_length, dtype, include_orig_model, 
                              device=device, num_experts=num_experts, moe_expert_interval=moe_expert_interval, check_correctness=check_correctness)

@reg_model("raf")
def partitioned_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, partition_number=2):
    return transformer_common("partitioned_moe", batch_size, seq_length, dtype, include_orig_model, 
                              device=device, num_experts=num_experts, partition_number=partition_number, input_is_float=True)

@reg_model("raf")
def bart_base(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("bart_base", batch_size, seq_length, dtype, include_orig_model)


@reg_model("raf")
def roberta_base(batch_size, seq_length, dtype, include_orig_model):
    return transformer_common("roberta_base", batch_size, seq_length, dtype, include_orig_model)
