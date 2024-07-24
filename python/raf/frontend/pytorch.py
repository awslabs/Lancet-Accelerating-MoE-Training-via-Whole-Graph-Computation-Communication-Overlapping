# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend that converts PyTorch models to RAF models via Relay."""
# pylint: disable=too-many-locals
from collections import OrderedDict
import os
import hashlib
import torch

from raf import distributed as dist
from .._core.ndarray import ndarray
from .._lib import relay
from .._ffi.pass_ import FromRelay, validate_relay_param_name
from ..frontend.model import FrameworkModel

def to_torch_dev(device_str):
    """Change device string form `cuda(id)` to pytorch style `cuda:id`"""
    import re
    tokens = re.search(r"(\w+).?(\d?)", device_str)
    dev_type = tokens.groups()[0]
    dev_id = int(tokens.groups()[1]) if tokens.groups()[1] else 0
    return "%s:%d" % (dev_type, dev_id)

def trace_model(model, input_type, input_shape, device=None):
    """Trace PyTorch model.

    Parameters
    ----------
    model: torch.nn.Module
        The PyTorch module to be converted.

    input_type: str
        Input type.

    input_shape: Tuple[int, ...]
        Input shape
    
    device: str
        The device where the model is traced

    Returns
    -------
    model: ScriptedModel
        PyTorch scripted model.
    """

    class TraceWrapper(torch.nn.Module):
        """A wrapper to process the forward output. This is required for object detection
        models which have multiple outputs.
        """

        # pylint: disable=missing-function-docstring, abstract-method ,arguments-differ

        # Enforce the output order of object detection models.
        od_model_output_keys = ["boxes", "scores", "labels", "masks"]

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inp):
            out = self.model(inp)
            if isinstance(out, list):
                ordered_outs = [out[0][key] for key in self.od_model_output_keys if key in out[0]]
                return tuple(ordered_outs)
            if isinstance(out, dict):
                return out.to_tuple()
            return out

        @property
        def dtype(self):
            for param in model.parameters():
                if param.dtype.is_floating_point:
                    return param.dtype
            return torch.float32

    def inner(model, input_type, input_shape, device=None):
        """Wrap the tracing process so that we could empty PyTorch CUDA cache afterward."""
        model = TraceWrapper(model)
        model.eval()

        # By default we trace the model on CPU.
        device = "cpu" if device is None else device
        orig_device = device

        # Some float16 ops are only available on GPU.
        if model.dtype != torch.float32:
            if not torch.cuda.is_available():
                raise RuntimeError("Trace PyTorch model with dtype %s requires GPU" % model.dtype)
            if not device.startswith("cuda"):
                # assign a cuda device using dctx
                dctx = dist.get_context()
                device = "cuda(" + str(dctx.local_rank) + ")"

        if input_type.startswith("float"):
            input_data = torch.randn(input_shape, dtype=getattr(torch, input_type), device=to_torch_dev(device))
        else:
            assert input_type.startswith("int64"), "Unsupported input type %s" % input_type
            input_data = torch.randint(10000, input_shape, device=to_torch_dev(device))

        with torch.no_grad():
            model.to(device=to_torch_dev(device))
            model(input_data)
            scripted_model = torch.jit.trace(model, input_data, check_trace=False).eval()

        if not orig_device.startswith("cuda"):
            model.to(device=to_torch_dev(orig_device))
            scripted_model = scripted_model.to(device=to_torch_dev(orig_device))

        return scripted_model

    scripted_model = inner(model, input_type, input_shape, device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return scripted_model


def from_pytorch(model, shape_dict, model_file=None, hash_file=None, device=None, custom_convert_map=None):
    """Load PyTorch model and convert into Meta via Relay.

    Parameters
    ----------
    model: torch.nn.Module
        The PyTorch module to be converted.

    shape_dict: Dict[str, Tuple[Tuple[int, ...], str]]
        A map from input name to its shape and type. Note that we currently only support
        the model with a single input.

    model_file: str
        The file that stores the scripted model

    hash_file: str
        The file that stores the scripted model hash
    
    device: str
        The device that the model should be placed on

    custom_convert_map: Dict[str, Callable[[List[Expr], List[Str]], Expr]]
        custom rules to convert a pytorch op to relay op

    Returns
    -------
    model: FrameworkModel
        The converted FrameworkModel.
    """
    if len(shape_dict) > 1:
        raise RuntimeError(
            "Do not support PyTorch model with multiple inputs (%d) yet" % len(shape_dict)
        )
    input_name, (input_shape, input_type) = list(shape_dict.items())[0]
    if model_file is not None and hash_file is not None:
        model_hash = hashlib.md5(str(model).encode(encoding="UTF-8")).hexdigest()
        if os.path.exists(model_file) and os.path.exists(hash_file):
            try:
                with open(hash_file, "r") as hashf:
                    mhash = hashf.read()
                    if mhash != model_hash:
                        raise RuntimeError("Hash check failed")
                    scripted_model = torch.jit.load(model_file)
            except:
                raise RuntimeError("Loading scripted model failed")
        else:
            scripted_model = trace_model(model, input_type, input_shape, device=device)
            scripted_model.eval()
            scripted_model.save(model_file)
            with open(hash_file, "w") as hashf:
                hashf.write(model_hash)
    else:
        scripted_model = trace_model(model, input_type, input_shape, device=device)
    shape_list = [(input_name, (input_shape, input_type))]

    py_parameters = scripted_model.named_parameters()
    param_dict = {}
    for name, value in py_parameters:
        param_dict[name] = value
    relay_mod, relay_params = relay.frontend.from_pytorch(scripted_model, shape_list, custom_convert_map=custom_convert_map)
    meta_mod = FromRelay()(relay_mod)
    meta_params = OrderedDict()
    aux_params = OrderedDict()
    for var in relay_mod["main"].params:
        name = var.name_hint
        if name in relay_params:
            array_value = ndarray(relay_params[name].numpy())
            valid_name = validate_relay_param_name(name)
            meta_params[valid_name] = array_value
            if name in param_dict:
                # If a learnable paratmer's requires_grad is False,
                # the training freezes the parameter update.
                if not param_dict[name].requires_grad:
                    aux_params[valid_name] = array_value
            else:
                aux_params[valid_name] = array_value

    pt_expert_parallel_param_names = set()
    for name, param in model.named_parameters():
        param_is_data_parallel = param.allreduce if hasattr(param, "allreduce") else True
        if not param_is_data_parallel:
            pt_expert_parallel_param_names.add(name)

    param_is_parallel = {}
    for var in relay_mod["main"].params:
        name = var.name_hint
        if name.startswith("model."):
            name = name.strip("model.")
        if name in pt_expert_parallel_param_names:
            param_is_parallel[validate_relay_param_name(var.name_hint)] = True

    param_attrs = {"is_expert_parallel": param_is_parallel}
    # relay_params may contain unused parameters, which are not present in meta_params
    assert len(meta_params) <= len(relay_params)
    return FrameworkModel(meta_mod, meta_mod, meta_params, aux_params, param_attrs=param_attrs)
