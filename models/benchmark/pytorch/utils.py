"""Utilities."""
# pylint: disable=not-callable
import re
import os
import sys
import copy

import numpy as np

import torch
import torch.nn.functional as F

from ..logger import get_logger

from contextlib import ContextDecorator

logger = get_logger("Torch")  # pylint: disable=inaalid-name


def to_torch_dev(device_str):
    import re

    """Change device string form `cuda(id)` to pytorch style `cuda:id`"""
    tokens = re.search(r"(\w+).?(\d?)", device_str)
    dev_type = tokens.groups()[0]
    dev_id = int(tokens.groups()[1]) if tokens.groups()[1] else 0
    return "%s:%d" % (dev_type, dev_id)


def get_world_size_from_env():
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))


def get_world_size():
    world_size = os.environ.get("WORLD_SIZE_OVERRIDE", None)
    if world_size is None:
        world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)
    if world_size is None:
        world_size = os.environ.get("REAL_WORLD_SIZE", None)
    if world_size is None:
        world_size = os.environ.get("WORLD_SIZE", None)
    if world_size is None:
        from raf.distributed import get_context

        dctx = get_context()
        world_size = dctx.size
    assert world_size is not None
    world_size = int(world_size)
    return world_size


def get_world_rank_from_env():
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))


def get_local_rank_from_env():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))


class ParallelModelLoader(ContextDecorator):
    def __enter__(self):
        from raf.distributed import get_context

        self.local_rank = get_local_rank_from_env()
        self.dctx = get_context()
        if self.local_rank != 0:
            self.dctx.barrier()
        return self

    def __exit__(self, *exc):
        if self.local_rank == 0:
            self.dctx.barrier()
        return False


def init_torch_ddp():
    world_size = get_world_size_from_env()
    world_rank = get_world_rank_from_env()
    local_rank = get_local_rank_from_env()
    logger.info("Using DDP on device {}.".format(local_rank))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=world_rank
        )
    return world_size, world_rank, local_rank


def init_horovod():
    # disable mpi multithreading to avoid conflict
    os.environ["HOROVOD_MPI_THREADS_DISABLE"] = "1"
    import horovod.torch as hvd

    hvd.init()
    local_rank = hvd.local_rank()
    torch.cuda.set_device(local_rank)
    logger.info("Using Horovod on device {}.".format(local_rank))
    world_size = hvd.size()
    world_rank = hvd.rank()
    return world_size, world_rank, local_rank


def init_deepspeed():
    import deepspeed

    deepspeed.init_distributed()
    local_rank = os.environ.get("LOCAL_RANK", None)
    local_rank = int(local_rank)
    assert (
        local_rank is not None
    ), "Failed to get LOCAL_RANK from env. Is DeepSpeed process group inited?"
    logger.info("Using DeepSpeed engine on device {}.".format(local_rank))
    torch.cuda.set_device(local_rank)
    return local_rank


def init_torch_based_moe_env_(num_experts=None):
    world_size, _, local_rank = init_torch_ddp()
    num_experts = max(2, world_size * 2) if num_experts is None else num_experts
    return num_experts, world_size, local_rank

def init_tutel_moe(num_experts=None, check_correctness=False):
    num_experts, world_size, local_rank = init_torch_based_moe_env_(num_experts)
    logger.info(
        "Using Tutel MoE on {} devices, {} experts in total".format(
            world_size, num_experts
        )
    )
    return num_experts, world_size, local_rank

def init_fastermoe_moe(num_experts=None, check_correctness=False):
    num_experts, world_size, local_rank = init_torch_based_moe_env_(num_experts)
    logger.info(
        "Using FasterMoE on {} devices, {} experts in total".format(
            world_size, num_experts
        )
    )
    return num_experts, world_size, local_rank


def init_deepspeed_moe_pt(num_experts=None, check_correctness=False):
    import deepspeed
    from deepspeed.moe import sharded_moe

    deepspeed.init_distributed()
    world_size = get_world_size()
    if not deepspeed.utils.groups.is_initialized():
        deepspeed.utils.groups.initialize(ep_size=world_size)
    local_rank = os.environ.get("LOCAL_RANK", None)
    local_rank = int(local_rank)
    assert (
        local_rank is not None
    ), "Failed to get LOCAL_RANK from env. Is DeepSpeed process group inited?"

    num_experts = max(2, world_size * 2) if num_experts is None else num_experts
    logger.info(
        "Using DeepSpeed MoE with {} devices, {} experts in total".format(
            world_size, num_experts
        )
    )
    return num_experts, world_size, local_rank


def deepspeed_create_moe_param_groups(model):
    from deepspeed.moe.utils import is_moe_param

    params = {"params": [], "name": "params"}
    moe_params = {"params": [], "moe": True, "name": "moe_params"}

    visited_params = set()

    for module_ in model.modules():
        for n, p in list(module_._parameters.items()):
            if p is not None:
                if p not in visited_params:
                    if is_moe_param(p):
                        moe_params["params"].append(p)
                    else:
                        params["params"].append(p)
                    visited_params.add(p)

    return params, moe_params


def init_deepspeed_moe_raf(num_experts=None, check_correctness=False):
    world_size = get_world_size()
    num_experts = max(2, world_size * 2) if num_experts is None else num_experts
    from raf.distributed import get_context

    dctx = get_context()
    local_rank = dctx.local_rank
    logger.info(
        "Using RAF MoE on {} devices, {} experts in total".format(
            world_size, num_experts
        )
    )
    if check_correctness:
        # also init torch env
        pt_num_experts, pt_world_size, pt_local_rank = init_torch_based_moe_env_(num_experts)
        assert pt_num_experts == num_experts
        assert pt_world_size == world_size
        assert pt_local_rank == local_rank
    return num_experts, world_size, local_rank


def to_torch_dev(device_str):
    """Change device string form `cuda(id)` to pytorch style `cuda:id`"""
    tokens = re.search(r"(\w+).?(\d?)", device_str)
    dev_type = tokens.groups()[0]
    dev_id = int(tokens.groups()[1]) if tokens.groups()[1] else 0
    return "%s:%d" % (dev_type, dev_id)


def randn_torch(
    shape,
    *,
    device="cpu",
    dtype="float32",
    requires_grad=False,
    mean=0.0,
    std=1.0,
    positive=False
):
    """Helper function to generate a torch array"""
    x = np.random.randn(*shape) * std + mean
    if positive:
        x = np.abs(x) + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    t_x = torch.tensor(n_x, requires_grad=requires_grad, device=to_torch_dev(device))
    return t_x


def one_hot_torch(batch_size, num_classes, device="cpu"):
    """Helper function to generate one hot tensors in torch"""
    targets = np.random.randint(0, num_classes, size=batch_size)
    t_x = torch.tensor(targets, requires_grad=False, device=to_torch_dev(device))
    assert list(t_x.shape) == [batch_size]
    return t_x


def get_layers_flattened(module, condition=None):
    children = list(module.children())
    flattened = []
    if not children:
        if condition is None:
            return [module]
        if condition(module):
            return [module]
        else:
            return []
    else:
        for child in children:
            flattened.extend(get_layers_flattened(child, condition=condition))
    return flattened


def turn_off_dropout(model):
    def is_dropout_layer(module):
        return isinstance(module, torch.nn.modules.dropout.Dropout)

    dropout_layers = get_layers_flattened(model, is_dropout_layer)
    for layer in dropout_layers:
        layer.eval()


def ddp_ignore_moe_parameters(model):
    params_to_ignore = []
    for param in model.parameters():
        if hasattr(param, "allreduce") and param.allreduce is False:
            params_to_ignore.append(param)
    model._ddp_params_and_buffers_to_ignore = params_to_ignore


################### Auxilliary models and wrappers ###################


class ModelWithLoss(torch.nn.Module):
    def __init__(self, orig_model, reshape_output):
        super().__init__()
        self.orig_model = orig_model
        self.reshape_output = reshape_output

    def forward(self, y_true, *args):
        t_y = self.orig_model(*args)
        if isinstance(t_y, tuple):
            t_y = t_y[0]
        elif isinstance(t_y, dict):
            assert (
                "logits" in t_y
            ), "Expect ModelingOutputs with logits, but got %s" % type(t_y)
            t_y = t_y["logits"]

        # Reshape output if necessary.
        reshape = self.reshape_output
        t_y = t_y.view(*reshape) if reshape is not None else t_y

        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = torch.nn.functional.nll_loss(t_ypred, y_true)
        if isinstance(t_loss, tuple):
            if hasattr(t_loss[0], "backward"):
                t_loss = t_loss[0]
            else:
                assert hasattr(t_loss[1], "backward")
                t_loss = t_loss[1]
        return t_loss


# counter = 0

class FakeMoeEncodeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, gate, remaining_capacity, capacity_factor):
        data_shape = data.size()
        gate_shape = gate.size()
        capacity_factor = capacity_factor.cpu().item()
        assert capacity_factor == 1.0, "Only support capacity_factor=1.0 for now, but got %f" % capacity_factor
        dim_S = data_shape[0]
        dim_M = data_shape[1]
        dim_E = gate_shape[1]
        dim_C = (dim_S + dim_E - 1) // dim_E
        mask_shape = (dim_S,)
        indices_locations_shape = (2, dim_S)
        out_shape = (dim_E, dim_C, dim_M)
        cap_shape = (dim_E,)
        ctx.dim_S = dim_S
        ctx.dim_M = dim_M
        ctx.dim_E = dim_E
        ctx.dim_C = dim_C
        ctx.data_shape = data_shape
        ctx.gate_shape = gate_shape
        ctx.dtype = data.dtype
        ctx.device = data.device
        if torch._C._get_tracing_state():
            return (
                torch.zeros(*mask_shape, dtype=data.dtype, device=data.device),
                torch.zeros(*indices_locations_shape, dtype=torch.int32, device=data.device),
                torch.zeros(*cap_shape, dtype=torch.int32, device=data.device),
                torch.zeros(*cap_shape, dtype=torch.int32, device=data.device),
                torch.zeros(*out_shape, dtype=data.dtype, device=data.device),
            )
        # output: gates_s, indices_locations, used_capacity, n_elems_per_expert, dispatched_input
        data = data.cpu()
        gate = gate.cpu()
        remaining_capacity = remaining_capacity.cpu()

        indices_s = torch.argmax(gate, axis=1)
        masks_se = torch.eye(dim_E)[indices_s]
        masked_gates_se = gate * masks_se
        masked_gates_s = torch.sum(masked_gates_se, axis=1)
        locations_cumsum = torch.cumsum(masks_se, axis=0).int()
        locations1 = locations_cumsum - 1
        masked_locations1 = locations1 * masks_se
        locations_s_float = torch.sum(masked_locations1, axis=1)
        locations_s = locations_s_float.int()
        dispatched_input = torch.zeros((dim_E, dim_C, dim_M), dtype=data.dtype)
        for i in range(dim_S):
            if locations_s[i].cpu().item() < dim_C and indices_s[i].cpu().item() < dim_E:
                dispatched_input[indices_s[i], locations_s[i], :] = data[i, :]
        out_capused1_e = torch.zeros((dim_E,), dtype=torch.int32)
        elements_per_expert = torch.zeros((dim_E,), dtype=torch.int32)
        for i in range(dim_S):
            loc = min(locations_s[i].cpu().item(), dim_C - 1)
            out_capused1_e[indices_s[i]] = max(out_capused1_e[indices_s[i]].cpu().item(), loc + 1)
            elements_per_expert[indices_s[i]] = (out_capused1_e[indices_s[i]] - remaining_capacity[indices_s[i]]) * dim_M
            if locations_s[i] >= (dim_C - remaining_capacity[indices_s[i]]):
                indices_s[i] = -1
        indices_locations = torch.concat([torch.unsqueeze(indices_s, 0), torch.unsqueeze(locations_s, 0)], axis=0)

        masked_gates_s = masked_gates_s.to(ctx.device)
        indices_locations = indices_locations.to(ctx.device)
        out_capused1_e = out_capused1_e.to(ctx.device)
        elements_per_expert = elements_per_expert.to(ctx.device)
        dispatched_input = dispatched_input.to(ctx.device)
        ctx.save_for_backward(indices_locations)
        ctx.mark_non_differentiable(indices_locations, out_capused1_e, elements_per_expert)
        return (
            masked_gates_s, indices_locations, out_capused1_e, elements_per_expert, dispatched_input,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        # dx: [S, M]
        # dg = [S, E]
        gate_grad, _, _, _, data_grad = grad_output

        gate_grad = gate_grad.cpu()
        data_grad = data_grad.cpu()

        ind_loc = ctx.saved_tensors[0].cpu()

        indices, locations = torch.split(ind_loc, 1, dim=0)
        indices = torch.squeeze(indices)
        locations = torch.squeeze(locations)
        dx = torch.zeros((ctx.dim_S, ctx.dim_M), dtype=ctx.dtype)
        dg = torch.zeros((ctx.dim_S, ctx.dim_E), dtype=ctx.dtype)
        for i in range(ctx.dim_S):
            idx_e = indices[i]
            idx_c = locations[i]
            if idx_e >= 0 and idx_c >= 0:
                dx[i, :] = data_grad[idx_e, idx_c, :]
                dg[i, idx_e] = gate_grad[i]
        # import os
        # import numpy as np
        # global counter
        # if not os.path.exists(f"./moe_encode_grad_inouts_{counter}.npz"):
        #     with open(f"./moe_encode_grad_inouts_{counter}.npz", "wb") as f:
        #         gate_grad_np = gate_grad.numpy()
        #         ind_loc_np = ind_loc.numpy()
        #         data_grad_np = data_grad.numpy()
        #         dx_np = dx.numpy()
        #         dg_np = dg.numpy()
        #         np.savez(f, gate_grad=gate_grad_np, ind_loc=ind_loc_np, data_grad=data_grad_np, dx=dx_np, dg=dg_np)
        #         counter += 1
        return (
            dx.to(ctx.device),
            dg.to(ctx.device),
            None,
            None
        )


class FakeMoeDecodeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, gate, indices_locations):  # type: ignore
        data_shape = data.size()  # [E, C, M]
        gate_shape = gate.size()  # [S]
        dim_E = data_shape[0]
        dim_C = data_shape[1]
        dim_M = data_shape[2]
        dim_S = gate_shape[0]

        ctx.dim_E = dim_E
        ctx.dim_C = dim_C
        ctx.dim_M = dim_M
        ctx.dim_S = dim_S
        ctx.data_shape = data_shape
        ctx.gate_shape = gate_shape
        ctx.mask_shape = (2, dim_S)
        ctx.device = data.device
        ctx.dtype = data.dtype

        if torch._C._get_tracing_state():
            out_shape = (dim_S, dim_M)
            return torch.zeros(*out_shape, dtype=data.dtype, device=data.device)

        ctx.save_for_backward(data, gate, indices_locations)
        data = data.cpu()
        gate = gate.cpu()
        indices_locations = indices_locations.cpu()
        # output: [S, M]
        rout = torch.zeros((dim_S, dim_M), dtype = data.dtype)
        indices, locations = torch.split(indices_locations, 1, dim=0)
        indices = torch.squeeze(indices)
        locations = torch.squeeze(locations)
        for i in range(dim_S):
            loc = locations[i].cpu().item()
            idx = indices[i].cpu().item()
            if idx < dim_E and loc < dim_C:
                rout[i] = data[idx, loc] * gate[i]
        # import os
        # import numpy as np
        # global counter
        # if not os.path.exists(f"./moe_decode_fw_inouts_{counter}.npz"):
        #     with open(f"./moe_decode_fw_inouts_{counter}.npz", "wb") as f:
        #         data_np = data.numpy()
        #         gate_np = gate.numpy()
        #         ind_locs = indices_locations.numpy()
        #         out_np = rout.numpy()
        #         np.savez(f, data=data_np, gate=gate_np, ind_locs=ind_locs, out=out_np)
        #         counter += 1
        return rout.to(ctx.device)

    @staticmethod
    def backward(ctx, grad_out):
        data, gate, indices_locations = ctx.saved_tensors

        grad_out = grad_out.cpu()
        data = data.cpu()
        gate = gate.cpu()
        indices_locations = indices_locations.cpu()
        # d_data, dgate
        d_data = torch.zeros(ctx.data_shape, dtype=ctx.dtype)
        dgate = torch.zeros(ctx.gate_shape, dtype=ctx.dtype)
        indices, locations = torch.split(indices_locations, 1, dim=0)
        indices = torch.squeeze(indices)
        locations = torch.squeeze(locations)
        for i in range(ctx.dim_S):
            idx = indices[i]
            loc = locations[i]
            if idx >= 0 and loc < ctx.dim_C:
                d_data[idx, loc, :] = grad_out[i] * gate[i]
                dgate[i] = torch.sum(grad_out[i] * data[idx, loc])
        # don't have gradient for d_indices_locations
        return (
            d_data.to(ctx.device),
            dgate.to(ctx.device),
            None,
        )


class FakeAllToAllOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.contiguous()

    @staticmethod
    def backward(ctx, *grad_output):
        return FakeAllToAllOp.apply(*grad_output)


class FakeAllToAllOpForCorrectness(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if torch._C._get_tracing_state():
            return input.contiguous()
        output = torch.zeros_like(input, device=input.device, dtype=input.dtype)
        torch.distributed.all_to_all_single(output, input)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return FakeAllToAllOpForCorrectness.apply(*grad_output)


# copied and modified from DeepSpeed
class Experts(torch.nn.Module):
    def __init__(self, expert, world_size, hidden_size, num_local_experts=1):
        super(Experts, self).__init__()

        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)]
        )
        self.num_local_experts = num_local_experts
        self.world_size = world_size
        self.hidden_size = hidden_size

        # TODO: revisit allreduce for moe.gate...
        for expert in self.experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            chunk = chunk.squeeze().transpose(1, 0).reshape(-1, self.hidden_size)
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            out = (
                out.reshape(-1, self.world_size, self.hidden_size)
                .transpose(1, 0)
                .unsqueeze(1)
            )
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output


class Gate(torch.nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, input):
        logits = self.wg(input)
        gates = F.softmax(logits, dim=1)
        return gates


class MOELayerImpl(torch.nn.Module):
    def __init__(self, gate, experts, world_size, num_local_experts, check_correctness=False):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.world_size = world_size
        self.num_local_experts = num_local_experts
        self.check_correctness = check_correctness

    def forward(self, *input, **kwargs):
        # pad sequence length to multiple of 4
        input_tensor = input[0]
        # Implement Algorithm 2 from GShard paper.
        d_model = input_tensor.shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input_tensor.reshape(-1, d_model)

        gate_weights = self.gate(reshaped_input)

        gates_s, indices_locations, used_capacity, n_elems_per_experts, dispatched_input = FakeMoeEncodeOp.apply(
            reshaped_input, gate_weights, torch.zeros((self.num_local_experts * self.world_size,)).int(), torch.ones(1),
        )

        if self.check_correctness:
            dispatched_input = FakeAllToAllOpForCorrectness.apply(dispatched_input)
        else:
            dispatched_input = FakeAllToAllOp.apply(dispatched_input)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(
            self.world_size, self.num_local_experts, -1, d_model
        )
        # DEBUG
        # dispatched_input = reshaped_input.reshape(
        #     self.world_size, self.num_local_experts, -1, d_model
        # )

        expert_output = self.experts(dispatched_input)

        if self.check_correctness:
            expert_output = FakeAllToAllOpForCorrectness.apply(expert_output)
        else:
            expert_output = FakeAllToAllOp.apply(expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.world_size * self.num_local_experts, -1, d_model
        )

        combined_output = FakeMoeDecodeOp.apply(
            expert_output, gates_s, indices_locations
        )
        combined_output = combined_output.reshape(input_tensor.shape)
        # DEBUG
        # combined_output = expert_output.reshape(input_tensor.shape)
        return combined_output


class RAFOutputOnlyMoE(torch.nn.Module):
    """
    Wrapper for deepspeed moe layer, return output only.
    """

    def __init__(
        self,
        model_hidden_size,
        expert_hidden_size,
        expert,
        num_experts,
        world_size,
        **kwargs
    ):
        super().__init__()
        num_local_experts = num_experts // world_size

        check_correctness = kwargs.get("check_correctness", False)
        experts = Experts(expert, world_size, model_hidden_size, num_local_experts)
        self.moe = MOELayerImpl(
            Gate(model_hidden_size, num_experts), experts, world_size, num_local_experts, check_correctness=check_correctness
        )

    def forward(self, x):
        output = self.moe(x)
        return output


class PTOutputOnlyMoE(torch.nn.Module):
    """
    Wrapper for deepspeed moe layer, return output only.
    """

    def __init__(
        self,
        model_hidden_size,
        expert_hidden_size,
        expert,
        num_experts,
        world_size,
        **kwargs
    ):
        super().__init__()
        import deepspeed

        self.moe = deepspeed.moe.layer.MoE(
            hidden_size=model_hidden_size,
            expert=expert,
            num_experts=num_experts,
            k=1,
            min_capacity=4,
            use_rts=False,
            use_tutel=False,
        )

    def forward(self, x):
        torch.cuda.nvtx.range_push("MoE Forward")
        output, _, _ = self.moe(x)
        torch.cuda.nvtx.range_pop()
        # register grad hook for marking MoE backward
        def output_hook(grad):
            torch.cuda.nvtx.range_push("MoE Backward")
            return grad
        
        def input_hook(grad):
            torch.cuda.nvtx.range_pop()
            return grad
        output.register_hook(output_hook)
        x.register_hook(input_hook)
        return output


class TutelOutputOnlyMoE(torch.nn.Module):
    """
    Wrapper for Tutel moe layer, return output only.
    """

    def __init__(
        self,
        model_hidden_size,
        expert_hidden_size,
        expert,
        num_experts,
        world_size,
        a2a_ffn_overlap_degree=1,
        **kwargs
    ):
        super().__init__()
        assert (
            num_experts % world_size == 0
        ), "num_experts must be divisible by world_size"
        from tutel import moe as tutel_moe

        # result function
        def del_aux_loss(output):
            del output.l_aux
            return output

        batch_prioritized_routing = kwargs.get("batch_prioritized_routing", False)

        self.moe = tutel_moe.moe_layer(
            gate_type={"type": "top", "k": 1, "capacity_factor": 1.0},
            model_dim=model_hidden_size,
            experts={
                "count_per_node": num_experts // world_size,
                "type": "ffn",
                "hidden_size_per_expert": expert_hidden_size,
                "activation_fn": lambda x: torch.nn.functional.relu(x),
            },
            a2a_ffn_overlap_degree=a2a_ffn_overlap_degree,
            scan_expert_func=lambda name, param: setattr(param, "allreduce", False),
            result_func=del_aux_loss,
            batch_prioritized_routing=batch_prioritized_routing
        )

    def forward(self, x):
        torch.cuda.nvtx.range_push("MoE Forward")
        output = self.moe(x)
        torch.cuda.nvtx.range_pop()
        # register grad hook for marking MoE backward
        def output_hook(grad):
            torch.cuda.nvtx.range_push("MoE Backward")
            return grad
        
        def input_hook(grad):
            torch.cuda.nvtx.range_pop()
            return grad
        output.register_hook(output_hook)
        x.register_hook(input_hook)
        return output

class FasterMoEOutputOnlyMoE(torch.nn.Module):
    """
    Wrapper for FasterMoE moe layer, return output only.
    """

    def __init__(self,
        model_hidden_size,
        expert_hidden_size,
        expert,
        num_experts,
        world_size,
        **kwargs):
        super().__init__()
        assert (
            num_experts % world_size == 0
        ), "num_experts must be divisible by world_size"
        # from fmoe.transformer import FMoETransformerMLP
        from fmoe import FMoE
        from fmoe.gates import SwitchGate

        class FasterMoEConfigurableSwitchGate(SwitchGate):
            def __init__(self, *args, **kwargs):
                kwargs["capacity"] = (1.0, 1.0)
                super().__init__(*args, **kwargs)

        class ExpertWrapper(torch.nn.Module):
            def __init__(self, d_model):
                super().__init__()
                # unused, since d_model is already specified in experts
                self.d_model = d_model
                self.expert = copy.deepcopy(expert)

            def forward(self, inp, fwd_expert_count):
                return self.expert(inp)

        # copied from fmoe.transformer
        class FMoETransformerMLP(FMoE):
            r"""
            A complete MoE MLP module in a Transformer block.
            * `activation` is the activation function to be used in MLP in each expert.
            * `d_hidden` is the dimension of the MLP layer.
            """

            def __init__(
                self,
                num_expert=32,
                d_model=1024,
                d_hidden=4096,
                activation=torch.nn.GELU(),
                expert_dp_comm="none",
                expert_rank=0,
                **kwargs
            ):
                super().__init__(num_expert=num_expert, d_model=d_model, expert=ExpertWrapper, **kwargs)
                self.mark_parallel_comm(expert_dp_comm)

            def forward(self, inp: torch.Tensor):
                r"""
                This module wraps up the FMoE module with reshape, residual and layer
                normalization.
                """
                original_shape = inp.shape
                inp = inp.reshape(-1, self.d_model)
                output = super().forward(inp)
                return output.reshape(original_shape)

        self.moe = FMoETransformerMLP(num_experts // world_size,
                                      model_hidden_size,
                                      expert_hidden_size,
                                      activation=torch.nn.ReLU(),
                                      world_size=world_size,
                                      mp_group=None,
                                      gate=FasterMoEConfigurableSwitchGate,
                                      top_k = 1)

    def forward(self, x):
        torch.cuda.nvtx.range_push("MoE Forward")
        output = self.moe(x)
        torch.cuda.nvtx.range_pop()
        # register grad hook for marking MoE backward
        def output_hook(grad):
            torch.cuda.nvtx.range_push("MoE Backward")
            return grad
        
        def input_hook(grad):
            torch.cuda.nvtx.range_pop()
            return grad
        output.register_hook(output_hook)
        x.register_hook(input_hook)
        return output