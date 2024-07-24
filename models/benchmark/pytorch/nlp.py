"""Register NLP models from PyTorch."""
# pylint: disable=not-callable, missing-function-docstring, protected-access, unused-argument
import os

import numpy as np
import torch
import transformers

from ..logger import get_logger
from ..registry import reg_model
from .torch_bencher import TorchBencher
from .utils import randn_torch, one_hot_torch, to_torch_dev, init_deepspeed_moe_pt, init_tutel_moe, \
                   init_deepspeed_moe_raf, init_fastermoe_moe, \
                   PTOutputOnlyMoE, RAFOutputOnlyMoE, TutelOutputOnlyMoE, \
                   FasterMoEOutputOnlyMoE, ParallelModelLoader
from ..utils import fix_seed
from ..dataset import load_wikitext2

logger = get_logger("PyTorch-NLP")  # pylint: disable=invalid-name


class ConvertNLPContext:
    """The context to deal with TOKENIZERS_PARALLELISM."""

    def __init__(self):
        self.tokenizers_parallelism = None

    def __enter__(self):
        if "TOKENIZERS_PARALLELISM" in os.environ:
            self.tokenizers_parallelism = os.environ["TOKENIZERS_PARALLELISM"]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def __exit__(self, ptype, value, trace):
        if self.tokenizers_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = self.tokenizers_parallelism
        else:
            del os.environ["TOKENIZERS_PARALLELISM"]


def transformer_common(model_config, batch_size, seq_length_or_cwh, dtype, include_orig_model,
                       moe_framework = 'raf', moe_gate_type = "switch", 
                       device="cpu", use_moe=False, num_experts=None, 
                       moe_expert_interval=2, a2a_ffn_overlap_degree=1, check_correctness=False):
    """The utility of processing transformer models.

    Parameters
    ----------
    model_config: Dict[str, Any]
        The Huggingface PyTorch model configuration.

    batch_size: int
        Batch size.

    seq_length: Optional[int]
        The sequence length. If None, 128 will be used.

    dtype: str
        The data type. Default is float32.

    include_orig_model: bool
        Whether to include the original model as the reference.

    use_moe: bool
        Whether to replace the mlp in transformer blocks with MoE layer.

    moe_expert_interval: int
        Interval between two transformer layers whose mlp is replaced by a MoE layer.
        i.e. blocks with layer_number % moe_expert_interval == 0 will be replaced.

    Returns
    -------
    mod_n_shape: Tuple[raf.Model, Tuple[int, int]]
        The converted model and input shape.
    """
    fix_seed(42)
    assert hasattr(model_config, "architectures"), '"architectures" is missing in the config'
    model_cls = model_config.architectures[0]
    model_config.use_cache = False  # Disable model cache to avoid unnecessary model outputs.
    assert hasattr(transformers, model_cls), "%s is not supported in transformers" % model_cls
    with ParallelModelLoader():
        t_model = getattr(transformers, model_cls)(model_config)

    # replace mlp with MoE layer
    if use_moe:
        assert "gpt2" in model_cls.lower() or "vit" in model_cls.lower() or "bert" in model_cls.lower(), "Only support GPT2, ViT and BERT in transformers for now."
        if moe_framework == 'raf':
            init_moe_envs = init_deepspeed_moe_raf
            OutputOnlyMoE = RAFOutputOnlyMoE
        elif moe_framework == 'deepspeed':
            init_moe_envs = init_deepspeed_moe_pt
            OutputOnlyMoE = PTOutputOnlyMoE
        elif moe_framework == 'tutel':
            init_moe_envs = init_tutel_moe
            OutputOnlyMoE = TutelOutputOnlyMoE
        elif moe_framework == "fastermoe":
            init_moe_envs = init_fastermoe_moe
            OutputOnlyMoE = FasterMoEOutputOnlyMoE
        # init deepspeed moe environment
        num_experts, world_size, local_rank = init_moe_envs(num_experts=num_experts, check_correctness=check_correctness)
        if device == "cuda":
            device = "cuda(%d)" % local_rank
        hidden_size = model_config.hidden_size
        moe_kwargs = {} if moe_framework != 'tutel' else {'a2a_ffn_overlap_degree': a2a_ffn_overlap_degree}
        if check_correctness:
            moe_kwargs['check_correctness'] = True
        if moe_gate_type == "batch_prioritized":
            moe_kwargs['batch_prioritized_routing'] = True
        if "vit" in model_cls.lower():
            from operator import attrgetter
            for attr in ["vit.encoder"]:
                blocks = attrgetter(attr)(t_model).layer
                for layer_idx, block in enumerate(blocks):
                    if layer_idx % moe_expert_interval == 0:
                        intermediate_size = attrgetter(attr)(t_model).layer[layer_idx].intermediate.dense.out_features
                        expert = torch.nn.Sequential(
                                        torch.nn.Linear(hidden_size, intermediate_size),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(intermediate_size, hidden_size)
                                        )
                        # TODO: somehow using MLP in MoE as experts results in NaN in gradients. fix later
                        # from transformers.models.gpt2.modeling_gpt2 import MLP
                        # expert = MLP(intermediate_size, model_config)
                        class MoEViTOutput(torch.nn.Module):
                            def __init__(self) -> None:
                                super().__init__()

                            def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
                                hidden_states = hidden_states + input_tensor
                                return hidden_states
                        attrgetter(attr)(t_model).layer[layer_idx].intermediate = OutputOnlyMoE(hidden_size, intermediate_size, expert, num_experts, world_size, **moe_kwargs)
                        attrgetter(attr)(t_model).layer[layer_idx].output = MoEViTOutput()
        elif "bert" in model_cls.lower():
            blocks = t_model.bert.encoder.layer
            for layer_idx, block in enumerate(blocks):
                if layer_idx % moe_expert_interval == 0:
                    intermediate_size = t_model.bert.encoder.layer[layer_idx].intermediate.dense.out_features
                    expert = torch.nn.Sequential(
                                    torch.nn.Linear(hidden_size, intermediate_size),
                                    torch.nn.GELU(),
                                    torch.nn.Linear(intermediate_size, hidden_size)
                                    )
                    # TODO: somehow using MLP in MoE as experts results in NaN in gradients. fix later
                    # from transformers.models.gpt2.modeling_gpt2 import MLP
                    # expert = MLP(intermediate_size, model_config)
                    class MoEBertOutput(torch.nn.Module):
                        def __init__(self) -> None:
                            super().__init__()
                            self.LayerNorm = torch.nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)

                        def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
                            hidden_states = self.LayerNorm(hidden_states + input_tensor)
                            return hidden_states

                    t_model.bert.encoder.layer[layer_idx].intermediate = OutputOnlyMoE(hidden_size, intermediate_size, expert, num_experts, world_size, **moe_kwargs)
                    t_model.bert.encoder.layer[layer_idx].output = MoEBertOutput()
        else:
            blocks = t_model.transformer.h
            for layer_idx, block in enumerate(blocks):
                if layer_idx % moe_expert_interval == 0:
                    intermediate_size = t_model.transformer.h[layer_idx].mlp.c_fc.nf
                    expert = torch.nn.Sequential(
                                    torch.nn.Linear(hidden_size, intermediate_size),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(intermediate_size, hidden_size)
                                    )
                    # TODO: somehow using MLP in MoE as experts results in NaN in gradients. fix later
                    # from transformers.models.gpt2.modeling_gpt2 import MLP
                    # expert = MLP(intermediate_size, model_config)
                    t_model.transformer.h[layer_idx].mlp = OutputOnlyMoE(hidden_size, intermediate_size, expert, num_experts, world_size, **moe_kwargs)

    t_model.to(to_torch_dev(device))

    if "vit" in model_cls.lower():
        chw = seq_length_or_cwh if seq_length_or_cwh is not None else (3, 224, 224)
        input_shape = [batch_size, *chw]
        np_x = np.random.randn(*input_shape).astype(dtype)
        t_x = torch.tensor(np_x).to(to_torch_dev(device))
    else:
        seq_length = seq_length_or_cwh if seq_length_or_cwh is not None else 128
        input_shape = [batch_size, seq_length]
        np_x = np.random.randint(0, 10000, input_shape)
        t_x = torch.tensor(np_x).to(to_torch_dev(device))
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_dy = randn_torch((), std=0.0, mean=1.0, requires_grad=False, dtype=dtype, device=to_torch_dev(device))

    if model_cls.find("ForSequenceClassification") != -1:
        t_y = t_model(t_x)
        t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=t_y[0].shape[1], device=to_torch_dev(device))
        output_shape = None
    elif model_cls.find("LM") != -1 or model_cls.find("Bart") != -1:
        # Language model (e.g., BertForMaskedLM, GPT2LMHeadModel, BertForPreTrainingPreLN)
        vocab_size = model_config.vocab_size
        t_ytrue = one_hot_torch(batch_size=batch_size * seq_length, num_classes=vocab_size, device=to_torch_dev(device))
        output_shape = (batch_size * seq_length, vocab_size)
    elif model_cls.find("ViTForImageClassification") != -1:
        t_ytrue = one_hot_torch(batch_size=batch_size, num_classes=1000, device=to_torch_dev(device))
        output_shape = None
    else:
        raise ValueError("Unsupported model type: %s" % model_cls)

    seq_length = seq_length_or_cwh if seq_length_or_cwh is not None else 128
    bencher = TorchBencher(t_model, input_shape, load_wikitext2(batch_size, seq_length), t_dy, reshape_output=output_shape)
    torch.cuda.empty_cache()
    return bencher


@reg_model("torch")
def bert_base_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        config.architectures = ["BertForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def bert_base_classify_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, moe_expert_interval=2, 
                           a2a_ffn_overlap_degree=1, moe_framework='raf'):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        config.architectures = ["BertForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model, moe_framework=moe_framework,
                                  device=device, use_moe=True, num_experts=num_experts, 
                                  moe_expert_interval=moe_expert_interval, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree)

@reg_model("torch")
def bert_test_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 1
        config.architectures = ["BertForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_large_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-large-uncased")
        config.architectures = ["BertForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def bert_large_test_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-large-uncased")
        config.num_hidden_layers = 1
        config.architectures = ["BertForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def gpt2_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.GPT2Config()
        config.pad_token_id = -1
        config.architectures = ["GPT2ForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def gpt2_test_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.GPT2Config()
        config.pad_token_id = -1
        config.n_layer = 1
        config.architectures = ["GPT2ForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_base_mlm(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_test_mlm(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 1
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_large_mlm(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-large-uncased")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def bert_large_test_mlm(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-large-uncased")
        config.num_hidden_layers = 1
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def gpt2(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("gpt2")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def gpt2_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", 
            num_experts=None, moe_expert_interval=2, a2a_ffn_overlap_degree=1,
            moe_framework='raf', moe_gate_type="switch",
            num_layers=12, d_model=768, n_head=12, check_correctness=False):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("gpt2")
        config.n_layer = num_layers
        config.n_embd = d_model
        config.n_head = n_head
        config.activation_function = "gelu" # the default new_gelu causes NaN in gradients
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model, 
                                  moe_framework=moe_framework, moe_gate_type=moe_gate_type,
                                  device=device, use_moe=True, num_experts=num_experts, 
                                  moe_expert_interval=moe_expert_interval, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, check_correctness=check_correctness)

@reg_model("torch")
def vit(batch_size, seq_length, dtype, include_orig_model, device="cuda"):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("google/vit-base-patch16-224")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def vit_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, moe_expert_interval=2, a2a_ffn_overlap_degree=1, moe_framework='raf'):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("google/vit-base-patch16-224")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model, moe_framework=moe_framework,
                                  device=device, use_moe=True, num_experts=num_experts, 
                                  moe_expert_interval=moe_expert_interval, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree)

@reg_model("torch")
def vit_test_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, moe_expert_interval=2, a2a_ffn_overlap_degree=1, moe_framework='raf'):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("google/vit-base-patch16-224")
        config.num_hidden_layers = 1
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model, moe_framework=moe_framework,
                                  device=device, use_moe=True, num_experts=num_experts,
                                  moe_expert_interval=moe_expert_interval, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree)

@reg_model("torch")
def gpt2_test(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("gpt2")
        config.n_layer = 1
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def gpt2_test_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, moe_expert_interval=1, a2a_ffn_overlap_degree=1, moe_framework='raf', check_correctness=False, **kwargs):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model, moe_framework=moe_framework,
                                  device=device, use_moe=True, num_experts=num_experts, 
                                  moe_expert_interval=moe_expert_interval, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, check_correctness=check_correctness)

@reg_model("torch")
def partitioned_moe(batch_size, seq_length, dtype, include_orig_model, device="cuda", num_experts=None, partition_number=2, moe_framework='raf'):
    fix_seed(42) # fix the random seed
    # init deepspeed moe environment
    if moe_framework == 'raf':
        init_moe_envs = init_deepspeed_moe_raf
        OutputOnlyMoE = RAFOutputOnlyMoE
    elif moe_framework == 'deepspeed':
        init_moe_envs = init_deepspeed_moe_pt
        OutputOnlyMoE = PTOutputOnlyMoE
    else:
        raise ValueError(f"moe_framework {moe_framework} not supported for partitioned moe.")
    num_experts, world_size = init_moe_envs(num_experts=num_experts)
    hidden_size = 768
    intermediate_size = 3072

    import torch.nn as nn
    class PartitionedMoE(nn.Module):
        def __init__(self):
            super(PartitionedMoE, self).__init__()
            expert = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, intermediate_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(intermediate_size, hidden_size))
            self.moe_layers = nn.ModuleList([OutputOnlyMoE(hidden_size, intermediate_size, expert, num_experts, world_size) for i in range(partition_number)])
            self.moe_layers_1 = nn.ModuleList([OutputOnlyMoE(hidden_size, intermediate_size, expert, num_experts, world_size) for i in range(partition_number)])

        def forward(self, x):
            if partition_number > 1:
                assert x.shape[0] % partition_number == 0, f"Input must be partitionable at dimension 0, but got: {x.shape[0]}"
            chunk_size = int(x.shape[0] / partition_number)
            splitted_inputs = torch.split(x, chunk_size, dim=0)
            splitted_outputs = []
            for part_idx, splitted_input in enumerate(splitted_inputs):
                out = self.moe_layers[part_idx](splitted_input)
                out = self.moe_layers_1[part_idx](out)
                splitted_outputs.append(out)
            concatted_outs = torch.cat(splitted_outputs, dim=0)
            return concatted_outs

    t_model = PartitionedMoE()
    t_model.to(to_torch_dev(device))

    seq_length = seq_length if seq_length is not None else 128
    input_shape = [batch_size * seq_length, hidden_size]

    t_x = torch.randn(input_shape, dtype=torch.float32, device=to_torch_dev(device))
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_dy = randn_torch((), std=0.0, mean=1.0, requires_grad=False, dtype=dtype, device=to_torch_dev(device))

    # Language model (e.g., BertForMaskedLM, GPT2LMHeadModel, BertForPreTrainingPreLN)
    t_ytrue = one_hot_torch(batch_size=batch_size * seq_length, num_classes=hidden_size, device=to_torch_dev(device))
    output_shape = (batch_size * seq_length, hidden_size)

    bencher = TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue, reshape_output=output_shape)
    torch.cuda.empty_cache()
    return bencher

@reg_model("torch")
def bart_base(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("facebook/bart-base")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def roberta_base(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("roberta-base")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)
