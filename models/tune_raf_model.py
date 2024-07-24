# usage:
# python3 tune_raf_model.py --model gpt2_moe \
#   --dp_group_size 16
# output: schedule.json
import os
import argparse

import benchmark
import raf
from raf.distributed import get_context
from benchmark.raf.raf_bencher import RAFBencher

parser = argparse.ArgumentParser("Run scheduling.")
parser.add_argument("--model-name", type=str, required=True, help="Model to tune.")
parser.add_argument("--dp-group-size", type=int, required=True, help="Size of data parallel group")
parser.add_argument("--batch", type=int, required=True, help="Batch size.")
parser.add_argument("--output", type=str, help="Output path")
parser.add_argument("--device", type=int, default=0, help="Device to use.")
parser.add_argument("--fp16", action="store_true", help="Use FP16")
parser.add_argument("--tune-op-only", type=str, nargs='+', default=[], help="Tune only the given op.")

args = parser.parse_args()

if args.output is None:
    args.output = f"schedule_{args.model_name}_b{args.batch}_dp{args.dp_group_size}.json"

os.environ["RAF_LOCAL_RANK_OVERRIDE"] = str(args.device)
dctx = get_context()
dctx.size = args.dp_group_size
dctx.set_local_rank_for_tuning(args.device)
os.environ["WORLD_SIZE_OVERRIDE"] = str(args.dp_group_size)

# model name must specify ndim, layers and seqlen in such order
if "seqlen" in args.model_name:
    seq_len = int(args.model_name.split("_seqlen")[-1])
    args.model_name = args.model_name.rsplit("_", 1)[0]
else:
    seq_len = 512
if "nlayers" in args.model_name:
    n_layers = int(args.model_name.split("_nlayers")[-1])
    args.model_name = args.model_name.rsplit("_", 1)[0]
else:
    n_layers = 12
if "dmodel" in args.model_name:
    d_model = int(args.model_name.split("_dmodel")[-1])
    args.model_name = args.model_name.rsplit("_", 1)[0]
else:
    d_model = 768
if "nheads" in args.model_name:
    n_heads = int(args.model_name.split("_nheads")[-1])
    args.model_name = args.model_name.rsplit("_", 1)[0]
else:
    n_heads = 12
print("Using model ", args.model_name, " with ", n_layers, " layers, dmodel ", d_model, ", nheads ", n_heads, ", seqlen ", seq_len , ", batch size ", args.batch, sep="")

if "moe" in args.model_name:
    bencher = benchmark.get_model_bencher(
        "raf",
        args.model_name,
        batch_size=args.batch,
        device=f"cuda({args.device})",
        num_experts=2*args.dp_group_size,
        shape=seq_len,
        num_layers=n_layers,
        d_model=d_model,
        n_head=n_heads,
        dtype="float16" if args.fp16 else "float32",
    )
else:
    bencher = benchmark.get_model_bencher("raf", args.model_name, batch_size=args.batch_size)

bencher: RAFBencher

bencher.tune(
    args.output,
    device=f"cuda({args.device})",
    warmup=50,
    number=50,
    train=True,
    use_interpreter=False,
    fuse_level=1,
    data_parallel=True,
    overlap_comm_forward=False,
    enable_lancet=False,
    only_tune_tasks_with_name=args.tune_op_only,
)