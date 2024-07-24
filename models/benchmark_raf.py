# usage:
# python3 benchmark_raf.py --model_name gpt2 --zero --batch 64
# output: timeline_*.json, debug_*.mod, debug_.profile
import os
import argparse
import subprocess

import benchmark
import raf
from raf.distributed import get_context
from benchmark.raf.raf_bencher import RAFBencher

N_GPUS_PER_NODE = 8

parser = argparse.ArgumentParser(description="Profile RAF models.")
parser.add_argument("--model-name", type=str, default="gpt2_moe", help="Model to profile")
parser.add_argument("--lancet-optimized-module", type=str, help="Load lancet optimized module from")
parser.add_argument("--batch", type=int, default=64, help="Batch size")
parser.add_argument("--gate-type", type=str, default="switch", help="MoE gate type")
parser.add_argument("--fp16", action="store_true", help="Use FP16")
parser.add_argument("--lancet-profile", action="store_true", help="Run lancet benchmark.")
parser.add_argument("--nsys-profile", action="store_true", help="Use with Nsight Systems profiler")

args = parser.parse_args()

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
print("Using model ", args.model_name, " with ", n_layers, " layers, dmodel ", d_model, ", nheads ", n_heads, ", seqlen ", seq_len , ", batch size ", args.batch, ", gate type ", args.gate_type, sep="", flush=True)


dctx = get_context()
local_rank = dctx.local_rank
world_size = dctx.size
num_experts = 2 * world_size

def get_instance_type():
    try:
        token = subprocess.check_output('curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"', shell=True)
        token = token.decode("utf-8")
        instance_type = subprocess.check_output(f'curl -s -H "X-aws-ec2-metadata-token: {token}" http://169.254.169.254/latest/meta-data/instance-type', shell=True)
        instance_type = instance_type.decode("utf-8").split(".")[0]
    except:
        instance_type = "unknown"
    return instance_type

descr = f"{args.model_name}_nl{n_layers}_m{d_model}_nh{n_heads}_sl{seq_len}_b{args.batch}{'_fp16' if args.fp16 else ''}_{int(world_size // N_GPUS_PER_NODE)}x{get_instance_type()}"

os.environ["SIMULATION_DEBUG_PREFIX"] = f"/models/timeline_{descr}"
os.environ["DEBUG_DUMP_PROFILE_PREFIX"] = f"/models/profile_{descr}"
if args.lancet_optimized_module:
    os.environ["LOAD_OPTIMIZED_MODULE_FROM"] = args.lancet_optimized_module

print(f"Running local rank: {local_rank}", flush=True)
print(f"Description: {descr}", flush=True)

assert "moe" in args.model_name, "Only MOE models are supported for this benchmark."

bencher = benchmark.get_model_bencher(
    "raf",
    args.model_name,
    batch_size=args.batch,
    device=f"cuda({local_rank})",
    num_experts=num_experts,
    shape=seq_len,
    num_layers=n_layers,
    d_model=d_model,
    n_head=n_heads,
    dtype = "float16" if args.fp16 else "float32",
    moe_gate_type=args.gate_type,
)

bencher: RAFBencher

warmup = 100
number = 100
if args.nsys_profile:
    warmup = 20
    number = 20

time = bencher.bench(
    device="cuda",
    warmup=warmup,
    number=number,
    train=True,
    use_interpreter=False,
    fuse_level=1,
    data_parallel=True,
    overlap_comm_forward=False,
    profile=50 if (args.lancet_profile or args.lancet_optimized_module) else 0,
    enable_lancet=True if args.lancet_optimized_module else False,
    partition_comm = False,
)
print(f"Per iteration time: {time}", flush=True)
