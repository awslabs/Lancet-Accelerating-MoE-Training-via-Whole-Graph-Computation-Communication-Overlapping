# usage:
# python3 benchmark_ds.py --model-name gpt2 --batch 64
# python3 benchmark_ds.py --model-name gpt2 --bucket_size 1e7 --zero --batch 64
import os
import argparse
import json
import subprocess

import benchmark
import raf
from raf.distributed import get_context
from benchmark.pytorch.torch_bencher import TorchBencher

N_GPUS_PER_NODE = 8

parser = argparse.ArgumentParser(description="Profile DeepSpeed models.")
parser.add_argument("--model-name", type=str, default="gpt2_moe", help="Model to profile")
parser.add_argument("--batch", type=int, default=64, help="Batch size")
parser.add_argument("--gate-type", type=str, default="switch", help="MoE gate type")
parser.add_argument("--fp16", action="store_true", help="Use FP16")
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

print(f"Running local rank: {local_rank}", flush=True)
print(f"Description: {descr}", flush=True)

# prepare DS config
ds_config = {
  "train_batch_size": args.batch * world_size,
  "steps_per_print": 2000,
  "optimizer": {
    "type": "SGD",
    "params": {
      "lr": 0.1,
      "momentum": 0.01
    }
  },
  "prescale_gradients": False,
  "wall_clock_breakdown": False,
  "zero_optimization": {
      "stage": 0,
  },
  "zero_allow_untested_optimizer": True
}

# dump config to file
ds_config_path = f"ds_config_{descr}.json"
if local_rank == 0:
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f)
dctx.barrier()

if "moe" in args.model_name:
    bencher = benchmark.get_model_bencher(
        "torch",
        args.model_name,
        batch_size=args.batch,
        device="cuda",
        moe_framework="deepspeed",
        num_experts=num_experts,
        shape=seq_len,
        num_layers=n_layers,
        d_model=d_model,
        n_head=n_heads,
        dtype = "float16" if args.fp16 else "float32",
        moe_gate_type=args.gate_type,
    )
else:
    bencher = benchmark.get_model_bencher(
        "torch", args.model_name, batch_size=args.batch
    )

bencher: TorchBencher

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
    deepspeed=True,
    moe_framework="deepspeed",
    ds_config_path=ds_config_path,
)
print(f"Per iteration time: {time}", flush=True)

# clean up
if local_rank == 0:
    os.remove(ds_config_path)
