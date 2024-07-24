# usage:
# python3 benchmark_pt.py --model-name gpt2 --ip 0.0.0.0 --batch 64
import os
import argparse
import subprocess

import benchmark
import raf
from raf.distributed import get_context
from benchmark.pytorch.torch_bencher import TorchBencher

N_GPUS_PER_NODE = 8

parser = argparse.ArgumentParser(description="Profile PyTorch models.")
parser.add_argument("--model-name", type=str, default="gpt2_moe", help="Model to profile")
parser.add_argument("--ip", type=str, required=True, help="Master IP.")
parser.add_argument("--moe-framework", type=str, required=True, choices=["tutel", "fastermoe"], help="MOE framework")
parser.add_argument("--overlap-degree", type=int, default=1, help="overlap degree")
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

os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["MASTER_ADDR"] = args.ip
os.environ["MASTER_PORT"] = "11451"

if args.moe_framework == "fastermoe":
    os.environ["FMOE_FASTER_SCHEDULE_ENABLE"] = "1"
    os.environ["FMOE_FASTER_GROUP_SIZE"] = str(args.overlap_degree)

print(f"Running local rank: {local_rank}", flush=True)
print(f"Description: {descr}", flush=True)

if "moe" in args.model_name:
    bencher = benchmark.get_model_bencher(
        "torch",
        args.model_name,
        batch_size=args.batch,
        device="cuda",
        moe_framework=args.moe_framework,
        a2a_ffn_overlap_degree=args.overlap_degree,
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

warmup = 100
number = 100
if args.nsys_profile:
    warmup = 20
    number = 20

bencher: TorchBencher
time = bencher.bench(
    device="cuda",
    warmup=warmup,
    number=number,
    train=True,
    data_parallel=True,
    use_interpreter=False,
    fuse_level=1,
    deepspeed=False,
    moe_framework=args.moe_framework,
    horovod=False,
)

print(f"Per iteration time: {time}", flush=True)
