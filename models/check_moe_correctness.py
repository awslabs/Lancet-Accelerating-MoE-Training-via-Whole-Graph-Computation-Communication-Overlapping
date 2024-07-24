import os
import argparse

parser = argparse.ArgumentParser(description='Test RAF MoE model correctness.')
parser.add_argument('--model-name', type=str, help='The MoE model to check.')
parser.add_argument('--batch', type=int, default=64, help='Model batch size.')
parser.add_argument('--ip', type=str, required=True, help="Master IP.")
parser.add_argument('--fp16', action='store_true', help="Use FP16.")
parser.add_argument('--lancet-optimized-module', type=str, help="Load lancet optimized module from")

args = parser.parse_args()

if args.lancet_optimized_module:
    os.environ["LOAD_OPTIMIZED_MODULE_FROM"] = args.lancet_optimized_module

import benchmark
import raf
from raf._core.device import Device
from raf.distributed import get_context, RemoveCommunicator

dctx = get_context()

os.environ["LOCAL_RANK"] = str(dctx.local_rank)
os.environ["MASTER_ADDR"] = args.ip
os.environ["MASTER_PORT"] = "11451"

profile=50 if args.lancet_optimized_module else 0

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
print("Using model ", args.model_name, " with ", n_layers, " layers, dmodel ", d_model, ", nheads ", n_heads, ", seqlen ", seq_len , ", batch size ", args.batch, sep="", flush=True)

bench = benchmark.get_model_bencher("raf", args.model_name, batch_size=args.batch, include_orig_model=True,
                                    num_experts=max(2, 2 * dctx.size), device=f"cuda({dctx.local_rank})",
                                    num_layers=n_layers,
                                    shape=seq_len,
                                    d_model=d_model,
                                    n_head=n_heads,
                                    dtype = "float16" if args.fp16 else "float32",
                                    check_correctness=True,)
bench.check_correctness(device="cuda", check_gradient=True, data_parallel=True, pt_is_moe=True, num_train_iter=1, enable_lancet=True if args.lancet_optimized_module else False, profile=profile)
RemoveCommunicator()
