# usage: 
# python3 run_lancet_optimization.py --load_profile /models/test.profile \
#   --dp_group_size 16
# output: timeline_*.json, optimized_*.mod
import os
from os.path import basename, splitext
import argparse
import glob
import time
import subprocess

import raf
from raf._core.device import Device
from raf.distributed import get_context


parser = argparse.ArgumentParser("Run scheduling.")
parser.add_argument("--load-profile", type=str, required=True, help="Profiled model to optimize")
parser.add_argument("--output-dir", type=str, default="/models", help="Output directory")
parser.add_argument("--max-partition", type=int, default=8, help="Max partition")
parser.add_argument("--disable-partition", action="store_true", help="Disable operator partition")
parser.add_argument("--disable-fusion", action="store_true", help="Disable tensor fusion")
parser.add_argument("--schedule-fifo", action="store_true", help="Use FIFO schedule")
parser.add_argument("--schedule-dw", action="store_true", help="Use dW schedule")
parser.add_argument("--timeline-opt-algo", default="heuristic", type=str, choices=["heuristic", "dp", "range"], help="Optimization algorithm for timeline")
parser.add_argument("--range-ngroups", default=1, type=int, help="Range (ngroups) for range based partition")
parser.add_argument("--world-size", type=int, required=True, help="World size (number of nodes)")
parser.add_argument("--dp-group-size", type=int, required=True, help="Size of data parallel group")
parser.add_argument("--alltoall-profile-path", type=str, help="Path to alltoall profile")
parser.add_argument("--device", type=int, default=0, help="Device to benchmark computation on.")

args = parser.parse_args()

if not args.schedule_fifo and not args.schedule_dw:
    raise ValueError("Must specify a schedule to use.")

descr_strs = splitext(basename(args.load_profile))[0].split("_")
time_str = "_".join(descr_strs[-5:])
prefix = "_".join(descr_strs[:-5])

dp_str = "_dp" if args.disable_partition else ""
df_str = "_df" if args.disable_fusion else ""
schedule_str = "fifo" if args.schedule_fifo else "dw"
rg_str = f"_range{args.range_ngroups}" if args.timeline_opt_algo == "range" else ""
# descr = f"{prefix}_{schedule_str}{dp_str}{df_str}_{time_str}"
descr = f"{prefix}_{schedule_str}{dp_str}{df_str}{rg_str}"

if args.disable_partition:
    os.environ["DISABLE_PARTITION"] = "1"
if args.disable_fusion:
    os.environ["DISABLE_FUSION"] = "1"

os.environ["SIMULATION_DEBUG_PREFIX"] = f"{args.output_dir}/opt_timeline_{descr}" 
os.environ["DUMP_OPTIMIZED_EXPR_PREFIX"] = f"{args.output_dir}/optimized_{descr}"
os.environ["LANCET_PARTITION_RANGE_NGROUPS"] = str(args.range_ngroups)

os.environ["MAX_PARTITION"] = str(args.max_partition)

# get default supplementary profile paths
def get_instance_type():
    try:
        token = subprocess.check_output('curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"', shell=True)
        token = token.decode("utf-8")
        instance_type = subprocess.check_output(f'curl -s -H "X-aws-ec2-metadata-token: {token}" http://169.254.169.254/latest/meta-data/instance-type', shell=True)
        instance_type = instance_type.decode("utf-8").split(".")[0]
    except:
        instance_type = "unknown"
    return instance_type

instance_type = get_instance_type()

profile_paths = {
    "alltoall_perf": args.alltoall_profile_path,
}
profile_env_prefix = {
    "alltoall_perf": "ALL2ALL",
}
for program_name, profile_path in profile_paths.items():
    if profile_path is None:
        default_path = f"/models/{program_name}_{args.world_size}x{instance_type}.csv"
        if os.path.isfile(default_path):
            profile_paths[program_name] = default_path
    if profile_paths[program_name] is not None:
        assert os.path.isfile(profile_paths[program_name])
        os.environ[f"{profile_env_prefix[program_name]}_SUPPLEMENT_PROFILE"] = profile_paths[program_name]

dctx = get_context()
dctx.size = args.dp_group_size
dctx.set_local_rank_for_tuning(args.device)

profile_prefix = splitext(args.load_profile)[0]
if args.schedule_fifo:
    heuristic = "FIFO"
    print("Using FIFO schedule for optimization.")
elif args.schedule_dw:
    heuristic = "dW"
    print("Using dW schedule for optimization.")
else:
    raise ValueError("Must specify a schedule to use.")

# record the time needed for optimization
start_time = time.time()
with Device(f"cuda({args.device})"):
    sim = raf.distributed.LancetScheduleSimulator()
    expr = sim.load_profile(profile_prefix)
    scheduled_expr = sim.run(expr, heuristic, args.timeline_opt_algo, args.dp_group_size)
end_time = time.time()
print("Optimization time: ", end_time - start_time)