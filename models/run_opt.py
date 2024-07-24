import argparse
import glob
import os
import subprocess

# args:
# 0. opt_params (e.g. fixed range)
# 1. load profile path
# 2. output dir
# 3. schedule algo
# 4. timeline opt algo
# 5. (optional) disable partition
# 6. world size
# 7. dp group size
# 8. max partition
# 9. nnodes (same as world size)
# 10. instance type
# 11. device to run profiling on
# 12. spec name (for logging)
CMD_TEMPLATE = "mpirun --allow-run-as-root -np 1 {} python3 run_lancet_optimization.py --load-profile {} --output-dir {} {} --timeline-opt-algo {} {} --world-size {} --dp-group-size {} --max-partition {} --alltoall-profile-path ./alltoall_perf_{}x{}.csv --device {} 2>&1 | tee {}.log"
FIX_RANGE_ENV = "-x LANCET_PARTITION_RANGE_FIXED=1"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Tuning")
    parser.add_argument('--dir', type=str, required=True, help='Dir containing the unoptimized mod')
    parser.add_argument('--no-dw-schedule', action='store_true', help='Do not use dw schedule')
    parser.add_argument('--no-partition', action='store_true', help='Do not use partition')
    parser.add_argument('--baseline', action='store_true', help='Only perform fusion')
    parser.add_argument('--fixed-range', action='store_true', help='Use fixed range')
    parser.add_argument('--max-partition', type=int, default=8, help='Max number of partitions')
    parser.add_argument('--device', type=int, default=0, help='Device to run profiling on')

    args = parser.parse_args()
    return args

def get_unopt_mod_path(args):
    fn = glob.glob(f"{args.dir}/*.mod")[0]
    return fn

def parse_specs(dir_name):
    # example: gpt2_moe_nl12_m768_nh16_sl512_b64_4xp4de
    model_name = dir_name.split('_nl')[0]
    get_str_after = lambda s: dir_name.split(s)[-1].split('_')[0]
    nl = int(get_str_after('nl'))
    m = int(get_str_after('m'))
    nh = int(get_str_after('nh'))
    sl = int(get_str_after('sl'))
    b = int(get_str_after('b'))
    nnodes = int(dir_name.split('_')[-1].split('x')[0])
    instance_type = dir_name.split('_')[-1].split('x')[1]
    return model_name, nl, m, nh, sl, b, nnodes, instance_type

def get_instance_type():
    try:
        token = subprocess.check_output('curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"', shell=True)
        token = token.decode("utf-8")
        instance_type = subprocess.check_output(f'curl -s -H "X-aws-ec2-metadata-token: {token}" http://169.254.169.254/latest/meta-data/instance-type', shell=True)
        instance_type = instance_type.decode("utf-8").split(".")[0]
    except:
        instance_type = "unknown"
    return instance_type

def gen_cmd(args):
    # hardcoded some values for now
    if args.fixed_range:
        fixed_range_env = FIX_RANGE_ENV
    else:
        fixed_range_env = ""
    load_profile_path = get_unopt_mod_path(args)
    if args.fixed_range:
        output_dir = os.path.join(args.dir, "optimized_part_fixed_range")
    elif args.baseline:
        output_dir = os.path.join(args.dir, "optimized_baseline")
    elif args.no_partition:
        output_dir = os.path.join(args.dir, "optimized_part_none")
    elif args.no_dw_schedule:
        output_dir = os.path.join(args.dir, "optimized_no_dw_schedule")
    else:
        output_dir = os.path.join(args.dir, "optimized")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.baseline or args.no_dw_schedule:
        schedule_algo = "--schedule-fifo"
    else:
        schedule_algo = "--schedule-dw"
    timeline_opt_algo = "dp" if not args.fixed_range else "range"
    if args.baseline or args.no_partition:
        disable_partition = "--disable-partition"
    else:
        disable_partition = ""
    world_size = args.world_size
    dp_group_size = args.dp_group_size
    spec_name = "opt_" + os.path.basename(args.dir)
    if args.no_dw_schedule:
        spec_name += "_no_dw_schedule"
    if args.baseline:
        spec_name += "_baseline"
    if args.no_partition:
        spec_name += "_no_partition"
    if args.fixed_range:
        spec_name += "_fixed_range"
    spec_path = os.path.join(output_dir, spec_name)
    return CMD_TEMPLATE.format(fixed_range_env, load_profile_path, output_dir, schedule_algo, timeline_opt_algo, disable_partition, world_size, dp_group_size, args.max_partition, world_size, get_instance_type(), args.device, spec_path)

if __name__ == "__main__":
    args = parse_args()
    _,_,_,_,_,_, nnodes, exp_instance_type = parse_specs(os.path.basename(args.dir))
    assert exp_instance_type in ["p4de", "p3dn", "p4d"], f"Unknown instance type {exp_instance_type}"
    args.world_size = nnodes
    args.dp_group_size = 8 * nnodes
    cmd = gen_cmd(args)
    subprocess.run(cmd, shell=True)