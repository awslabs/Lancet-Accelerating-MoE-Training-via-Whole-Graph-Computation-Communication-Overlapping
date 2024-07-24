# benchmark and parse nccl-test log of all2all.
# usage: python3 create_nccl_profiles.py --world-size <world size>
import os
import subprocess
import requests
import argparse

DATA_PARAMS = {
    ("alltoall_perf", "alltoall_perf_small") : "-b 1024 -e 67108864 -i 1000000",
    ("alltoall_perf", "alltoall_perf_mid") : "-b 64M -e 128M -i 2000000",
    ("alltoall_perf", "alltoall_perf_large") : "-b 128M -e 2G -f 2"
}

def parse_args():
    parser = argparse.ArgumentParser(
            description="Create supplementary nccl profiles."
    )
    parser.add_argument("--world-size", type=int, required=True, help="World size")
    parser.add_argument("--n-gpus-per-node", type=int, default=8, help="N GPUs per node.")
    parser.add_argument("--from-file", type=str, default=None, help="Read profile from file (one type of communication at a time).")

    args = parser.parse_args()
    return args

def get_instance_type():
    try:
        token = subprocess.check_output('curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"', shell=True)
        token = token.decode("utf-8")
        instance_type = subprocess.check_output(f'curl -s -H "X-aws-ec2-metadata-token: {token}" http://169.254.169.254/latest/meta-data/instance-type', shell=True)
        instance_type = instance_type.decode("utf-8").split(".")[0]
    except:
        instance_type = "unknown"
    return instance_type

def run_profile():
    for (program_name, output_name), params in DATA_PARAMS.items():
        subprocess.run(f"./run_benchmark_nccl_perf.sh {program_name} {output_name} {params}", shell=True)

def main(args):
    instance_type = get_instance_type()
    n_machine = args.world_size
    prog_names = []
    prof_files = []
    out_files = []
    if args.from_file is not None:
        prof_file = args.from_file
        program_name = prof_file.split("/")[-1].split(".")[0]
        prog_names.append(program_name)
        prof_files.append(prof_file)
    else:
        run_profile()
        for program_name, output_name in DATA_PARAMS.keys():
            prof_file = f"/models/nccl_perf/{output_name}.log"
            prog_names.append(program_name)
            prof_files.append(prof_file)
    for prog_name in prog_names:
        out = f"/models/{prog_name}_{n_machine}x{instance_type}.csv"
        out_files.append(out)
    profiled_prog_names = set()
    for prog_name, prof_file, out in zip(prog_names, prof_files, out_files):
        prof_lines = open(prof_file, "r").readlines()
        beg_idx, end_idx = 0, -1
        for idx, line in enumerate(prof_lines):
            if "(GB/s)" in line:
                beg_idx = idx + 1
            if "Out of bounds values" in line:
                end_idx = idx

        assert beg_idx != 0 and end_idx != -1

        out_str = ""
        for idx in range(beg_idx, end_idx):
            line_splits = prof_lines[idx].split()
            if len(line_splits) != 13:
                continue
            # lancet profiler uses output sizes to calculate throughput
            # do conversion here. size should be in number of elements
            size = float(line_splits[0])
            if program_name == "reduce_scatter_perf":
                # nccl-test use max(send size, recv size)
                # needs to scale down by dp group size
                size = size / args.world_size / args.n_gpus_per_node
            size = size * 8 / 32
            time = line_splits[-4]
            out_str += f"{size},{time}\n"

        open_method = "w" if prog_name not in profiled_prog_names else "a"
        with open(out, open_method) as writer:
            writer.write(out_str)
        profiled_prog_names.add(prog_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)