import os
import subprocess
import psutil
import time
import glob
import argparse
from dataclasses import dataclass
from typing import List
import json
from tqdm import tqdm


EXP_DIR_PREFIX = "/models/experiments"
FP16=False
BASELINE_CANDIDATE_PARTITONS = [1, 2, 4, 8]

@dataclass
class ExpConfig:
    exp_name: str
    framework: str
    moe_gate_type: str
    max_partition: int
    no_dw_scheduling: bool
    partition_type: str

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAF optimized benchmarks.")
    parser.add_argument("--config", required=True, type=str, help="Path to experiment config file")
    parser.add_argument("--nnodes", required=True, type=int, help="Number of nodes")
    parser.add_argument("--ip", required=True, type=str, help="IP address of master node")
    parser.add_argument("--lancet-profile", action="store_true", help="Run profiling with RAF")
    parser.add_argument("--lancet-opt", action="store_true", help="Run lancet optimization on profiled models")
    parser.add_argument("--nsys-profile", action="store_true", help="Use with Nsight Systems profiler")
    args = parser.parse_args()
    return args

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

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(subprocess.check_output(COMMAND.split(),stderr=subprocess.STDOUT))[1:]
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for x in memory_use_info]
    return memory_use_values

def get_num_gpus():
    COMMAND = "nvidia-smi -L | wc -l"
    try:
        num_gpus = int(subprocess.check_output(COMMAND,stderr=subprocess.STDOUT, shell=True))
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    return num_gpus

def kill_process_by_name(name):
    for proc in psutil.process_iter():
        if name in proc.name():
            proc.kill()

def create_full_model_name(model_name, nheads, dmodel, nlayers, seqlen):
    full_model_name = f"{model_name}_nheads{nheads}_dmodel{dmodel}_nlayers{nlayers}_seqlen{seqlen}"
    return full_model_name

def create_descr(model_name, nheads, dmodel, nlayers, seqlen, batch_size, nnodes, instance_type):
    descr = f"{model_name}_nl{nlayers}_m{dmodel}_nh{nheads}_sl{seqlen}_b{batch_size}{'_fp16' if FP16 else ''}_{nnodes}x{instance_type}"
    return descr

def should_skip(dir_name, nnodes, instance_type, exp_config: ExpConfig, nsys_profile=False):
    output_dir_name = "optimized_profile"
    if exp_config.no_dw_scheduling:
        output_dir_name += "_no_dw_schedule"
    if exp_config.partition_type == "fixed_range":
        output_dir_name += "_part_fixed_range"
    elif exp_config.partition_type == "none":
        output_dir_name += "_part_none"
    current_out_dir = os.path.join(EXP_DIR_PREFIX, exp_config.moe_gate_type, "lancet", dir_name, output_dir_name)
    if nsys_profile:
        current_out_dir = os.path.join(current_out_dir, "nsys_profile")
    if os.path.exists(current_out_dir):
        return True
    _,_,_,_,_,_, nnodes, exp_instance_type = parse_specs(dir_name)
    if nnodes != nnodes or exp_instance_type != instance_type:
        return True
    return False

def get_opt_mod_path(dir_name, exp_config: ExpConfig):
    current_dir = os.path.join(EXP_DIR_PREFIX, exp_config.moe_gate_type, "lancet", dir_name)
    try:
        output_dir_name = "optimized"
        if exp_config.no_dw_scheduling:
            output_dir_name += "_no_dw_schedule"
        if exp_config.partition_type == "fixed_range":
            output_dir_name += "_part_fixed_range"
        elif exp_config.partition_type == "none":
            output_dir_name += "_part_none"
        mod_path = glob.glob(os.path.join(current_dir, output_dir_name, "*.mod"))[0]
    except IndexError:
        raise RuntimeError(f"No optimized module found in {os.path.join(current_dir, 'optimized')}")
    return mod_path

def parse_exp_configs(config_path, nnodes, instance_type) -> List[ExpConfig]:
    exp_configs = []
    with open(config_path, "r") as f:
        for line in f:
            if line:
                obj = json.loads(line)
                exp_name = obj['name'] + "_" + str(nnodes) + "x" + instance_type
                framework = obj['framework']
                moe_gate_type = obj['moe_gate_type']
                if 'max_partition' in obj:
                    max_partition = obj['max_partition']
                else:
                    max_partition = 8
                if 'no_dw_scheduling' in obj:
                    no_dw_scheduling = obj['no_dw_scheduling']
                else:
                    no_dw_scheduling = False
                if 'partition_type' in obj:
                    partition_type = obj['partition_type']
                else:
                    partition_type = "dp"
                exp_config = ExpConfig(exp_name, framework, moe_gate_type, max_partition, no_dw_scheduling, partition_type)
                exp_configs.append(exp_config)
    return exp_configs

def run_lancet_optimized_benchmarks(args, exp_config: ExpConfig):
    instance_type = get_instance_type()
    if should_skip(exp_config.exp_name, args.nnodes, instance_type, exp_config, args.nsys_profile):
        return
    model_name, nl, m, nh, sl, b, _, _ = parse_specs(exp_config.exp_name)
    descr = create_descr(model_name, nh, m, nl, sl, b, args.nnodes, instance_type)
    model_name = create_full_model_name(model_name, nh, m, nl, sl)
    print(f"Running {descr}", flush=True)
    opt_mod_path = get_opt_mod_path(exp_config.exp_name, exp_config)
    output_dir_name = "optimized_profile"
    if exp_config.no_dw_scheduling:
        output_dir_name += "_no_dw_schedule"
    if exp_config.partition_type == "fixed_range":
        output_dir_name += "_part_fixed_range"
    elif exp_config.partition_type == "none":
        output_dir_name += "_part_none"
    curr_dir_name = os.path.join(EXP_DIR_PREFIX, exp_config.moe_gate_type, "lancet", exp_config.exp_name, output_dir_name)
    if args.nsys_profile:
        print("Profiling with Nsight Systems", flush=True)
        curr_dir_name = os.path.join(curr_dir_name, "nsys_profile")
        os.makedirs(curr_dir_name)
        nsys_outpath = os.path.join(curr_dir_name, f"{descr}.nsys-rep")
        nsys_cmd = f"nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -c cudaProfilerApi --capture-range-end stop-shutdown -o {nsys_outpath}"
        p = subprocess.Popen(f"{nsys_cmd} ./mpi_wrapper.sh raf --model-name {model_name} --batch {b} {'--fp16' if FP16 else ''} --lancet-optimized-module {opt_mod_path} --nsys-profile --gate-type {exp_config.moe_gate_type}", shell=True)
    else:
        os.makedirs(curr_dir_name)
        p = subprocess.Popen(f"./mpi_wrapper.sh raf --model-name {model_name} --batch {b} {'--fp16' if FP16 else ''} --lancet-optimized-module {opt_mod_path} --gate-type {exp_config.moe_gate_type}", shell=True)
    max_mem = 0
    while p.poll() is None:
        time.sleep(2)
        mem = max(get_gpu_memory())
        max_mem = max(mem, max_mem)
    time.sleep(10)
    # move all generated files to current_run_dir
    subprocess.run(f"mv /models/log*.txt {curr_dir_name}", shell=True)
    # write peak memory to file
    with open(os.path.join(curr_dir_name, "peak_memory.txt"), "w") as f:
        f.write(str(max_mem))
        f.write("\n")
    time.sleep(1)

def run_lancet_profile(args, dir_name, gate_type):
    instance_type = get_instance_type()
    dir_path = os.path.join(EXP_DIR_PREFIX, gate_type, "lancet", dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        # already profiled
        return
    model_name, nl, m, nh, sl, b, _, _ = parse_specs(dir_name)
    descr = create_descr(model_name, nh, m, nl, sl, b, args.nnodes, instance_type)
    print(f"Profiling {descr}", flush=True)
    model_name = create_full_model_name(model_name, nh, m, nl, sl)
    assert not args.nsys_profile, "Nsight Systems profiling not supported for RAF profiling"
    p = subprocess.Popen(f"./mpi_wrapper.sh raf --model-name {model_name} --batch {b} {'--fp16' if FP16 else ''} --gate-type {gate_type} --lancet-profile", shell=True)
    max_mem = 0
    while p.poll() is None:
        time.sleep(2)
        mem = max(get_gpu_memory())
        max_mem = max(mem, max_mem)
    time.sleep(10)
    # move all generated files to current_run_dir
    subprocess.run(f"mv /models/log*.txt /models/*.mod /models/*.profile /models/timeline_*.json {dir_path}", shell=True)
    # write peak memory to file
    with open(os.path.join(dir_path, "peak_memory.txt"), "w") as f:
        f.write(str(max_mem))
        f.write("\n")
    time.sleep(1)

def run_raf_profile(args, dir_name, gate_type):
    instance_type = get_instance_type()
    curr_dir_name = os.path.join(EXP_DIR_PREFIX, gate_type, "raf", dir_name)
    if os.path.exists(curr_dir_name) and not args.nsys_profile:
        # already profiled
        return
    if args.nsys_profile:
        curr_dir_name = os.path.join(curr_dir_name, "nsys_profile")
        if os.path.exists(curr_dir_name):
            # already profiled
            return
    os.makedirs(curr_dir_name)
    model_name, nl, m, nh, sl, b, _, _ = parse_specs(dir_name)
    descr = create_descr(model_name, nh, m, nl, sl, b, args.nnodes, instance_type)
    print(f"Profiling {descr}", flush=True)
    model_name = create_full_model_name(model_name, nh, m, nl, sl)
    if args.nsys_profile:
        print("Profiling with Nsight Systems", flush=True)
        nsys_outpath = os.path.join(curr_dir_name, f"{descr}.nsys-rep")
        nsys_cmd = f"nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -c cudaProfilerApi --capture-range-end stop-shutdown -o {nsys_outpath}"
        p = subprocess.Popen(f"{nsys_cmd} ./mpi_wrapper.sh raf --model-name {model_name} --batch {b} {'--fp16' if FP16 else ''} --nsys-profile --gate-type {gate_type}", shell=True)
    else:
        p = subprocess.Popen(f"./mpi_wrapper.sh raf --model-name {model_name} --batch {b} {'--fp16' if FP16 else ''} --gate-type {gate_type}", shell=True)
    max_mem = 0
    while p.poll() is None:
        time.sleep(2)
        mem = max(get_gpu_memory())
        max_mem = max(mem, max_mem)
    time.sleep(10)
    # move all generated files to current_run_dir
    subprocess.run(f"mv /models/log*.txt {curr_dir_name}", shell=True)
    # write peak memory to file
    with open(os.path.join(curr_dir_name, "peak_memory.txt"), "w") as f:
        f.write(str(max_mem))
        f.write("\n")
    time.sleep(1)

def run_deepspeed_benchmarks(args, dir_name, gate_type):
    instance_type = get_instance_type()
    dir_path = os.path.join(EXP_DIR_PREFIX, gate_type, "deepspeed", dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif not args.nsys_profile:
        # already profiled
        return
    model_name, nl, m, nh, sl, b, _, _ = parse_specs(dir_name)
    descr = create_descr(model_name, nh, m, nl, sl, b, args.nnodes, instance_type)
    print(f"Profiling {descr}", flush=True)
    model_name = create_full_model_name(model_name, nh, m, nl, sl)
    if args.nsys_profile:
        print("Profiling with Nsight Systems", flush=True)
        dir_path = os.path.join(dir_path, "nsys_profile")
        if os.path.exists(dir_path):
            return
        os.makedirs(dir_path)
        nsys_outpath = os.path.join(dir_path, f"{descr}.nsys-rep")
        nsys_cmd = f"nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -c cudaProfilerApi --capture-range-end stop-shutdown -o {nsys_outpath}"
        p = subprocess.Popen(f"{nsys_cmd} ./mpi_wrapper.sh ds --model-name {model_name} --batch {b} {'--fp16' if FP16 else ''} --nsys-profile --gate-type {gate_type}", shell=True)
    else:
        p = subprocess.Popen(f"./mpi_wrapper.sh ds --model-name {model_name} --batch {b} {'--fp16' if FP16 else ''} --gate-type {gate_type}", shell=True)
    max_mem = 0
    while p.poll() is None:
        time.sleep(2)
        mem = max(get_gpu_memory())
        max_mem = max(mem, max_mem)
    time.sleep(10)
    # move all generated files to current_run_dir
    subprocess.run(f"mv /models/log*.txt {dir_path}", shell=True)
    # write peak memory to file
    with open(os.path.join(dir_path, "peak_memory.txt"), "w") as f:
        f.write(str(max_mem))
        f.write("\n")
    time.sleep(1)

def run_pt_benchmarks(args, dir_name, framework, gate_type):
    assert framework in ["tutel", "fastermoe"]
    instance_type = get_instance_type()
    dir_path = os.path.join(EXP_DIR_PREFIX, gate_type, framework, dir_name)
    model_name, nl, m, nh, sl, b, _, _ = parse_specs(dir_name)
    descr = create_descr(model_name, nh, m, nl, sl, b, args.nnodes, instance_type)
    print(f"Profiling {descr}", flush=True)
    model_name = create_full_model_name(model_name, nh, m, nl, sl)
    for ovlp_degree in BASELINE_CANDIDATE_PARTITONS:
        per_degree_dir_path = os.path.join(dir_path, f"ovlp_{ovlp_degree}")
        if not os.path.exists(per_degree_dir_path):
            os.makedirs(per_degree_dir_path)
        elif not args.nsys_profile:
            # already profiled
            continue
        if args.nsys_profile:
            print("Profiling with Nsight Systems", flush=True)
            per_degree_dir_path = os.path.join(per_degree_dir_path, "nsys_profile")
            if os.path.exists(per_degree_dir_path):
                continue
            os.makedirs(per_degree_dir_path)
            nsys_outpath = os.path.join(per_degree_dir_path, f"{descr}.nsys-rep")
            nsys_cmd = f"nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -c cudaProfilerApi --capture-range-end stop-shutdown -o {nsys_outpath}"
            p = subprocess.Popen(f"{nsys_cmd} ./mpi_wrapper.sh pt --model-name {model_name} --ip {args.ip} --moe-framework {framework} --overlap-degree {ovlp_degree} --batch {b} {'--fp16' if FP16 else ''} --nsys-profile --gate-type {gate_type}", shell=True)
        else:
            p = subprocess.Popen(f"./mpi_wrapper.sh pt --model-name {model_name} --ip {args.ip} --moe-framework {framework} --overlap-degree {ovlp_degree} --batch {b} {'--fp16' if FP16 else ''} --gate-type {gate_type}", shell=True)
        max_mem = 0
        while p.poll() is None:
            time.sleep(2)
            mem = max(get_gpu_memory())
            max_mem = max(mem, max_mem)
        time.sleep(10)
        # move all generated files to current_run_dir
        subprocess.run(f"mv /models/log*.txt {per_degree_dir_path}", shell=True)
        # write peak memory to file
        with open(os.path.join(per_degree_dir_path, "peak_memory.txt"), "w") as f:
            f.write(str(max_mem))
            f.write("\n")
        time.sleep(1)

def run_lancet_opt(args, exp_configs: List[ExpConfig]):
    n_gpus_available = get_num_gpus()
    gpu_in_use = [False] * n_gpus_available
    pbar = tqdm(total=len(exp_configs))
    processes = []
    for exp_config in exp_configs:
        if exp_config.framework == "lancet":
            instance_type = get_instance_type()
            exp_dir = os.path.join(EXP_DIR_PREFIX, exp_config.moe_gate_type, "lancet", exp_config.exp_name)
            output_dir_name = "optimized"
            if exp_config.no_dw_scheduling:
                output_dir_name += "_no_dw_schedule"
            if exp_config.partition_type == "fixed_range":
                output_dir_name += "_part_fixed_range"
            elif exp_config.partition_type == "none":
                output_dir_name += "_part_none"
            current_out_dir = os.path.join(exp_dir, output_dir_name)
            if os.path.exists(current_out_dir):
                pbar.update(1)
                continue
            model_name, nl, m, nh, sl, b, _, _ = parse_specs(exp_config.exp_name)
            descr = create_descr(model_name, nh, m, nl, sl, b, args.nnodes, instance_type)
            model_name = create_full_model_name(model_name, nh, m, nl, sl)
            # find available GPU
            gpu_id = -1
            for i in range(n_gpus_available):
                if not gpu_in_use[i]:
                    gpu_id = i
                    gpu_in_use[i] = True
                    break
            assert gpu_id != -1, "No available GPU"
            pbar.write(f"Optimizing {descr} on GPU {gpu_id}")
            additional_args = ""
            if exp_config.no_dw_scheduling:
                additional_args += " --no-dw-schedule"
            if exp_config.partition_type == "fixed_range":
                additional_args += " --fixed-range"
            elif exp_config.partition_type == "none":
                additional_args += " --no-partition"
            p = subprocess.Popen(f"python3 run_opt.py --dir {exp_dir} --device {gpu_id} --max-partition {exp_config.max_partition}{additional_args}", shell=True)
            processes.append((p, gpu_id))
            # wait for empty GPU if no GPU available
            all_gpus_in_use = True
            for i in range(n_gpus_available):
                if not gpu_in_use[i]:
                    all_gpus_in_use = False
                    break
            if all_gpus_in_use:
                while True:
                    for p in processes:
                        if p[0].poll() is not None:
                            gpu_in_use[p[1]] = False
                            processes.remove(p)
                            pbar.update(1)
                            break
    while processes:
        for p in processes:
            if p[0].poll() is not None:
                gpu_in_use[p[1]] = False
                processes.remove(p)
                pbar.update(1)
                break

if __name__ == "__main__":
    args = parse_args()
    config_name = os.path.basename(args.config).split('.')[0]
    EXP_DIR_PREFIX = os.path.join(EXP_DIR_PREFIX, config_name)
    exp_configs = parse_exp_configs(args.config, args.nnodes, get_instance_type())
    if args.lancet_opt:
        # run parallel optimization
        run_lancet_opt(args, exp_configs)
    else:
        for exp_config in exp_configs:
            if exp_config.framework == "lancet":
                if args.lancet_profile:
                    run_lancet_profile(args, exp_config.exp_name, exp_config.moe_gate_type)
                else:
                    run_lancet_optimized_benchmarks(args, exp_config)
            else:
                if args.lancet_profile:
                    continue
                if exp_config.framework == "deepspeed":
                    run_deepspeed_benchmarks(args, exp_config.exp_name, exp_config.moe_gate_type)
                elif exp_config.framework == "raf":
                    run_raf_profile(args, exp_config.exp_name, exp_config.moe_gate_type)
                else:
                    run_pt_benchmarks(args, exp_config.exp_name, exp_config.framework, exp_config.moe_gate_type)
