import os
import subprocess
import psutil
import time
import glob
from tqdm import tqdm

MODEL_NAME = "gpt2_moe"
DMODEL = [512, 768, 1024, 1600]
NLAYERS = [4, 8, 12, 16, 24, 36]
BATCH_SIZE = [1, 2, 4, 8, 16, 24, 32, 48, 64, 72]
SEQ_LEN = 512
NNODES = 4
FP16 = False

OUT_DIR = "/models/grid_profile"

MAX_TIMEOUT = 2 * 60 * 60 # 2 hours

NHEADS_MAP = {
    512: 16,
    768: 16,
    1024: 16,
    1600: 16,
}

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

def kill_process_by_name(name):
    for proc in psutil.process_iter():
        if name in proc.name():
            proc.kill()

def create_model_name(dmodel, nlayers):
    nheads = NHEADS_MAP[dmodel]
    model_name = f"{MODEL_NAME}_nheads{nheads}_dmodel{dmodel}_nlayers{nlayers}_seqlen{SEQ_LEN}"
    return model_name

def create_descr(dmodel, nlayers, batch_size, instance_type):
    nheads = NHEADS_MAP[dmodel]
    descr = f"{MODEL_NAME}_nl{nlayers}_m{dmodel}_nh{nheads}_sl{SEQ_LEN}_b{batch_size}{'_fp16' if FP16 else ''}_{NNODES}x{instance_type}"
    return descr

def is_success(current_out_dir):
    if len(glob.glob(os.path.join(current_out_dir, "*.mod"))) == 0 or len(glob.glob(os.path.join(current_out_dir, "*.profile"))) == 0:
        return False
    return True

def should_skip(dmodel, nlayers, batch_size, instance_type):
    descr = create_descr(dmodel, nlayers, batch_size, instance_type)
    current_out_dir = os.path.join(OUT_DIR, descr)
    if os.path.exists(current_out_dir):
        return True
    # check if any larger batch size is successful
    for bs in BATCH_SIZE:
        if bs > batch_size:
            descr = create_descr(dmodel, nlayers, bs, instance_type)
            current_out_dir = os.path.join(OUT_DIR, descr)
            if os.path.exists(current_out_dir) and is_success(current_out_dir):
                return True
    # check if any smaller batch size or smaller dmodel or smaller nlayers OOMed
    for bs in BATCH_SIZE:
        for d in DMODEL:
            for l in NLAYERS:
                if bs <= batch_size and d <= dmodel and l <= nlayers and not (bs == batch_size and d == dmodel and l == nlayers):
                    descr = create_descr(d, l, bs, instance_type)
                    current_out_dir = os.path.join(OUT_DIR, descr)
                    if os.path.exists(current_out_dir) and not is_success(current_out_dir):
                        return True
    return False

def run_grid_profile():
    instance_type = get_instance_type()
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    with tqdm(total=len(DMODEL)*len(NLAYERS)*len(BATCH_SIZE)) as pbar:
        for dmodel in DMODEL:
            for nlayers in NLAYERS:
                for i, batch_size in enumerate(reversed(BATCH_SIZE)):
                    descr = create_descr(dmodel, nlayers, batch_size, instance_type)
                    current_run_dir = os.path.join(OUT_DIR, descr)
                    if should_skip(dmodel, nlayers, batch_size, instance_type):
                        pbar.update(1)
                        continue
                    os.makedirs(current_run_dir)
                    print(f"Running {descr}", flush=True)
                    pbar.set_description(descr)
                    model_name = create_model_name(dmodel, nlayers)
                    p = subprocess.Popen(f"./mpi_wrapper_grid_profile.sh raf --model-name {model_name} --batch {batch_size} {'--fp16' if FP16 else ''} --lancet-profile", shell=True)
                    max_mem = 0
                    start_time = time.time()
                    while p.poll() is None:
                        time.sleep(2)
                        mem = max(get_gpu_memory())
                        max_mem = max(mem, max_mem)
                        if time.time() - start_time > MAX_TIMEOUT:
                            print(f"Killing {descr} due to timeout", flush=True)
                            kill_process_by_name("mpi_wrapper")
                            kill_process_by_name("benchmark_raf")
                            break
                    time.sleep(10)
                    # test if success by looking for *.mod or *.profile
                    success = False
                    if len(glob.glob("/models/*.mod")) != 0 and len(glob.glob("/models/*.profile")) != 0:
                        print(f"Success: {descr}", flush=True)
                        success = True
                    # move all generated files to current_run_dir
                    subprocess.run(f"mv /models/*.mod /models/*.profile /models/timeline_*.json /models/log*.txt {current_run_dir}", shell=True)
                    # write peak memory to file
                    with open(os.path.join(current_run_dir, "peak_memory.txt"), "w") as f:
                        f.write(str(max_mem))
                        f.write("\n")
                    time.sleep(1)
                    if success:
                        # scp the dir to the optimizing node
                        subprocess.run(f"scp -r {current_run_dir} 172.31.21.100:{current_run_dir}", shell=True)
                        # no need to run smaller batch sizes
                        pbar.update(len(BATCH_SIZE) - i)
                        break
                    else:
                        # try smaller batch size
                        pbar.update(1)

if __name__ == "__main__":
    run_grid_profile()
