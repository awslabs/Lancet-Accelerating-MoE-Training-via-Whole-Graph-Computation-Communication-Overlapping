import subprocess
import os

CMD = " python3 run_opt.py --dir {} --device {};"
EXP_DIRS = "/models/grid_profile"

fns = [x for x in os.listdir(EXP_DIRS) if os.path.isdir(os.path.join(EXP_DIRS, x)) and os.path.isdir(os.path.join(EXP_DIRS, x, "optimized"))]
fns = sorted(fns)

ngpus = 8
per_gpu_cmds = {}

for i, fn in enumerate(fns):
    device_id = i % ngpus
    if device_id not in per_gpu_cmds:
        per_gpu_cmds[device_id] = ""
    per_gpu_cmds[device_id] += CMD.format(fn, device_id)

procs = []
for device_id, cmd in per_gpu_cmds.items():
    cmd = cmd.strip()
    p = subprocess.Popen(cmd, shell=True)
    procs.append(p)
for p in procs:
    p.wait()
