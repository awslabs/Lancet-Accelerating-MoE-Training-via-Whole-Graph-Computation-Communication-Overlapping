import subprocess
import os

EXP_DIRS = "/models/grid_profile"

fns = [x for x in os.listdir(EXP_DIRS) if os.path.isdir(os.path.join(EXP_DIRS, x)) and os.path.isdir(os.path.join(EXP_DIRS, x, "optimized"))]
fns = sorted(fns)

procs = []
for i, fn in enumerate(fns):
    device_id = i % 8
    cmd = ["python3", "run_opt.py", "--dir", fn, "--device", str(device_id)]
    p = subprocess.Popen(cmd)
    procs.append(p)
for p in procs:
    p.wait()
