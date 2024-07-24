import os
import multiprocessing
import subprocess
import time

WATCH_DIR = "/models/grid_profile"
CMD = "python3 run_opt.py --dir {} --device {};"

processed_dirs = set()

def fetch_jobs():
    fns = [x for x in os.listdir(WATCH_DIR) 
            if os.path.isdir(os.path.join(WATCH_DIR, x)) and 
                not os.path.isdir(os.path.join(WATCH_DIR, x, "optimized")) and 
                x not in processed_dirs]
    return fns


def worker_thread(queue: multiprocessing.Queue, rqueue: multiprocessing.Queue, device_id: int):
    while True:
        fn = queue.get()
        if fn is None:
            break
        cmd = CMD.format(fn, device_id)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rqueue.put((device_id, fn))

def main():
    idling_devices = set(range(8))
    # launch 8 worker threads
    queues = [multiprocessing.Queue() for _ in range(8)]
    rqueue = multiprocessing.Queue()
    ps = []
    for i in range(8):
        p = multiprocessing.Process(target=worker_thread, args=(queues[i], rqueue, i))
        p.start()
        ps.append(p)
    try:
        while True:
            new_jobs = fetch_jobs()
            for fn in new_jobs:
                if len(idling_devices) == 0:
                    break
                device_id = idling_devices.pop()
                print(f"Launching {fn} on device {device_id}.", flush=True)
                queues[device_id].put(fn)
                processed_dirs.add(fn)
            while not rqueue.empty():
                device_id, fn = rqueue.get()
                idling_devices.add(device_id)
                print("Device {} finished running {}.".format(device_id, fn), flush=True)
            time.sleep(10)
    except KeyboardInterrupt:
        print("Terminating...", flush=True)
        for q in queues:
            q.put(None)
        for p in ps:
            p.join()
    for p in ps:
        p.join()

if __name__ == "__main__":
    main()

