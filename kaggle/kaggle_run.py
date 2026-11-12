#!/usr/bin/env python3
"""Run Mass Prime Siever inside one Kaggle session and leave a resumable state in the output.

One siever process per GPU, each with its own slice of [P0, P1). Before the session limit the
processes get SIGINT, which makes them checkpoint into core_*.p. Everything is kept under
/kaggle/working/siever_state, which Kaggle saves as the notebook's output. For the next session,
add the previous version's output as an input ("Add Input" -> your notebook) and run again:
the state is copied back and every slice resumes where it stopped.

Example (first session):
    python kaggle_run.py --b 2 --c -25 --n0 3600000 --n1 5000000 --mod 2 --res 1 \
        --p0 1000000000000 --p1 2000000000000 --hours 11.5
"""
import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "build", "mass-prime-siever")
STATE_FILES = ("worktodo.txt", "prime.ini", "siever.log", "elapsed.checkpoint",
               "relations.p", "results.json")


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU"))
    except (OSError, subprocess.CalledProcessError):
        return 0


def restore(state_dir, input_root):
    """Copy the newest previous state from /kaggle/input into state_dir (only if state_dir is empty)."""
    if os.path.isdir(state_dir) and os.listdir(state_dir):
        return False
    candidates = glob.glob(os.path.join(input_root, "**", "siever_state"), recursive=True)
    if not candidates:
        return False
    newest = max(candidates, key=lambda d: max((os.path.getmtime(f) for f in
                                                glob.glob(os.path.join(d, "**", "*"), recursive=True)),
                                               default=0))
    shutil.copytree(newest, state_dir, dirs_exist_ok=True)
    print(f"Restored state from {newest}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--c", type=int, default=-25)
    ap.add_argument("--n0", type=int, default=3600000)
    ap.add_argument("--n1", type=int, default=5000000)
    ap.add_argument("--mod", type=int, default=2)
    ap.add_argument("--res", type=int, default=1)
    ap.add_argument("--p0", type=int, help="first sieve prime (new run only)")
    ap.add_argument("--p1", type=int, help="end of sieve prime range (new run only)")
    ap.add_argument("--hours", type=float, default=11.5, help="stop and checkpoint after this long")
    ap.add_argument("--gpus", type=int, default=None, help="number of GPU workers (default: all)")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="CPU threads per worker (default: spread the free cores over the workers)")
    ap.add_argument("--state", default="/kaggle/working/siever_state")
    ap.add_argument("--input", default="/kaggle/input")
    args = ap.parse_args()

    if not os.path.exists(BINARY):
        subprocess.run(["bash", os.path.join(ROOT, "kaggle", "setup.sh")], check=True)

    restored = restore(args.state, args.input)
    os.makedirs(args.state, exist_ok=True)
    slots = sorted(d for d in os.listdir(args.state) if d.startswith("gpu"))

    if not slots:
        if args.p0 is None or args.p1 is None:
            sys.exit("No previous state found: give --p0 and --p1 for a new run.")
        n = args.gpus if args.gpus is not None else max(1, gpu_count())
        step = (args.p1 - args.p0) // n
        cpu = args.cpu_threads
        if cpu is None:
            cpu = max(0, (os.cpu_count() or 2) - n) // n  # leave one core per GPU feeder thread
        for i in range(n):
            lo = args.p0 + i * step
            hi = args.p1 if i == n - 1 else lo + step
            d = os.path.join(args.state, f"gpu{i}")
            os.makedirs(d)
            line = f"Factor={args.b},{args.c},{lo},{hi},{args.n0},{args.n1}"
            if args.mod > 1:
                line += f",{args.mod},{args.res}"
            with open(os.path.join(d, "worktodo.txt"), "w") as f:
                f.write(line + "\n")
            with open(os.path.join(d, "prime.ini"), "w") as f:
                f.write(f"CPU_count={cpu}\nMemoryAllowed=64\nUseGPU=1\nGPU_device={i}\n")
            print(f"gpu{i}: {line}  (CPU_count={cpu})")
        slots = sorted(d for d in os.listdir(args.state) if d.startswith("gpu"))
    elif not restored:
        print("Continuing state already in", args.state)

    procs = []
    for slot in slots:
        d = os.path.join(args.state, slot)
        if os.path.exists(os.path.join(d, "results.json")):
            print(f"{slot}: already finished, results.json present")
            continue
        log = open(os.path.join(d, "stdout.txt"), "a")
        procs.append((slot, subprocess.Popen([BINARY], cwd=d, stdin=subprocess.DEVNULL,
                                              stdout=log, stderr=subprocess.STDOUT)))
        print(f"{slot}: started")

    deadline = time.time() + args.hours * 3600
    while procs and time.time() < deadline:
        if all(p.poll() is not None for _, p in procs):
            break
        time.sleep(30)
    for slot, p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for slot, p in procs:
        try:
            p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            p.kill()
        print(f"{slot}: exit code {p.returncode}")

    for slot in slots:
        d = os.path.join(args.state, slot)
        tail = subprocess.run(["tail", "-n", "3", os.path.join(d, "siever.log")],
                              capture_output=True, text=True).stdout
        print(f"--- {slot} ---\n{tail}")


if __name__ == "__main__":
    main()
