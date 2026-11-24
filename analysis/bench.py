"""Crawler benchmark on the latency-emulating testbed.

For every (crawler, concurrency, latency, repetition) a fresh testbed server is
started with seed = repetition, the crawler runs once, and the crawler's own CPU
time and peak resident memory are read from os.wait4 (per-process rusage).
Results are appended to results/logs/bench.jsonl (one JSON object per run).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
LOG = ROOT / "results" / "logs" / "bench.jsonl"
PIN = False  # set by --pin: server on CPU 0, crawler on CPU 1
PORT = 8088


def budget(c: int) -> int:
    """Page budget: about 40 latency rounds per run, at least 120, at most 2000."""
    return int(min(2000, max(120, 40 * c)))


def wait_port(port, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError("server did not start")


def crawler_cmd(name, c, maxp):
    seed = f"http://127.0.0.1:{PORT}/"
    if name == "go":
        return [str(BIN / "hfhcrawl"), "-seed", seed, "-c", str(c), "-max", str(maxp), "-retries", "1", "-timeout", "30s"]
    mode = {"asyncio": "asyncio", "threads": "threads"}[name]
    return [sys.executable, "-I", str(ROOT / "baselines" / "pycrawl.py"), "--mode", mode, "--seed", seed,
            "--c", str(c), "--max", str(maxp), "--retries", "1", "--timeout", "30"]


def run_one(name, c, median, sigma, p503, rep, maxp, tag):
    srv_pre = ["taskset", "-c", "0"] if PIN else []
    cr_pre = ["taskset", "-c", "1"] if PIN else []
    srv = subprocess.Popen(srv_pre + [str(BIN / "testbed"), "-addr", f"127.0.0.1:{PORT}", "-site", str(ROOT / "data/site/site.json"),
                            "-median", str(median), "-sigma", str(sigma), "-p503", str(p503), "-seed", str(rep)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_port(PORT)
        p = subprocess.Popen(cr_pre + crawler_cmd(name, c, maxp), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, status, ru = os.wait4(p.pid, 0)
        out = p.stdout.read().decode().strip().splitlines()
        err = p.stderr.read().decode()[-500:]
        res = json.loads(out[-1]) if out else {"error": err}
        res.update({"tag": tag, "impl": name, "median_s": median, "sigma": sigma, "p503": p503, "rep": rep,
                    "cpu_s": ru.ru_utime + ru.ru_stime, "max_rss_mb": ru.ru_maxrss / 1024, "exit": status, "pinned": PIN})
        return res
    finally:
        srv.terminate()
        srv.wait()


def main():
    global PIN, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["smoke", "main", "sensitivity", "startup"], required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--log", default=str(LOG))
    args = ap.parse_args()
    PIN = args.pin
    LOG = Path(args.log)
    runs = []
    if args.plan == "smoke":
        for name in ["go", "asyncio", "threads"]:
            runs.append((name, 10, 0.05, 0.5, 0.02, 1, 200, "smoke"))
    elif args.plan == "main":
        for rep in range(1, args.reps + 1):
            for c in [1, 5, 10, 25, 50, 100]:
                for name in ["go", "asyncio", "threads"]:
                    runs.append((name, c, 1.0, 0.5, 0.02, rep, budget(c), "main"))
    elif args.plan == "startup":
        # fixed start-up cost: one page at short latency, so CPU time is almost all interpreter/runtime start-up
        for rep in range(1, args.reps + 1):
            for name in ["go", "asyncio", "threads"]:
                runs.append((name, 1, 0.05, 0.5, 0.0, rep, 1, "startup"))
    else:
        for rep in range(1, args.reps + 1):
            for median in [0.25, 0.5, 1.0, 2.0]:
                for name in ["go", "asyncio", "threads"]:
                    runs.append((name, 20, median, 0.5, 0.02, rep, 800, "sensitivity"))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(runs, 1):
        res = run_one(*r)
        with LOG.open("a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[{i}/{len(runs)}] {r[0]:8s} c={r[1]:3d} med={r[2]} rep={r[5]} -> "
              f"{res.get('pages_per_min', 0):8.1f} p/min cpu={res['cpu_s']:.2f}s rss={res['max_rss_mb']:.1f}MB "
              f"ok={res.get('ok')} att={res.get('attempted')}", flush=True)


if __name__ == "__main__":
    main()
