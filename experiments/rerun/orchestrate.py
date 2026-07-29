"""4-GPU work-queue dispatcher for the T1 rerun.

Jobs are ordered longest-first (1.4b -> 410m -> 160m) so the makespan is not
dominated by a straggler.  A slot skips over jobs whose weights have not been
prefetched yet rather than blocking, so compute overlaps the 158 GB download.

Idempotent: a job whose result JSON already exists is skipped, so the watchdog
can restart this process at any time without losing work.
"""
import os, sys, json, time, subprocess, threading

ROOT = "/workspace/desink"
READY = f"{ROOT}/ready"
RES = f"{ROOT}/results"
LOGS = f"{ROOT}/logs/jobs"
PY = f"{ROOT}/venv/bin/python"
WORKER = f"{ROOT}/code/t1_worker.py"
N_GPU = int(os.environ.get("N_GPU", "4"))
MAX_RETRY = 2

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000,
         16000, 32000, 64000, 128000, 143000]


def jobs():
    j = []
    for m in ("pythia-1.4b", "pythia-410m", "pythia-160m"):
        for s in STEPS:
            j.append({"model": f"EleutherAI/{m}", "short": m, "step": s, "mode": "e1",
                      "name": f"e1_{m}_step{s}"})
    for s in (512, 8000, 143000):                     # E3 token-count sweep
        j.append({"model": "EleutherAI/pythia-1.4b", "short": "pythia-1.4b", "step": s,
                  "mode": "e3", "name": f"e3_pythia-1.4b_step{s}"})
    return j


def ready(j):
    return os.path.exists(f"{READY}/{j['short']}_step{j['step']}.done")


def done(j):
    return os.path.exists(f"{RES}/{j['name']}.json")


def run(j, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [PY, WORKER, "--model", j["model"], "--revision", f"step{j['step']}",
           "--step", str(j["step"]), "--mode", j["mode"],
           "--out", f"{RES}/{j['name']}.json"]
    with open(f"{LOGS}/{j['name']}.log", "w") as f:
        return subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)


def main():
    for d in (RES, LOGS):
        os.makedirs(d, exist_ok=True)
    Q = jobs()
    lock = threading.Lock()
    state = {j["name"]: {"status": "done" if done(j) else "pending", "tries": 0,
                         "gpu": None, "t": None} for j in Q}

    def claim():
        """First job that is ready, not done, not running, under the retry cap."""
        with lock:
            for j in Q:
                st = state[j["name"]]
                if st["status"] in ("pending", "retry") and st["tries"] < MAX_RETRY and ready(j):
                    st["status"], st["tries"] = "running", st["tries"] + 1
                    return j
        return None

    def slot(gpu):
        while True:
            j = claim()
            if j is None:
                with lock:
                    left = [n for n, s in state.items() if s["status"] in ("pending", "retry")
                            and s["tries"] < MAX_RETRY]
                if not left:
                    return
                time.sleep(15)
                continue
            t0 = time.time()
            state[j["name"]].update(gpu=gpu, t=t0)
            rc = run(j, gpu)
            ok = rc == 0 and done(j)
            with lock:
                state[j["name"]].update(status="done" if ok else "retry",
                                        sec=round(time.time() - t0))
            print(f"[gpu{gpu}] {j['name']} rc={rc} ok={ok} {time.time()-t0:.0f}s", flush=True)

    threads = [threading.Thread(target=slot, args=(g,), daemon=True) for g in range(N_GPU)]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        with lock:
            snap = {"ts": time.time(),
                    "done": sum(1 for s in state.values() if s["status"] == "done"),
                    "running": {n: s["gpu"] for n, s in state.items() if s["status"] == "running"},
                    "failed": [n for n, s in state.items()
                               if s["status"] == "retry" and s["tries"] >= MAX_RETRY],
                    "total": len(Q), "state": state}
        with open(f"{ROOT}/status.json", "w") as f:
            json.dump(snap, f, indent=1)
        time.sleep(20)

    print("ORCHESTRATOR EXIT", json.dumps({n: s["status"] for n, s in state.items()}), flush=True)


if __name__ == "__main__":
    main()
