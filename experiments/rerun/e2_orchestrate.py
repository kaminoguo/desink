"""4-GPU dispatcher for E2 (OLMo-2 early-training).  Same contract as
orchestrate.py: idempotent, skips jobs whose weights are not prefetched yet."""
import os, json, time, subprocess, threading

ROOT = "/workspace/desink"
READY, RES, LOGS = f"{ROOT}/ready_e2", f"{ROOT}/results", f"{ROOT}/logs/jobs"
PY, WORKER = f"{ROOT}/venv/bin/python", f"{ROOT}/code/t1_worker.py"
CORPUS = f"{ROOT}/corpus/corpus_olmo.npy"
REPO = "allenai/OLMo-2-0425-1B-early-training"
N_GPU, MAX_RETRY = int(os.environ.get("N_GPU", "4")), 2

from e2_setup import REVS


def jobs():
    return [{"rev": r, "step": int(r.split("-step")[1].split("-")[0]),
             "name": f"e2_olmo1b_{r}"} for r in REVS]


def main():
    os.makedirs(LOGS, exist_ok=True)
    Q, lock = jobs(), threading.Lock()
    state = {j["name"]: {"status": "done" if os.path.exists(f"{RES}/{j['name']}.json")
                         else "pending", "tries": 0} for j in Q}

    def claim():
        with lock:
            for j in Q:
                s = state[j["name"]]
                if s["status"] in ("pending", "retry") and s["tries"] < MAX_RETRY \
                        and os.path.exists(f"{READY}/{j['rev']}.done"):
                    s["status"], s["tries"] = "running", s["tries"] + 1
                    return j
        return None

    def slot(gpu):
        while True:
            j = claim()
            if j is None:
                with lock:
                    left = [n for n, s in state.items()
                            if s["status"] in ("pending", "retry") and s["tries"] < MAX_RETRY]
                if not left:
                    return
                time.sleep(15)
                continue
            t0 = time.time()
            cmd = [PY, WORKER, "--model", REPO, "--revision", j["rev"],
                   "--step", str(j["step"]), "--mode", "e1", "--corpus", CORPUS,
                   "--out", f"{RES}/{j['name']}.json"]
            with open(f"{LOGS}/{j['name']}.log", "w") as f:
                rc = subprocess.call(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                     stdout=f, stderr=subprocess.STDOUT)
            ok = rc == 0 and os.path.exists(f"{RES}/{j['name']}.json")
            with lock:
                state[j["name"]]["status"] = "done" if ok else "retry"
            print(f"[gpu{gpu}] {j['name']} rc={rc} ok={ok} {time.time()-t0:.0f}s", flush=True)

    ts = [threading.Thread(target=slot, args=(g,), daemon=True) for g in range(N_GPU)]
    for t in ts:
        t.start()
    while any(t.is_alive() for t in ts):
        json.dump({"ts": time.time(), "state": state}, open(f"{ROOT}/status_e2.json", "w"), indent=1)
        time.sleep(20)
    print("E2 ORCHESTRATOR EXIT", json.dumps({n: s["status"] for n, s in state.items()}), flush=True)


if __name__ == "__main__":
    main()
