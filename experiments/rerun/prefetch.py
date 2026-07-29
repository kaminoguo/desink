"""Prefetch all Pythia checkpoint weights (safetensors only) in compute-priority order.

Runs standalone in background; writes a .done marker per (model, step) that the
orchestrator polls before dispatching that job.
No bare except: every failure is logged with full traceback and retried.
"""
import os, sys, json, time, traceback
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HOME", "/workspace/desink/hf")

from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000,
         16000, 32000, 64000, 128000, 143000]
READY = "/workspace/desink/ready"
os.makedirs(READY, exist_ok=True)

# download order: sanity ckpts -> 1.4b (longest compute) -> 410m -> rest of 160m
ORDER = (
    [("EleutherAI/pythia-160m", s) for s in (143000, 0)]
    + [("EleutherAI/pythia-1.4b", s) for s in STEPS]
    + [("EleutherAI/pythia-410m", s) for s in STEPS]
    + [("EleutherAI/pythia-160m", s) for s in STEPS if s not in (143000, 0)]
)

ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json",
         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def marker(model, step):
    return f"{READY}/{model.split('/')[-1]}_step{step}.done"


def already_done(model, step):
    """Skip weights whose result already exists — after a box loss the surviving
    results are restored first, so only the unfinished checkpoints are fetched."""
    short = model.split("/")[-1]
    e1 = f"/workspace/desink/results/e1_{short}_step{step}.json"
    e3 = f"/workspace/desink/results/e3_{short}_step{step}.json"
    if not os.path.exists(e1):
        return False
    if short == "pythia-1.4b" and step in (512, 8000, 143000):
        return os.path.exists(e3)          # still needed by the E3 sweep
    return True


def fetch(job):
    model, step = job
    mk = marker(model, step)
    if os.path.exists(mk):
        return
    if already_done(model, step):
        print(f"[dl] skip {model} step{step} (result exists)", flush=True)
        return
    for attempt in range(4):
        try:
            t = time.time()
            p = snapshot_download(model, revision=f"step{step}",
                                  allow_patterns=ALLOW, max_workers=8)
            with open(mk, "w") as f:
                json.dump({"path": p, "sec": round(time.time() - t, 1)}, f)
            print(f"[dl] {model} step{step} {time.time()-t:.0f}s", flush=True)
            return
        except Exception:
            print(f"[dl] FAIL attempt{attempt} {model} step{step}", flush=True)
            traceback.print_exc()
            time.sleep(10 * (attempt + 1))
    print(f"[dl] GIVING UP {model} step{step}", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(fetch, ORDER))
    print("[dl] ALL DONE", flush=True)
