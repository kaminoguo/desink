#!/usr/bin/env python3
"""
Exp 77: lm_head Spectral Evolution vs Carrier Direction
=========================================================
Is sub-random sem=0.53 because lm_head actively rotates away from carrier,
or because carrier was always in lm_head null space?

For ~20 key checkpoints:
  1. L12 top-1 PC (carrier direction)
  2. lm_head SVD top-20 left singular vectors
  3. max cos(carrier, u_i) — best alignment
  4. sum cos²(carrier, u_i) — energy in lm_head top-20 subspace
  5. Same for random baseline
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time, os, gc, json, shutil

DEVICE = "cuda"
# os.environ["HF_HOME"] = ...  # set if needed
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "exp77_output.log")

MODEL_NAME = "EleutherAI/pythia-410m"
TARGET_LAYER = 12
N_SV = 20
N_RANDOM = 30

KEY_STEPS = [
    0, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000,
    8000, 10000, 20000, 50000, 100000, 143000,
]

CALIB_TEXTS = [
    "The history of artificial intelligence began in antiquity with myths",
    "In mathematics a proof is an inferential argument for a mathematical statement",
    "The global economy is interconnected through trade finance and technology",
    "Quantum mechanics is a fundamental theory in physics that describes nature",
    "Machine learning algorithms build a model based on sample data",
    "Climate change refers to long-term shifts in temperatures and weather",
    "The structure of DNA was discovered by Watson and Crick in 1953",
    "Philosophy is the study of general and fundamental questions about existence",
]


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def free_all():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clean_hf_cache():
    cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            path = os.path.join(cache_dir, item)
            if item.startswith("models--"):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except:
                    pass


def get_layers(model):
    for chain in ['gpt_neox.layers', 'model.layers', 'transformer.h']:
        obj = model
        ok = True
        for a in chain.split('.'):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and hasattr(obj, '__len__'):
            return list(obj)
    raise ValueError(f"No layers in {type(model)}")


def get_lm_head(model):
    if hasattr(model, 'embed_out'):
        return model.embed_out.weight.data.float()
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight.data.float()
    raise ValueError("Cannot find lm_head")


def measure(model, tokenizer, texts, device):
    layers = get_layers(model)
    li = TARGET_LAYER

    # Collect L12 activations
    hidden = []
    def hook_fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hidden.append(h.detach().float().cpu())
    handle = layers[li].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**ids)
    handle.remove()

    H = torch.cat(hidden, dim=1).squeeze(0)
    H_c = H - H.mean(0, keepdim=True)
    _, S_h, Vt_h = torch.linalg.svd(H_c, full_matrices=False)
    v_carrier = Vt_h[0]  # [d]
    carrier_energy = (S_h[0]**2).item() / (S_h**2).sum().item()

    # lm_head SVD
    lm_head = get_lm_head(model).cpu()  # [vocab, d]
    U, S_lm, Vt_lm = torch.linalg.svd(lm_head, full_matrices=False)
    # Left singular vectors U[:, i] are in vocab space
    # Right singular vectors Vt_lm[i] are in hidden space — these we compare with carrier
    top_sv = Vt_lm[:N_SV]  # [N_SV, d]

    # cos(carrier, each top SV)
    cos_vals = (top_sv @ v_carrier).abs()  # [N_SV]
    max_cos = cos_vals.max().item()
    sum_cos2 = (cos_vals ** 2).sum().item()
    argmax_cos = cos_vals.argmax().item()

    # lm_head singular value spectrum
    sv_spectrum = S_lm[:N_SV].tolist()

    # Random baseline
    rand_max_cos = []
    rand_sum_cos2 = []
    for _ in range(N_RANDOM):
        rv = torch.randn_like(v_carrier)
        rv = rv / rv.norm()
        rc = (top_sv @ rv).abs()
        rand_max_cos.append(rc.max().item())
        rand_sum_cos2.append((rc ** 2).sum().item())

    result = {
        "carrier_energy": carrier_energy,
        "max_cos": max_cos,
        "argmax_cos": int(argmax_cos),
        "sum_cos2": sum_cos2,
        "rand_max_cos_mean": float(np.mean(rand_max_cos)),
        "rand_max_cos_std": float(np.std(rand_max_cos)),
        "rand_sum_cos2_mean": float(np.mean(rand_sum_cos2)),
        "rand_sum_cos2_std": float(np.std(rand_sum_cos2)),
        "cos_per_sv": cos_vals.tolist(),
        "sv_spectrum": sv_spectrum,
    }

    del H, H_c, S_h, Vt_h, lm_head, U, S_lm, Vt_lm, top_sv, hidden
    gc.collect()
    return result


def main():
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 70)
    log("EXP 77: lm_head ALIGNMENT vs CARRIER DIRECTION")
    log("=" * 70)
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Model: {MODEL_NAME}, L{TARGET_LAYER}")
    log(f"lm_head SVD: top-{N_SV} right singular vectors")
    log(f"Random baseline: {N_RANDOM} samples")
    log(f"Checkpoints: {len(KEY_STEPS)}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    json_path = os.path.join(LOG_DIR, "exp77_data.json")

    for si, step in enumerate(KEY_STEPS):
        log(f"\n  [{si+1}/{len(KEY_STEPS)}] Step {step}")

        try:
            ts = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, revision=f"step{step}",
                torch_dtype=torch.float32, trust_remote_code=True
            ).to(DEVICE)
            model.eval()

            result = measure(model, tokenizer, CALIB_TEXTS, DEVICE)
            result["step"] = step
            all_results.append(result)

            elapsed = time.time() - ts

            z_max = (result["max_cos"] - result["rand_max_cos_mean"]) / result["rand_max_cos_std"] if result["rand_max_cos_std"] > 0 else 0
            z_sum = (result["sum_cos2"] - result["rand_sum_cos2_mean"]) / result["rand_sum_cos2_std"] if result["rand_sum_cos2_std"] > 0 else 0

            log(f"    max_cos={result['max_cos']:.4f} (SV{result['argmax_cos']+1})  "
                f"rand={result['rand_max_cos_mean']:.4f}±{result['rand_max_cos_std']:.4f}  "
                f"z={z_max:+.1f}")
            log(f"    sum_cos²={result['sum_cos2']:.4f}  "
                f"rand={result['rand_sum_cos2_mean']:.4f}±{result['rand_sum_cos2_std']:.4f}  "
                f"z={z_sum:+.1f}  energy={result['carrier_energy']*100:.1f}%")
            log(f"    ({elapsed:.1f}s)")

            del model
            free_all()

        except Exception as e:
            log(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            free_all()

        if (si + 1) % 5 == 0:
            clean_hf_cache()

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    log(f"\n{'='*70}")
    log(f"CARRIER vs lm_head ALIGNMENT TRAJECTORY")
    log(f"{'='*70}")
    log(f"\n  {'Step':>7s}  {'MaxCos':>7s}  {'RandMax':>8s}  {'Z':>6s}  {'SumCos2':>8s}  {'RandSum':>8s}  {'Z':>6s}  {'CE':>5s}")
    log(f"  {'-'*65}")
    for r in all_results:
        z_max = (r["max_cos"] - r["rand_max_cos_mean"]) / r["rand_max_cos_std"] if r["rand_max_cos_std"] > 0 else 0
        z_sum = (r["sum_cos2"] - r["rand_sum_cos2_mean"]) / r["rand_sum_cos2_std"] if r["rand_sum_cos2_std"] > 0 else 0
        log(f"  {r['step']:>7d}  {r['max_cos']:>7.4f}  {r['rand_max_cos_mean']:>8.4f}  {z_max:>+5.1f}  "
            f"{r['sum_cos2']:>8.4f}  {r['rand_sum_cos2_mean']:>8.4f}  {z_sum:>+5.1f}  {r['carrier_energy']*100:>4.1f}%")

    # Detect when carrier enters null space (max_cos drops below random)
    log(f"\n  NULL SPACE ENTRY:")
    for i, r in enumerate(all_results):
        if i > 0 and r["max_cos"] < r["rand_max_cos_mean"]:
            prev = all_results[i-1]
            if prev["max_cos"] >= prev["rand_max_cos_mean"]:
                log(f"    Carrier enters null space between step {prev['step']} and {r['step']}")
                log(f"    max_cos: {prev['max_cos']:.4f} → {r['max_cos']:.4f}")
                break

    elapsed = time.time() - t0
    log(f"\nTotal: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log("Done.")


if __name__ == "__main__":
    main()
