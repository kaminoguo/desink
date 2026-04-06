#!/usr/bin/env python3
"""
Exp 106: Non-Pythia Alignment Dynamics (OLMo-2 1B)
====================================================
Critical test: is the step 512 alignment peak + departure Pythia-specific?

OLMo-2 1B: different architecture (Llama-like), different data (Dolma), different training.
Checkpoints every 1000 steps from step 0 to step 37000.

If alignment peak + departure exists in OLMo → UNIVERSAL phenomenon
If not → Pythia-specific (data ordering / architecture artifact)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time, os, gc, json, shutil

DEVICE = "cuda"
# os.environ["HF_HOME"] = ...  # set if needed
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "exp106_output.log")

MODEL_NAME = "allenai/OLMo-2-0425-1B-early-training"

# Dense early steps + sparse late steps (actual revision names from HF)
STEPS = [
    ("stage1-step0-tokens0B", 0),
    ("stage1-step1000-tokens3B", 1000),
    ("stage1-step2000-tokens5B", 2000),
    ("stage1-step3000-tokens7B", 3000),
    ("stage1-step4000-tokens9B", 4000),
    ("stage1-step5000-tokens11B", 5000),
    ("stage1-step6000-tokens13B", 6000),
    ("stage1-step8000-tokens17B", 8000),
    ("stage1-step10000-tokens21B", 10000),
    ("stage1-step15000-tokens32B", 15000),
    ("stage1-step20000-tokens42B", 20000),
    ("stage1-step30000-tokens63B", 30000),
    ("stage1-step36000-tokens76B", 36000),
]

TEXTS = [
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
    """Try common layer paths."""
    for chain in ['model.layers', 'gpt_neox.layers', 'transformer.h',
                  'model.transformer.layers']:
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
    raise ValueError(f"Cannot find layers. Model type: {type(model)}")


def get_lm_head(model):
    """Try common lm_head paths."""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight.data.float()
    if hasattr(model, 'embed_out'):
        return model.embed_out.weight.data.float()
    # Check if lm_head is tied to embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.data.float()
    raise ValueError(f"Cannot find lm_head. Model type: {type(model)}")


def collect_hidden_states(model, tokenizer, texts, device, layer_idx):
    layers = get_layers(model)
    per_sent = []

    def hook_fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        per_sent.append(h.detach().float().cpu().squeeze(0))

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**ids)
    handle.remove()

    nosink = [h[1:] for h in per_sent if h.shape[0] > 1]
    H = torch.cat(nosink, dim=0)
    H_c = H - H.mean(0, keepdim=True)
    return H_c


def measure_alignment(H_c, lm_head_weight):
    """Measure fraction of H_c variance captured by lm_head top-100 SVs."""
    _, S_lm, Vt_lm = torch.linalg.svd(lm_head_weight, full_matrices=False)
    total_var = (H_c ** 2).sum().item()

    n_svs = min(100, len(S_lm))
    lm_vars = []
    for k in range(n_svs):
        proj = H_c @ Vt_lm[k]
        lm_vars.append((proj ** 2).sum().item() / total_var)

    return {
        "total_100": sum(lm_vars),
        "band_1_10": sum(lm_vars[:10]),
        "lm_sv1": S_lm[0].item(),
    }


def measure_nosink_er(H_c):
    """Nosink effective rank."""
    S = torch.linalg.svdvals(H_c)
    S = S / S.sum()
    S = S[S > 1e-10]
    er = torch.exp(-torch.sum(S * torch.log(S))).item()
    return er


def measure_E1(H_c):
    """Top-1 PCA energy."""
    S = torch.linalg.svdvals(H_c)
    return (S[0] ** 2 / (S ** 2).sum()).item()


def main():
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 70)
    log("EXP 106: NON-PYTHIA ALIGNMENT DYNAMICS (OLMo-2 1B)")
    log("=" * 70)
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Model: {MODEL_NAME}")
    log(f"Steps: {[s[1] for s in STEPS]}")
    t0 = time.time()

    # First load to discover architecture
    log("\n--- Discovering architecture ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=STEPS[0][0],
        torch_dtype=torch.float32, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = layers[0].self_attn.q_proj.weight.shape[0] if hasattr(layers[0], 'self_attn') else None
    log(f"  Layers: {n_layers}")
    log(f"  Hidden dim: {hidden_dim}")

    # Choose mid layer and late layer
    mid_layer = n_layers // 2
    late_layer = n_layers - 1
    log(f"  Mid layer: L{mid_layer}")
    log(f"  Late layer: L{late_layer}")

    lm_head_w = get_lm_head(model)
    log(f"  lm_head shape: {lm_head_w.shape}")
    del model, lm_head_w
    free_all()
    clean_hf_cache()

    # Main loop: measure at each step
    log(f"\n{'=' * 70}")
    log(f"ALIGNMENT DYNAMICS ACROSS TRAINING")
    log(f"{'=' * 70}")
    header = f"  {'Step':>6s}  {'NsER_mid':>8s}  {'E1_mid':>7s}  {'Lm100_mid':>9s}  {'Lm100_late':>10s}  {'LmSV1':>7s}"
    log(f"\n{header}")
    log(f"  {'-' * 65}")

    results = []
    for revision, step in STEPS:
        log(f"  Loading step {step}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, revision=revision,
                torch_dtype=torch.float32, trust_remote_code=True
            ).to(DEVICE)
            model.eval()
        except Exception as e:
            log(f"    FAILED: {e}")
            clean_hf_cache()
            continue

        # Collect hidden states at mid and late layers
        H_mid = collect_hidden_states(model, tokenizer, TEXTS, DEVICE, mid_layer)
        H_late = collect_hidden_states(model, tokenizer, TEXTS, DEVICE, late_layer)
        lm_head_w = get_lm_head(model).cpu()

        # Measure
        align_mid = measure_alignment(H_mid, lm_head_w)
        align_late = measure_alignment(H_late, lm_head_w)
        ns_er = measure_nosink_er(H_mid)
        e1 = measure_E1(H_mid)

        r = {
            "step": step,
            "revision": revision,
            "nosink_er_mid": ns_er,
            "E1_mid": e1,
            "lm100_mid": align_mid["total_100"],
            "lm100_late": align_late["total_100"],
            "lm_sv1": align_mid["lm_sv1"],
            "band_1_10_mid": align_mid["band_1_10"],
        }
        results.append(r)

        line = f"  {step:>6d}  {ns_er:>8.1f}  {e1*100:>6.1f}%  {align_mid['total_100']*100:>8.1f}%  {align_late['total_100']*100:>9.1f}%  {align_mid['lm_sv1']:>7.1f}"
        log(line)

        del model, H_mid, H_late, lm_head_w
        free_all()
        clean_hf_cache()

    # Analysis
    log(f"\n{'=' * 70}")
    log(f"ANALYSIS")
    log(f"{'=' * 70}")

    if len(results) >= 3:
        # Find peak alignment
        lm100_mid_vals = [(r["step"], r["lm100_mid"]) for r in results]
        peak_step, peak_val = max(lm100_mid_vals, key=lambda x: x[1])
        log(f"\n  Peak mid-layer alignment: step {peak_step} ({peak_val*100:.1f}%)")

        # Compare to Pythia
        log(f"\n  COMPARISON WITH PYTHIA:")
        log(f"    Pythia-410M: peak at step 512 (39.7%), L12")
        log(f"    OLMo-2 1B:  peak at step {peak_step} ({peak_val*100:.1f}%), L{mid_layer}")

        # Check if there's departure after peak
        peak_idx = next(i for i, r in enumerate(results) if r["step"] == peak_step)
        if peak_idx < len(results) - 1:
            final_val = results[-1]["lm100_mid"]
            log(f"\n  Departure check:")
            log(f"    Peak: {peak_val*100:.1f}%  Final: {final_val*100:.1f}%")
            if final_val < peak_val * 0.7:
                log(f"    >>> DEPARTURE EXISTS: {final_val/peak_val*100:.0f}% retained")
                log(f"    >>> Alignment peak + departure is NOT Pythia-specific!")
            else:
                log(f"    >>> No clear departure: {final_val/peak_val*100:.0f}% retained")

        # ER dynamics
        log(f"\n  ER dynamics around peak:")
        for r in results:
            if abs(r["step"] - peak_step) <= 5000 or r["step"] == 0 or r == results[-1]:
                log(f"    step {r['step']:>6d}: nsER={r['nosink_er_mid']:.1f}  lm100={r['lm100_mid']*100:.1f}%  E1={r['E1_mid']*100:.1f}%")

        # lm_head SV1 evolution
        log(f"\n  lm_head SV1 evolution:")
        sv1_line = "    "
        for r in results:
            sv1_line += f"  {r['step']}:{r['lm_sv1']:.1f}"
        log(sv1_line)

    # Save
    json_path = os.path.join(LOG_DIR, "exp106_data.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    log(f"\nTotal: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    log("Done.")


if __name__ == "__main__":
    main()
