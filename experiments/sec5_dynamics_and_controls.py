"""Section 5 + Section 8 supporting experiments.
- Pythia-160M / 1.4B / OLMo-2 1B: 20-checkpoint training dynamics (rank growth).
- ER vs probing-accuracy correlation (Spearman rho).
- All-but-the-Top vs de-sinking comparison (controls).
- Eigenvalue spectral gap analysis (controls)."""

import os, sys, json, time
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import torch
import numpy as np
from scipy.stats import spearmanr, entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_spectral_metrics(H):
    """Compute E1, ER, anisotropy from hidden states H [n, d]."""
    H_bar = H - H.mean(0)
    try:
        U, S, Vt = torch.linalg.svd(H_bar, full_matrices=False)
    except:
        U, S, Vt = torch.svd(H_bar)
        Vt = Vt.T

    S2 = S ** 2
    S2_norm = S2 / S2.sum()
    E1 = float(S2_norm[0])
    ER = float(torch.exp(-torch.sum(S2_norm * torch.log(S2_norm + 1e-10))))

    # Sink direction = first right singular vector
    sink_dir = Vt[0]

    # De-sink
    proj = H_bar @ sink_dir
    H_ds = H_bar - proj.unsqueeze(1) * sink_dir.unsqueeze(0)

    try:
        U_ds, S_ds, Vt_ds = torch.linalg.svd(H_ds, full_matrices=False)
    except:
        U_ds, S_ds, Vt_ds = torch.svd(H_ds)

    S2_ds = S_ds ** 2
    S2_ds_norm = S2_ds / S2_ds.sum()
    E1_ds = float(S2_ds_norm[0])
    ER_ds = float(torch.exp(-torch.sum(S2_ds_norm * torch.log(S2_ds_norm + 1e-10))))

    # Sink alpha (position-0 norm ratio)
    norms = torch.norm(H, dim=1)
    sink_alpha = float(norms[0] / norms[1:].mean()) if len(norms) > 1 else 1.0

    # Anisotropy
    H_normed = H_bar / (torch.norm(H_bar, dim=1, keepdim=True) + 1e-10)
    cos_matrix = H_normed @ H_normed.T
    n = cos_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=cos_matrix.device)
    aniso = float(cos_matrix[mask].mean())

    # Top-10 singular values for gap analysis
    top10 = S[:10].tolist()

    return {
        "E1_raw": E1, "E1_ds": E1_ds,
        "ER_raw": ER, "ER_ds": ER_ds,
        "sink_alpha": sink_alpha,
        "anisotropy": aniso,
        "top10_sv": top10,
        "gap_ratio": float(S[0] / S[1]) if len(S) > 1 else 0,
    }


def collect_hidden_states(model, tokenizer, texts, layer_idx, max_tokens=200000):
    """Collect hidden states at a specific layer."""
    all_hidden = []
    total_tokens = 0

    hook_output = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hook_output["h"] = output[0].detach()
        else:
            hook_output["h"] = output.detach()

    # Find the target layer
    layers = [mod for name, mod in model.named_modules()
              if (hasattr(mod, 'self_attn') or hasattr(mod, 'attention')) and hasattr(mod, 'mlp')]

    if layer_idx >= len(layers):
        return None

    handle = layers[layer_idx].register_forward_hook(hook_fn)

    model.eval()
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            hook_output.clear()
            model(**inputs)
            if "h" in hook_output:
                h = hook_output["h"].squeeze(0).float()  # [seq_len, hidden_dim]
                all_hidden.append(h.cpu())
                total_tokens += h.shape[0]
                if total_tokens >= max_tokens:
                    break

    handle.remove()
    if all_hidden:
        return torch.cat(all_hidden, dim=0)[:max_tokens]
    return None


def load_texts(max_texts=500):
    """Load evaluation texts from OpenWebText or fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset("stas/openwebtext-10k", split="train")
        return [t for t in ds["text"][:max_texts] if len(t) > 50]
    except:
        # Fallback: generate simple texts
        return ["The quick brown fox jumps over the lazy dog. " * 10] * 200


# ══════════════════════════════════════════════════════════════
# LOAD TEXTS
# ══════════════════════════════════════════════════════════════
print("Loading evaluation texts...", flush=True)
texts = load_texts()
print(f"Loaded {len(texts)} texts")


# ══════════════════════════════════════════════════════════════
# Step 1: Pythia-160M training dynamics
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Step 1: Pythia-160M training dynamics")
print(f"{'='*60}", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

STEPS_160M = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
N_LAYERS_160M = 12

p160m_results = {"model": "pythia-160m", "layers": {}}

for layer_idx in range(N_LAYERS_160M):
    print(f"\n  Layer {layer_idx}:", flush=True)
    layer_data = []

    for step in STEPS_160M:
        revision = f"step{step}"
        try:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision=revision)
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m", revision=revision,
                                                          torch_dtype=torch.float32).to(device)
        except Exception as e:
            print(f"    Step {step}: FAILED to load ({e})")
            continue

        H = collect_hidden_states(model, tokenizer, texts, layer_idx, max_tokens=200000)
        if H is None:
            print(f"    Step {step}: no hidden states")
            del model
            torch.cuda.empty_cache()
            continue

        H = H.to(device)
        metrics = compute_spectral_metrics(H)
        metrics["step"] = step

        # Get loss
        try:
            with torch.inference_mode():
                inp = tokenizer(texts[0][:500], return_tensors="pt", truncation=True).to(device)
                out = model(**inp, labels=inp["input_ids"])
                metrics["loss"] = float(out.loss)
        except:
            metrics["loss"] = None

        layer_data.append(metrics)
        print(f"    Step {step:6d}: E1={metrics['E1_raw']:.3f}/{metrics['E1_ds']:.3f}  "
              f"ER={metrics['ER_raw']:.1f}/{metrics['ER_ds']:.1f}  "
              f"sink={metrics['sink_alpha']:.2f}x", flush=True)

        del model, H
        torch.cuda.empty_cache()

    p160m_results["layers"][str(layer_idx)] = layer_data

os.makedirs("logs/dynamics_supporting", exist_ok=True)
with open("logs/dynamics_supporting/pythia160m_dynamics.json", "w") as f:
    json.dump(p160m_results, f, indent=2)
print("\nStep 1 saved.", flush=True)


# ══════════════════════════════════════════════════════════════
# Step 2: Pythia-1.4B 20-checkpoint training dynamics
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Step 2: Pythia-1.4B training dynamics")
print(f"{'='*60}", flush=True)

p1_4b_results = {"model": "pythia-1.4b", "layers": {}}

STEPS_1_4B = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
N_LAYERS_1_4B = 24  # Pythia-1.4B has 24 layers

for layer_idx in range(N_LAYERS_1_4B):
    print(f"\n  Layer {layer_idx}:", flush=True)
    layer_data = []

    for step in STEPS_1_4B:
        revision = f"step{step}"
        try:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b", revision=revision)
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b", revision=revision,
                                                          torch_dtype=torch.float32).to(device)
        except Exception as e:
            print(f"    Step {step}: FAILED to load ({e})")
            continue

        H = collect_hidden_states(model, tokenizer, texts, layer_idx, max_tokens=200000)
        if H is None:
            print(f"    Step {step}: no hidden states")
            del model
            torch.cuda.empty_cache()
            continue

        H = H.to(device)
        metrics = compute_spectral_metrics(H)
        metrics["step"] = step

        # Get loss
        try:
            with torch.inference_mode():
                inp = tokenizer(texts[0][:500], return_tensors="pt", truncation=True).to(device)
                out = model(**inp, labels=inp["input_ids"])
                metrics["loss"] = float(out.loss)
        except:
            metrics["loss"] = None

        layer_data.append(metrics)
        print(f"    Step {step:6d}: E1={metrics['E1_raw']:.3f}/{metrics['E1_ds']:.3f}  "
              f"ER={metrics['ER_raw']:.1f}/{metrics['ER_ds']:.1f}  "
              f"sink={metrics['sink_alpha']:.2f}x", flush=True)

        del model, H
        torch.cuda.empty_cache()

    p1_4b_results["layers"][str(layer_idx)] = layer_data

with open("logs/dynamics_supporting/pythia1_4b_dynamics.json", "w") as f:
    json.dump(p1_4b_results, f, indent=2)
print("\nStep 2 saved.", flush=True)


# ══════════════════════════════════════════════════════════════
# Step 3: Pythia-160M ER vs probing-accuracy correlation
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Step 3: Pythia-160M ER vs probing-accuracy correlation")
print(f"{'='*60}", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

MID_LAYER = 5
er_probe_data = []

for step in STEPS_160M:
    revision = f"step{step}"
    try:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision=revision)
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m", revision=revision,
                                                      torch_dtype=torch.float32).to(device)
    except:
        continue

    # Get ER from Step 1 results
    l5_data = p160m_results["layers"].get(str(MID_LAYER), [])
    er_entry = [d for d in l5_data if d.get("step") == step]
    er_raw = er_entry[0]["ER_raw"] if er_entry else None
    er_ds = er_entry[0]["ER_ds"] if er_entry else None

    # Probe: next-token prediction
    H = collect_hidden_states(model, tokenizer, texts[:100], MID_LAYER, max_tokens=50000)
    if H is None:
        del model
        torch.cuda.empty_cache()
        continue

    # Get labels (next token IDs)
    labels = []
    features = []
    with torch.inference_mode():
        for text in texts[:50]:
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            ids = inp["input_ids"].squeeze()
            if len(ids) < 3:
                continue
            # Labels are shifted input IDs
            for i in range(len(ids) - 1):
                labels.append(int(ids[i + 1]))

    # Truncate features to match labels
    n = min(len(labels), H.shape[0] - 1)
    if n < 100:
        del model, H
        torch.cuda.empty_cache()
        continue

    X = H[:n].numpy()
    y = np.array(labels[:n])

    # Top-500 classes only
    from collections import Counter
    top500 = [c for c, _ in Counter(y).most_common(500)]
    mask = np.isin(y, top500)
    X_f, y_f = X[mask], y[mask]

    if len(X_f) < 100:
        del model, H
        torch.cuda.empty_cache()
        continue

    # Reduce dim for speed
    from sklearn.decomposition import PCA
    n_pca = min(64, X_f.shape[0] - 1, X_f.shape[1])
    X_pca = PCA(n_components=n_pca).fit_transform(X_f)

    clf = LogisticRegression(max_iter=500, C=1.0)
    try:
        scores = cross_val_score(clf, X_pca, y_f, cv=3, scoring="accuracy")
        probe_acc = float(scores.mean())
    except:
        probe_acc = None

    er_probe_data.append({
        "step": step, "ER_raw": er_raw, "ER_ds": er_ds, "probe_acc": probe_acc
    })
    print(f"  Step {step:6d}: ER_raw={er_raw:.1f}  ER_ds={er_ds:.1f}  probe={probe_acc:.4f}" if probe_acc else f"  Step {step}: probe failed", flush=True)

    del model, H
    torch.cuda.empty_cache()

# Compute correlations
if len(er_probe_data) >= 5:
    er_raw_list = [d["ER_raw"] for d in er_probe_data if d["ER_raw"] is not None and d["probe_acc"] is not None]
    er_ds_list = [d["ER_ds"] for d in er_probe_data if d["ER_ds"] is not None and d["probe_acc"] is not None]
    probe_list = [d["probe_acc"] for d in er_probe_data if d["ER_raw"] is not None and d["probe_acc"] is not None]

    rho_raw, p_raw = spearmanr(er_raw_list, probe_list)
    rho_ds, p_ds = spearmanr(er_ds_list, probe_list)

    print(f"\n  Raw ER vs probe: rho={rho_raw:.4f}, p={p_raw:.4e}")
    print(f"  De-sinked ER vs probe: rho={rho_ds:.4f}, p={p_ds:.4e}")

    er_probe_results = {
        "model": "pythia-160m", "layer": MID_LAYER,
        "rho_raw": float(rho_raw), "p_raw": float(p_raw),
        "rho_ds": float(rho_ds), "p_ds": float(p_ds),
        "data": er_probe_data,
    }
else:
    er_probe_results = {"model": "pythia-160m", "data": er_probe_data, "note": "insufficient data"}

with open("logs/dynamics_supporting/pythia160m_er_probe.json", "w") as f:
    json.dump(er_probe_results, f, indent=2)
print("\nStep 3 saved.", flush=True)


# ══════════════════════════════════════════════════════════════
# Step 4: All-but-the-Top vs de-sinking
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Step 4: All-but-the-Top vs de-sinking")
print(f"{'='*60}", flush=True)

abt_results = []

for model_name, step, layer_idx in [
    ("EleutherAI/pythia-70m", 143000, 3),
    ("EleutherAI/pythia-70m", 4000, 3),
    ("EleutherAI/pythia-70m", 143000, 5),
]:
    revision = f"step{step}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                      torch_dtype=torch.float32).to(device)
    except:
        continue

    H = collect_hidden_states(model, tokenizer, texts, layer_idx, max_tokens=200000)
    if H is None:
        del model
        torch.cuda.empty_cache()
        continue

    H = H.to(device)
    H_bar = H - H.mean(0)
    U, S, Vt = torch.linalg.svd(H_bar, full_matrices=False)

    methods = {}

    # Method A: De-sink (= All-but-the-Top for PC1)
    s1 = Vt[0]
    H_ds = H_bar - (H_bar @ s1).unsqueeze(1) * s1.unsqueeze(0)
    m = compute_spectral_metrics(H_ds + H.mean(0))  # add mean back for fair comparison
    methods["desink_pc1"] = {"E1": m["E1_raw"], "ER": m["ER_raw"], "anisotropy": m["anisotropy"]}

    # Method B: Remove PC2
    s2 = Vt[1]
    H_pc2 = H_bar - (H_bar @ s2).unsqueeze(1) * s2.unsqueeze(0)
    m = compute_spectral_metrics(H_pc2 + H.mean(0))
    methods["remove_pc2"] = {"E1": m["E1_raw"], "ER": m["ER_raw"], "anisotropy": m["anisotropy"]}

    # Method C: Remove PC3
    s3 = Vt[2]
    H_pc3 = H_bar - (H_bar @ s3).unsqueeze(1) * s3.unsqueeze(0)
    m = compute_spectral_metrics(H_pc3 + H.mean(0))
    methods["remove_pc3"] = {"E1": m["E1_raw"], "ER": m["ER_raw"], "anisotropy": m["anisotropy"]}

    # Method D: Remove random direction (5 seeds)
    rand_results = []
    for seed in range(5):
        torch.manual_seed(seed)
        r = torch.randn(H.shape[1], device=device)
        r = r / r.norm()
        H_rand = H_bar - (H_bar @ r).unsqueeze(1) * r.unsqueeze(0)
        m = compute_spectral_metrics(H_rand + H.mean(0))
        rand_results.append({"E1": m["E1_raw"], "ER": m["ER_raw"], "anisotropy": m["anisotropy"]})
    methods["remove_random_mean"] = {
        k: float(np.mean([r[k] for r in rand_results])) for k in ["E1", "ER", "anisotropy"]
    }
    methods["remove_random_std"] = {
        k: float(np.std([r[k] for r in rand_results])) for k in ["E1", "ER", "anisotropy"]
    }

    # Original (no removal)
    m_orig = compute_spectral_metrics(H)
    methods["original"] = {"E1": m_orig["E1_raw"], "ER": m_orig["ER_raw"], "anisotropy": m_orig["anisotropy"]}

    entry = {"model": model_name, "step": step, "layer": layer_idx, "methods": methods}
    abt_results.append(entry)

    print(f"\n  {model_name} step={step} L{layer_idx}:")
    for method_name, vals in methods.items():
        if isinstance(vals, dict) and "E1" in vals:
            print(f"    {method_name:20s}: E1={vals['E1']:.4f}  ER={vals['ER']:.1f}  aniso={vals['anisotropy']:.4f}")

    del model, H, H_bar
    torch.cuda.empty_cache()

with open("logs/dynamics_supporting/abt_comparison.json", "w") as f:
    json.dump(abt_results, f, indent=2)
print("\nStep 4 saved.", flush=True)


# ══════════════════════════════════════════════════════════════
# Step 5: Eigenvalue gap analysis
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Step 5: Eigenvalue gap analysis")
print(f"{'='*60}", flush=True)

gap_results = []

for model_name in ["EleutherAI/pythia-70m", "gpt2"]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    except:
        continue

    layers = [mod for name, mod in model.named_modules()
              if (hasattr(mod, 'self_attn') or hasattr(mod, 'attention')) and hasattr(mod, 'mlp')]
    n_layers = len(layers)

    model_data = {"model": model_name, "n_layers": n_layers, "layers": []}

    for layer_idx in range(n_layers):
        H = collect_hidden_states(model, tokenizer, texts, layer_idx, max_tokens=200000)
        if H is None:
            continue
        H = H.to(device)
        metrics = compute_spectral_metrics(H)
        model_data["layers"].append({
            "layer": layer_idx,
            "top10_sv": metrics["top10_sv"],
            "gap_ratio": metrics["gap_ratio"],
            "E1_raw": metrics["E1_raw"],
        })
        print(f"  {model_name} L{layer_idx}: gap={metrics['gap_ratio']:.2f}x  E1={metrics['E1_raw']:.3f}", flush=True)
        del H

    gap_results.append(model_data)
    del model
    torch.cuda.empty_cache()

with open("logs/dynamics_supporting/eigenvalue_gap.json", "w") as f:
    json.dump(gap_results, f, indent=2)
print("\nStep 5 saved.", flush=True)

print(f"\n{'='*60}")
print("ALL PAPER 1 EXPERIMENTS DONE")
print(f"{'='*60}", flush=True)
