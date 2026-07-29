"""Figure 1(a,b): Phantom phases in transformer training.
Compute raw and de-sinked spectral metrics (E1, ER, anisotropy) across 19 Pythia-70M training checkpoints."""

import torch
import numpy as np
import json
import os
import time
from collections import defaultdict

CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000
]

TARGET_LAYERS = [1, 3, 6]  # Early, middle, final for Pythia-70M (6 layers)

N_TEXTS = 200


def compute_metrics(activations):
    """Compute E1, ER, anisotropy, RankMe from activations (N, d)."""
    X = activations - activations.mean(axis=0, keepdims=True)
    cov = X.T @ X / max(X.shape[0] - 1, 1)

    # SVD for E1 and ER
    vals = np.linalg.svdvals(cov)
    vals = np.maximum(vals, 0)
    total = vals.sum()
    e1 = vals[0] / total if total > 0 else 0

    # Effective rank (exponential of entropy of normalized singular values)
    p = vals / total if total > 0 else vals
    p = p[p > 1e-10]
    er = np.exp(-np.sum(p * np.log(p))) if len(p) > 0 else 1.0

    # RankMe (same as ER but on singular values, not squared)
    sv = np.sqrt(np.maximum(vals, 0))
    sv_total = sv.sum()
    if sv_total > 0:
        p_sv = sv / sv_total
        p_sv = p_sv[p_sv > 1e-10]
        rankme = np.exp(-np.sum(p_sv * np.log(p_sv)))
    else:
        rankme = 1.0

    # Anisotropy (average pairwise cosine)
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = activations / norms
    n = normed.shape[0]
    if n > 1:
        # Sample for efficiency
        sample_size = min(500, n)
        indices = np.random.choice(n, sample_size, replace=False)
        sampled = normed[indices]
        cos_matrix = sampled @ sampled.T
        # Exclude diagonal
        mask = ~np.eye(sample_size, dtype=bool)
        anisotropy = cos_matrix[mask].mean()
    else:
        anisotropy = 0.0

    return {
        "e1": float(e1),
        "er": float(er),
        "rankme": float(rankme),
        "anisotropy": float(anisotropy),
    }


def desink(activations):
    """Remove the position-0 dominated direction."""
    # Sink direction = mean of position-0 tokens across samples
    # In our setup, each sample has position-0 as the first token
    # But we're using pooled activations, so we need a different approach:
    # Use PC1 of the raw data as sink proxy (standard approach from our experiments)
    X = activations - activations.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    sink_dir = Vt[0]  # Top PC direction
    sink_dir = sink_dir / (np.linalg.norm(sink_dir) + 1e-10)

    # Project out sink direction
    proj = activations @ sink_dir
    desinked = activations - np.outer(proj, sink_dir)
    return desinked


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load texts
    print("\nLoading texts...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:N_TEXTS]
        print(f"  Loaded {len(texts)} texts from WikiText-2")
    except Exception:
        raise RuntimeError(
            "corpus load failed; refusing to fall back to synthetic text. "
            "Every number in the paper comes from natural text (see Table 6).")

    print(f"\nProcessing {len(CHECKPOINTS)} checkpoints × {len(TARGET_LAYERS)} layers...")

    all_results = []
    t0 = time.time()

    for ci, step in enumerate(CHECKPOINTS):
        print(f"\n[{ci+1}/{len(CHECKPOINTS)}] Step {step}")

        # Load model at this checkpoint
        model_name = f"EleutherAI/pythia-70m"
        revision = f"step{step}"

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision,
                torch_dtype=torch.float32, device_map=device
            )
            model.eval()
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        # Extract hidden states
        all_hidden = defaultdict(list)

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            for li in TARGET_LAYERS:
                if li + 1 < len(out.hidden_states):
                    h = out.hidden_states[li + 1][0].float().cpu().numpy()
                    # Use all token positions
                    all_hidden[li].append(h)

        # Compute metrics per layer
        checkpoint_results = {"step": step, "layers": {}}

        for li in TARGET_LAYERS:
            if li not in all_hidden or not all_hidden[li]:
                continue
            acts = np.concatenate(all_hidden[li], axis=0)  # (total_tokens, d)
            if acts.shape[0] < 10:
                continue

            # Raw metrics
            raw = compute_metrics(acts)

            # De-sinked metrics
            ds_acts = desink(acts)
            ds = compute_metrics(ds_acts)

            checkpoint_results["layers"][li] = {
                "raw": raw,
                "desinked": ds,
                "n_tokens": acts.shape[0],
            }

            print(f"  L{li}: E1 {raw['e1']:.3f}→{ds['e1']:.3f}  "
                  f"ER {raw['er']:.1f}→{ds['er']:.1f}  "
                  f"RankMe {raw['rankme']:.1f}→{ds['rankme']:.1f}  "
                  f"Aniso {raw['anisotropy']:.3f}→{ds['anisotropy']:.3f}")

        all_results.append(checkpoint_results)

        # Free memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print("DE-SINKED TRAINING TRAJECTORY SUMMARY")
    print(sep)

    for li in TARGET_LAYERS:
        print(f"\n--- Layer {li} ---")
        print(f"{'Step':<10} {'E1_raw':>8} {'E1_ds':>8} {'ER_raw':>8} {'ER_ds':>8} "
              f"{'RM_raw':>8} {'RM_ds':>8} {'Ani_raw':>8} {'Ani_ds':>8}")
        print("-" * 80)
        for r in all_results:
            if li in r["layers"]:
                raw = r["layers"][li]["raw"]
                ds = r["layers"][li]["desinked"]
                print(f"{r['step']:<10} {raw['e1']:>7.3f} {ds['e1']:>7.3f} "
                      f"{raw['er']:>7.1f} {ds['er']:>7.1f} "
                      f"{raw['rankme']:>7.1f} {ds['rankme']:>7.1f} "
                      f"{raw['anisotropy']:>7.3f} {ds['anisotropy']:>7.3f}")

    os.makedirs("logs/desink_trajectory", exist_ok=True)
    with open("logs/desink_trajectory/desink_trajectory.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved. Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
