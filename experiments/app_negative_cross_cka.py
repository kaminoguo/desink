"""Appendix C: Cross-model CKA (GPT-2 vs Pythia-70M) raw vs de-sinked."""

import torch
import numpy as np
import json
import os
import time


def desink(X):
    """GPU-accelerated desink."""
    if isinstance(X, np.ndarray): X = torch.from_numpy(X).cuda()
    X_c = X - X.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    d = Vt[0]; d = d / (d.norm() + 1e-10)
    return (X - (X @ d).unsqueeze(-1) * d.unsqueeze(0)).cpu().numpy()


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-10 else 0.0


def extract_hidden(model_name, texts, tokenizer_name=None, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device)
    model.eval()
    n_layers = model.config.num_hidden_layers

    hidden = {}
    for idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0].float().cpu().numpy()
            if li not in hidden:
                hidden[li] = []
            hidden[li].append(h)

    # Align by using same number of tokens per text (min across texts)
    # For cross-model CKA, we need same N (number of observations)
    min_tokens = min(len(hidden[0][i]) for i in range(len(texts)))
    for li in hidden:
        hidden[li] = np.concatenate([h[:min_tokens] for h in hidden[li]], axis=0)

    del model
    torch.cuda.empty_cache()
    return hidden, n_layers


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:100]
    print(f"Texts: {len(texts)}")

    # Extract from both models using SAME texts
    print("\nExtracting GPT-2...")
    h_gpt2, n_gpt2 = extract_hidden("gpt2", texts, device=device)
    print(f"  GPT-2: {n_gpt2} layers, {h_gpt2[0].shape[0]} tokens, dim={h_gpt2[0].shape[1]}")

    print("\nExtracting Pythia-70M...")
    h_pythia, n_pythia = extract_hidden("EleutherAI/pythia-70m", texts, device=device)
    print(f"  Pythia: {n_pythia} layers, {h_pythia[0].shape[0]} tokens, dim={h_pythia[0].shape[1]}")

    # Align token counts
    min_n = min(h_gpt2[0].shape[0], h_pythia[0].shape[0])
    for li in h_gpt2:
        h_gpt2[li] = h_gpt2[li][:min_n]
    for li in h_pythia:
        h_pythia[li] = h_pythia[li][:min_n]
    print(f"Aligned to {min_n} tokens")

    # Compute cross-model CKA matrix
    print("\nComputing cross-model CKA (raw)...")
    cka_raw = np.zeros((n_gpt2 + 1, n_pythia + 1))
    for i in range(n_gpt2 + 1):
        for j in range(n_pythia + 1):
            cka_raw[i, j] = linear_cka(h_gpt2[i], h_pythia[j])

    print("Computing cross-model CKA (de-sinked)...")
    cka_ds = np.zeros((n_gpt2 + 1, n_pythia + 1))
    for i in range(n_gpt2 + 1):
        for j in range(n_pythia + 1):
            cka_ds[i, j] = linear_cka(desink(h_gpt2[i]), desink(h_pythia[j]))

    diff = cka_ds - cka_raw

    # Results
    print(f"\n{'='*60}")
    print("CROSS-MODEL CKA: GPT-2 vs Pythia-70M")
    print(f"{'='*60}")

    print(f"\nRaw CKA: mean={cka_raw.mean():.4f}, max={cka_raw.max():.4f}")
    print(f"De-sinked CKA: mean={cka_ds.mean():.4f}, max={cka_ds.max():.4f}")
    print(f"Diff: mean={diff.mean():.4f}, max abs={np.abs(diff).max():.4f}")

    # Diagonal-ish comparison (matched relative positions)
    print(f"\nMatched-position CKA (relative depth):")
    print(f"{'GPT-2':>8} {'Pythia':>8} {'raw':>8} {'ds':>8} {'diff':>8}")
    for rel in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gi = int(rel * n_gpt2)
        pi = int(rel * n_pythia)
        print(f"L{gi:>7} L{pi:>7} {cka_raw[gi,pi]:>7.4f} {cka_ds[gi,pi]:>7.4f} {diff[gi,pi]:>+7.4f}")

    # Key finding
    raw_max = cka_raw.max()
    ds_max = cka_ds.max()
    drop = raw_max - ds_max
    drop_pct = drop / raw_max * 100

    print(f"\nMax cross-model CKA: raw={raw_max:.4f}, de-sinked={ds_max:.4f}")
    print(f"Drop: {drop:.4f} ({drop_pct:.1f}%)")

    if drop_pct > 20:
        print(f"\n→ SIGNIFICANT DROP: {drop_pct:.0f}% of cross-model similarity is shared sink structure")
        print("  'Models converging to similar representations' is partially a sink artifact")
    elif drop_pct > 5:
        print(f"\n→ MODERATE DROP: {drop_pct:.0f}% of similarity is sink-driven")
    else:
        print(f"\n→ MINIMAL DROP: Cross-model similarity is genuine, not sink-driven")

    os.makedirs("logs/exp_H_cross_model", exist_ok=True)
    out = {
        "models": ["gpt2", "EleutherAI/pythia-70m"],
        "n_tokens": min_n,
        "raw_cka_mean": float(cka_raw.mean()),
        "ds_cka_mean": float(cka_ds.mean()),
        "raw_cka_max": float(raw_max),
        "ds_cka_max": float(ds_max),
        "drop_pct": float(drop_pct),
        "raw_cka_matrix": cka_raw.tolist(),
        "ds_cka_matrix": cka_ds.tolist(),
    }
    with open("logs/exp_H_cross_model/cross_model_cka.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
