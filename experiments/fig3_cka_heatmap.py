"""
Paper 1 Exp C: De-Sinked CKA Heatmap
=======================================
Compute layer×layer CKA matrix with and without de-sinking.
A5 showed aggregate CKA barely changes (0.640→0.644 for GPT-2).
But the heatmap might show local changes in specific layer pairs.

Models: GPT-2 and Pythia-70M.
"""
import torch
import numpy as np
import json
import os
import time


def linear_cka(X, Y):
    """Compute Linear CKA between X (n, p) and Y (n, q)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def desink(activations):
    """Remove PC1 direction (sink proxy)."""
    X = activations - activations.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    sink_dir = Vt[0]
    sink_dir = sink_dir / (np.linalg.norm(sink_dir) + 1e-10)
    proj = activations @ sink_dir
    return activations - np.outer(proj, sink_dir)


def extract_hidden_states(model, tokenizer, texts, device, max_len=128):
    """Extract hidden states at all layers."""
    all_hidden = {}  # layer -> list of arrays
    model.eval()

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        for li in range(len(out.hidden_states)):
            h = out.hidden_states[li][0].float().cpu().numpy()
            if li not in all_hidden:
                all_hidden[li] = []
            all_hidden[li].append(h)

    # Concatenate
    for li in all_hidden:
        all_hidden[li] = np.concatenate(all_hidden[li], axis=0)

    return all_hidden


def compute_cka_matrix(hidden_states, desink_flag=False):
    """Compute layer×layer CKA matrix."""
    layers = sorted(hidden_states.keys())
    n = len(layers)
    matrix = np.zeros((n, n))

    acts = {}
    for li in layers:
        a = hidden_states[li]
        if desink_flag:
            a = desink(a)
        acts[li] = a

    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            if j >= i:
                cka = linear_cka(acts[li], acts[lj])
                matrix[i, j] = cka
                matrix[j, i] = cka

    return matrix, layers


def run_model(model_name, device, n_texts=200):
    """Run CKA analysis on one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device
    )

    # Load texts
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:n_texts]
    except Exception:
        texts = [f"The quick brown fox jumps. Sentence {i}." for i in range(n_texts)]
    print(f"Texts: {len(texts)}")

    # Extract hidden states
    print("Extracting hidden states...")
    hidden = extract_hidden_states(model, tokenizer, texts, device)
    print(f"Layers: {len(hidden)}, tokens per layer: {hidden[0].shape[0]}")

    # Raw CKA
    print("Computing raw CKA matrix...")
    raw_matrix, layers = compute_cka_matrix(hidden, desink_flag=False)

    # De-sinked CKA
    print("Computing de-sinked CKA matrix...")
    ds_matrix, _ = compute_cka_matrix(hidden, desink_flag=True)

    # Diff
    diff_matrix = ds_matrix - raw_matrix

    # Print summary
    print(f"\nRaw CKA (min={raw_matrix.min():.3f}, max={raw_matrix.max():.3f}, "
          f"mean off-diag={raw_matrix[~np.eye(len(layers), dtype=bool)].mean():.3f})")
    print(f"De-sinked CKA (min={ds_matrix.min():.3f}, max={ds_matrix.max():.3f}, "
          f"mean off-diag={ds_matrix[~np.eye(len(layers), dtype=bool)].mean():.3f})")
    print(f"Diff (min={diff_matrix.min():.3f}, max={diff_matrix.max():.3f}, "
          f"mean abs={np.abs(diff_matrix[~np.eye(len(layers), dtype=bool)]).mean():.3f})")

    # Print matrices
    print(f"\nRaw CKA matrix:")
    header = "     " + "".join([f"L{l:<5}" for l in layers])
    print(header)
    for i, li in enumerate(layers):
        row = f"L{li:<4}" + "".join([f"{raw_matrix[i,j]:>5.2f} " for j in range(len(layers))])
        print(row)

    print(f"\nDiff matrix (de-sinked - raw):")
    print(header)
    for i, li in enumerate(layers):
        row = f"L{li:<4}" + "".join([f"{diff_matrix[i,j]:>+5.3f}" for j in range(len(layers))])
        print(row)

    # Find biggest changes
    triu = np.triu_indices(len(layers), k=1)
    diffs = [(layers[triu[0][k]], layers[triu[1][k]], diff_matrix[triu[0][k], triu[1][k]])
             for k in range(len(triu[0]))]
    diffs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\nTop 5 biggest CKA changes:")
    for l1, l2, d in diffs[:5]:
        print(f"  L{l1}-L{l2}: {d:+.4f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "layers": [int(l) for l in layers],
        "raw_cka": raw_matrix.tolist(),
        "desinked_cka": ds_matrix.tolist(),
        "diff_cka": diff_matrix.tolist(),
        "raw_mean_offdiag": float(raw_matrix[~np.eye(len(layers), dtype=bool)].mean()),
        "ds_mean_offdiag": float(ds_matrix[~np.eye(len(layers), dtype=bool)].mean()),
        "max_abs_diff": float(np.abs(diff_matrix).max()),
        "mean_abs_diff": float(np.abs(diff_matrix[~np.eye(len(layers), dtype=bool)]).mean()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    t0 = time.time()
    results = []

    for model_name in ["gpt2", "EleutherAI/pythia-70m"]:
        r = run_model(model_name, device)
        results.append(r)

    os.makedirs("logs/desink_cka", exist_ok=True)
    with open("logs/desink_cka/desink_cka_heatmap.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll done. Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
