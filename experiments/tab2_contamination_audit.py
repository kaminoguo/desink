"""Table 2: Contamination audit on five published metrics. Reproduces E1, rogue dimensions, RankMe, anisotropy, and CKA on GPT-2 and Pythia-70M (200K OpenWebText tokens), with raw vs de-sinked values."""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# De-sink utility
# ============================================================

def compute_sink_direction(hidden_states):
    """Compute the dominant sink direction from position-0 tokens.
    hidden_states: (n_seqs, seq_len, dim)
    """
    pos0 = hidden_states[:, 0, :]  # (n_seqs, dim)
    mean_dir = pos0.mean(axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm < 1e-10:
        return mean_dir
    return mean_dir / norm


def desink(tokens, sink_dir):
    """Remove sink direction from token representations.
    tokens: (n, dim), sink_dir: (dim,)
    """
    proj = (tokens @ sink_dir[:, None]) * sink_dir[None, :]
    return tokens - proj


# ============================================================
# Metric 1: Anisotropy (Ethayarajh 2019)
# ============================================================

def compute_anisotropy(tokens, n_pairs=5000):
    """Average pairwise cosine similarity among random token pairs."""
    n = len(tokens)
    if n < 2:
        return 0.0
    tokens_norm = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-10)
    # Random pairs
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    idx1 = np.random.randint(0, n, n_pairs)
    idx2 = np.random.randint(0, n, n_pairs)
    # Avoid self-pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    cos_sims = np.sum(tokens_norm[idx1] * tokens_norm[idx2], axis=1)
    return float(np.mean(cos_sims))


# ============================================================
# Metric 2: Rogue Dimensions (Timkey & van Schijndel 2021)
# ============================================================

def compute_rogue_dims(tokens, sink_dir=None, top_k=5):
    """Identify dimensions with disproportionate variance.
    Returns variance per dimension, top-k rogue dims, and their overlap with sink direction.
    """
    centered = tokens - tokens.mean(axis=0)
    var_per_dim = np.var(centered, axis=0)  # (dim,)
    total_var = var_per_dim.sum()
    sorted_idx = np.argsort(-var_per_dim)
    top_k_idx = sorted_idx[:top_k]
    top_k_var_frac = var_per_dim[top_k_idx].sum() / total_var * 100 if total_var > 0 else 0

    # Overlap with sink direction
    sink_overlap = 0.0
    if sink_dir is not None:
        # How much of the top-k variance aligns with sink direction?
        top_k_mask = np.zeros(len(var_per_dim))
        top_k_mask[top_k_idx] = 1.0
        sink_in_topk = np.abs(sink_dir[top_k_idx]).sum() / (np.abs(sink_dir).sum() + 1e-10) * 100
        sink_overlap = float(sink_in_topk)

    return {
        'top_k_var_frac': float(top_k_var_frac),
        'top_k_dims': top_k_idx.tolist(),
        'sink_overlap': sink_overlap,
    }


# ============================================================
# Metric 3: RankMe / Effective Rank (Li et al. 2025, Garrido 2023)
# ============================================================

def compute_rankme(tokens):
    """RankMe: effective rank based on Shannon entropy of normalized singular values."""
    if len(tokens) < 3:
        return 1.0
    centered = tokens - tokens.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    # Normalize singular values to form a probability distribution
    S_pos = S[S > 1e-10]
    p = S_pos / S_pos.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(entropy))


def compute_eigenspectrum_alpha(tokens):
    """Estimate power-law decay coefficient of eigenspectrum (alpha-ReQ style)."""
    if len(tokens) < 3:
        return 0.0
    centered = tokens - tokens.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    eigs = S ** 2
    eigs = eigs[eigs > 1e-10]
    if len(eigs) < 5:
        return 0.0
    # Fit log-log line: log(eig) = -alpha * log(rank) + c
    ranks = np.arange(1, len(eigs) + 1)
    log_ranks = np.log(ranks)
    log_eigs = np.log(eigs)
    # Linear regression
    A = np.vstack([log_ranks, np.ones(len(log_ranks))]).T
    result = np.linalg.lstsq(A, log_eigs, rcond=None)
    alpha = -result[0][0]  # negative slope = decay rate
    return float(alpha)


def compute_e1_share(tokens):
    """First eigenvalue share (%)."""
    if len(tokens) < 3:
        return 0.0
    centered = tokens - tokens.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    eigs = S ** 2
    total = eigs.sum()
    if total < 1e-12:
        return 100.0
    return float(eigs[0] / total * 100)


# ============================================================
# Metric 4: CKA (Raghu et al. 2021)
# ============================================================

def linear_CKA(X, Y):
    """Linear CKA between two representation matrices."""
    # X, Y: (n, d1), (n, d2)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtX = X.T @ X  # (d1, d1)
    YtY = Y.T @ Y  # (d2, d2)
    XtY = X.T @ Y  # (d1, d2)
    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def compute_cka_matrix(layer_tokens_list, max_tokens=5000):
    """Compute CKA between all pairs of layers."""
    n_layers = len(layer_tokens_list)
    cka = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(i, n_layers):
            Xi = layer_tokens_list[i][:max_tokens]
            Xj = layer_tokens_list[j][:max_tokens]
            c = linear_CKA(Xi, Xj)
            cka[i, j] = c
            cka[j, i] = c
    return cka


# ============================================================
# Data collection
# ============================================================

def collect_hidden_states(model, tokenizer, device, max_tokens=200000, seq_len=256):
    """Collect hidden states from OpenWebText."""
    print("Loading OpenWebText...")
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    all_tokens = []
    for ex in ds:
        toks = tokenizer.encode(ex['text'])
        all_tokens.extend(toks)
        if len(all_tokens) >= max_tokens:
            break

    tokens = np.array(all_tokens[:max_tokens], dtype=np.int64)
    n = len(tokens) // seq_len * seq_len
    chunks = torch.tensor(tokens[:n].reshape(-1, seq_len), dtype=torch.long)
    print(f"  {len(tokens)} tokens, {len(chunks)} sequences")

    model.eval()
    all_hidden = None
    n_batches = min(len(chunks) // 32, 20)

    with torch.no_grad():
        for i in range(n_batches):
            batch = chunks[i*32:(i+1)*32].to(device)
            out = model(batch, output_hidden_states=True)

            if all_hidden is None:
                all_hidden = [[] for _ in range(len(out.hidden_states))]

            for li, hs in enumerate(out.hidden_states):
                all_hidden[li].append(hs.float().cpu().numpy())

    for li in range(len(all_hidden)):
        all_hidden[li] = np.concatenate(all_hidden[li], axis=0)

    print(f"  {len(all_hidden)} layers, shape {all_hidden[0].shape}")
    return all_hidden  # list of (n_seqs, seq_len, dim)


# ============================================================
# Main analysis
# ============================================================

def analyze_model(model_name, device, save_dir):
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_hidden = collect_hidden_states(model, tokenizer, device)
    n_layers = len(all_hidden)
    dim = all_hidden[0].shape[-1]

    del model
    torch.cuda.empty_cache()

    results = {
        'model': model_name,
        'n_layers': n_layers,
        'dim': dim,
        'layers': []
    }

    # Sample layers for CKA (every 1 layer for small models)
    cka_layers_raw = []
    cka_layers_desink = []

    for li in range(n_layers):
        hs = all_hidden[li]  # (n_seqs, seq_len, dim)

        # Compute sink direction for this layer
        sink_dir = compute_sink_direction(hs)

        # All tokens flattened
        all_tok = hs.reshape(-1, dim).astype(np.float64)
        # Without pos-0
        no_p0 = hs[:, 1:, :].reshape(-1, dim).astype(np.float64)
        # De-sinked (all tokens)
        all_tok_ds = desink(all_tok, sink_dir)

        # --- Metric 1: Anisotropy ---
        aniso_raw = compute_anisotropy(all_tok)
        aniso_no_p0 = compute_anisotropy(no_p0)
        aniso_desink = compute_anisotropy(all_tok_ds)

        # --- Metric 2: Rogue dimensions ---
        rogue_raw = compute_rogue_dims(all_tok, sink_dir)
        rogue_desink = compute_rogue_dims(all_tok_ds, sink_dir)

        # --- Metric 3: RankMe + alpha + E1 ---
        rankme_raw = compute_rankme(all_tok)
        rankme_no_p0 = compute_rankme(no_p0)
        rankme_desink = compute_rankme(all_tok_ds)

        alpha_raw = compute_eigenspectrum_alpha(all_tok)
        alpha_desink = compute_eigenspectrum_alpha(all_tok_ds)

        e1_raw = compute_e1_share(all_tok)
        e1_no_p0 = compute_e1_share(no_p0)
        e1_desink = compute_e1_share(all_tok_ds)

        # Norm ratio
        norms = np.linalg.norm(hs, axis=-1)
        norm_ratio = norms[:, 0].mean() / norms[:, 1:].mean()

        # Store for CKA
        cka_layers_raw.append(all_tok[:5000])
        cka_layers_desink.append(all_tok_ds[:5000])

        layer_result = {
            'layer': li,
            'norm_ratio': float(norm_ratio),
            # Anisotropy (Ethayarajh)
            'aniso_raw': aniso_raw,
            'aniso_no_p0': aniso_no_p0,
            'aniso_desink': aniso_desink,
            # Rogue dims (Timkey)
            'rogue_top5_var_pct_raw': rogue_raw['top_k_var_frac'],
            'rogue_top5_var_pct_desink': rogue_desink['top_k_var_frac'],
            'rogue_sink_overlap': rogue_raw['sink_overlap'],
            # RankMe (Li et al.)
            'rankme_raw': rankme_raw,
            'rankme_no_p0': rankme_no_p0,
            'rankme_desink': rankme_desink,
            # Eigenspectrum alpha
            'alpha_raw': alpha_raw,
            'alpha_desink': alpha_desink,
            # E1 share
            'e1_raw': e1_raw,
            'e1_no_p0': e1_no_p0,
            'e1_desink': e1_desink,
        }
        results['layers'].append(layer_result)

        if li % 2 == 0 or li == n_layers - 1:
            print(f"L{li}: aniso {aniso_raw:.3f}→{aniso_desink:.3f}  "
                  f"rankme {rankme_raw:.0f}→{rankme_desink:.0f}  "
                  f"e1 {e1_raw:.1f}%→{e1_desink:.1f}%  "
                  f"rogue_top5 {rogue_raw['top_k_var_frac']:.1f}%→{rogue_desink['top_k_var_frac']:.1f}%")

    # --- Metric 4: CKA matrix ---
    print("\nComputing CKA matrices...")
    cka_raw = compute_cka_matrix(cka_layers_raw)
    cka_desink = compute_cka_matrix(cka_layers_desink)

    results['cka_raw'] = cka_raw.tolist()
    results['cka_desink'] = cka_desink.tolist()

    # CKA summary: average off-diagonal similarity
    n = len(cka_raw)
    mask = ~np.eye(n, dtype=bool)
    avg_cka_raw = cka_raw[mask].mean()
    avg_cka_desink = cka_desink[mask].mean()
    results['avg_cka_raw'] = float(avg_cka_raw)
    results['avg_cka_desink'] = float(avg_cka_desink)
    print(f"CKA avg off-diag: {avg_cka_raw:.3f} → {avg_cka_desink:.3f}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"{'Layer':<6} {'aniso':>10} {'→desink':>10} {'rankme':>8} {'→desink':>8} "
          f"{'e1%':>6} {'→desink':>8} {'rogue5%':>8} {'→desink':>8}")
    for lr in results['layers']:
        li = lr['layer']
        if li % 2 == 0 or li == n_layers - 1:
            print(f"L{li:<5} {lr['aniso_raw']:>9.3f} {lr['aniso_desink']:>9.3f} "
                  f"{lr['rankme_raw']:>7.0f} {lr['rankme_desink']:>7.0f} "
                  f"{lr['e1_raw']:>5.1f}% {lr['e1_desink']:>7.1f}% "
                  f"{lr['rogue_top5_var_pct_raw']:>7.1f}% {lr['rogue_top5_var_pct_desink']:>7.1f}%")

    with open(os.path.join(save_dir, f"a5_{model_name.replace('/', '_')}.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    save_dir = "logs/contamination_audit"
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(42)

    models = ["EleutherAI/pythia-70m", "gpt2"]
    all_results = {}

    for model_name in models:
        results = analyze_model(model_name, device, save_dir)
        all_results[model_name] = results

    # Cross-model comparison
    print(f"\n{'='*70}")
    print("CROSS-MODEL: De-sink impact magnitude")
    print(f"{'='*70}")
    for model_name, res in all_results.items():
        aniso_changes = [lr['aniso_raw'] - lr['aniso_desink'] for lr in res['layers']]
        rankme_changes = [lr['rankme_desink'] - lr['rankme_raw'] for lr in res['layers']]
        e1_changes = [lr['e1_raw'] - lr['e1_desink'] for lr in res['layers']]

        print(f"\n{model_name}:")
        print(f"  Anisotropy reduction: mean={np.mean(aniso_changes):.3f}, "
              f"max={np.max(aniso_changes):.3f} (L{np.argmax(aniso_changes)})")
        print(f"  RankMe increase: mean={np.mean(rankme_changes):.0f}, "
              f"max={np.max(rankme_changes):.0f} (L{np.argmax(rankme_changes)})")
        print(f"  E1 share reduction: mean={np.mean(e1_changes):.1f}%, "
              f"max={np.max(e1_changes):.1f}% (L{np.argmax(e1_changes)})")
        print(f"  CKA: {res['avg_cka_raw']:.3f} → {res['avg_cka_desink']:.3f}")

    print(f"\nAll results saved to {save_dir}/")


if __name__ == "__main__":
    main()
