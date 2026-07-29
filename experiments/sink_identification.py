"""Table 1: the leading direction is the attention sink.

For each (model, layer, checkpoint) this pools hidden states over 32,768
OpenWebText tokens, centers, takes s = PC1, splits H_bar into its component
along s and the residual C, and reports:

  e1_H_bar                     leading eigenvalue share of H
  e1_of_C                      leading eigenvalue share of the residual
  p0_concentration_of_sink_dir fraction of the projection onto s carried by
                               the rows at sequence position 0
  norm_ratio_p0_vs_rest        the sink strength alpha of Section 2

ratio_Cs and cos_content_pc1_sink are zero by construction (s is PC1 and the
sink component is fit by projection); they are kept as a numerical check.
"""
import os, sys, json, time, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEQ_LEN = 256
N_BATCHES = 4
BATCH_SIZE = 32
TOKEN_CAP = 60_000
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "data", "sink_identification.json")

CONFIGS = [
    ("EleutherAI/pythia-70m",  3,  [512, 2000, 4000, 8000, 64000, 143000]),
    ("EleutherAI/pythia-160m", 5,  [512, 8000, 143000]),
    ("EleutherAI/pythia-1.4b", 12, [512, 8000, 143000]),
]


def load_tokens(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    all_toks = []
    for ex in ds:
        all_toks.extend(tok.encode(ex["text"]))
        if len(all_toks) >= TOKEN_CAP:
            break
    arr = np.array(all_toks[:TOKEN_CAP], dtype=np.int64)
    return arr


def collect_hidden_states(model, tokens_arr, layer_idx, device):
    n = (len(tokens_arr) // SEQ_LEN) * SEQ_LEN
    chunks = torch.tensor(
        tokens_arr[:n].reshape(-1, SEQ_LEN), dtype=torch.long
    )
    n_batches = min(N_BATCHES, len(chunks) // BATCH_SIZE)
    all_hs = []
    with torch.no_grad():
        for i in range(n_batches):
            batch = chunks[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].to(device)
            out = model(batch, output_hidden_states=True)
            hs = out.hidden_states[layer_idx].float().cpu().numpy()
            all_hs.append(hs)
    return np.concatenate(all_hs, axis=0)  # (N_seqs, T, d)


def analyze(hs, device):
    """Compute R3 metrics on hidden states (N_seqs, T, d)."""
    N_seqs, T, d = hs.shape
    H = hs.reshape(-1, d).astype(np.float64)  # (N, d)
    H_bar = H - H.mean(axis=0, keepdims=True)

    # SVD on GPU for speed
    H_gpu = torch.from_numpy(H_bar).to(device).float()
    U, S, Vh = torch.linalg.svd(H_gpu, full_matrices=False)
    s = Vh[0].cpu().numpy().astype(np.float64)  # unit
    s = s / np.linalg.norm(s)  # re-normalize

    # ───────── User's spec (trivially ~0 by construction) ─────────
    projections = H_bar @ s  # (N,)
    sink_component_norm_sq = float(np.sum(projections ** 2))  # ||αa⊗s||_F^2 = sum α_i^2 (since ||s||=1)
    sink_norm = float(np.sqrt(sink_component_norm_sq))

    # Explicit C @ s (should be ~float precision)
    # Cs_i = (H_bar_i - (H_bar_i · s) * s) · s = H_bar_i·s - (H_bar_i·s) = 0
    Cs_val = H_bar @ s - projections  # (N,)
    Cs_norm = float(np.linalg.norm(Cs_val))
    ratio_Cs = Cs_norm / sink_norm if sink_norm > 0 else 0.0

    # PC1 of C: compute in float32 on GPU to stay fast
    # C = H_bar - outer(projections, s); equivalent to removing top-1 SVD component
    # So PC1 of C is V[1] of H_bar's SVD (second right singular vector)
    s_pc2 = Vh[1].cpu().numpy().astype(np.float64)
    s_pc2 = s_pc2 / np.linalg.norm(s_pc2)
    cos_sim = float(abs(np.dot(s_pc2, s)))  # ~0 by orthogonality of singular vectors

    # ───────── Non-trivial diagnostics (genuinely test Theorem 2) ─────────
    # (a) Sink direction concentration at position 0
    proj_reshape = projections.reshape(N_seqs, T)
    alpha_p0_sq = float(np.sum(proj_reshape[:, 0] ** 2))
    alpha_rest_sq = float(np.sum(proj_reshape[:, 1:] ** 2))
    alpha_total_sq = alpha_p0_sq + alpha_rest_sq
    p0_concentration = alpha_p0_sq / alpha_total_sq if alpha_total_sq > 0 else 0.0

    # (b) Effective rank-1 share: E1(H_bar) vs E1(C)
    eigs = (S.cpu().numpy().astype(np.float64) ** 2) / len(H_bar)
    total_var = float(eigs.sum())
    e1_H = float(eigs[0] / total_var) if total_var > 0 else 0.0
    e1_C = float(eigs[1] / (total_var - eigs[0])) if total_var > eigs[0] else 0.0

    # (c) Position-0 norm ratio (classic sink measure)
    norms = np.linalg.norm(hs, axis=-1)  # (N_seqs, T)
    p0_norm = float(norms[:, 0].mean())
    rest_norm = float(norms[:, 1:].mean())
    norm_ratio_p0 = p0_norm / rest_norm if rest_norm > 0 else 0.0

    return {
        # user's spec
        "Cs_norm_over_sink_norm": ratio_Cs,
        "cos_content_pc1_sink": cos_sim,
        # non-trivial
        "p0_concentration_of_sink_dir": p0_concentration,
        "e1_H_bar": e1_H,
        "e1_of_C": e1_C,
        "norm_ratio_p0_vs_rest": norm_ratio_p0,
        # meta
        "n_tokens": int(H.shape[0]),
        "d": int(d),
        "top_singular_value": float(S[0].item()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[R3] Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[R3] GPU: {torch.cuda.get_device_name()}", flush=True)

    # One-time token cache per tokenizer family
    # All three Pythias share the same tokenizer (GPT-NeoX), so we can load once.
    print(f"[R3] Loading openwebtext tokens (cap={TOKEN_CAP})...", flush=True)
    t0 = time.time()
    tokens_arr = load_tokens("EleutherAI/pythia-70m")
    print(f"[R3] Got {len(tokens_arr)} tokens in {time.time()-t0:.1f}s", flush=True)

    results = {
        "experiment": "R3_orthogonality_verification",
        "seq_len": SEQ_LEN,
        "n_batches": N_BATCHES,
        "batch_size": BATCH_SIZE,
        "results": [],
    }

    # Incremental save: if partial file exists, resume
    if os.path.exists(SAVE_PATH):
        try:
            results = json.load(open(SAVE_PATH))
            print(f"[R3] Resumed with {len(results['results'])} prior entries", flush=True)
        except Exception:
            pass
    done = {(r["model"], r["step"]) for r in results["results"]}

    for model_name, layer_idx, steps in CONFIGS:
        for step in steps:
            if (model_name, step) in done:
                print(f"[R3] Skip {model_name} step {step} (already done)", flush=True)
                continue
            print(f"\n[R3] {model_name} step{step} L{layer_idx}", flush=True)
            t0 = time.time()
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    revision=f"step{step}",
                    torch_dtype=torch.float16,
                ).to(device).eval()
            except Exception as e:
                print(f"[R3]   load failed: {e}", flush=True)
                continue
            t_load = time.time() - t0

            t0 = time.time()
            hs = collect_hidden_states(model, tokens_arr, layer_idx, device)
            t_fwd = time.time() - t0

            del model
            gc.collect()
            torch.cuda.empty_cache()

            t0 = time.time()
            metrics = analyze(hs, device)
            t_an = time.time() - t0

            entry = {
                "model": model_name.split("/")[-1],
                "layer": f"L{layer_idx}",
                "step": step,
                **metrics,
                "t_load_s": round(t_load, 2),
                "t_forward_s": round(t_fwd, 2),
                "t_analyze_s": round(t_an, 2),
            }
            results["results"].append(entry)
            print(f"[R3]   ratio_Cs={entry['Cs_norm_over_sink_norm']:.2e}  "
                  f"cos={entry['cos_content_pc1_sink']:.2e}  "
                  f"p0_conc={entry['p0_concentration_of_sink_dir']:.4f}  "
                  f"e1_H={entry['e1_H_bar']:.4f}  e1_C={entry['e1_of_C']:.4f}  "
                  f"normR={entry['norm_ratio_p0_vs_rest']:.2f}x  "
                  f"load={t_load:.1f}s fwd={t_fwd:.1f}s",
                  flush=True)

            with open(SAVE_PATH, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n[R3] DONE. Saved to {SAVE_PATH}", flush=True)
    print(f"[R3] Total entries: {len(results['results'])}", flush=True)


if __name__ == "__main__":
    main()
