"""Appendix D: Sink-unembedding geometry analysis (final-layer non-sink structure)."""

import torch
import numpy as np
import json
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layers(model):
    for chain in ['transformer.h', 'gpt_neox.layers', 'model.layers']:
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
    raise ValueError("No layers found")


def get_lm_head(model):
    """Return the unembedding (lm_head) weight matrix."""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight.detach().float().cpu()
    if hasattr(model, 'embed_out'):
        return model.embed_out.weight.detach().float().cpu()
    raise ValueError("Cannot find lm_head")


def collect_hidden_states(model, tokenizer, texts, device, max_len=256):
    """Collect hidden states at every layer, positions 1+ (skip BOS/pos-0)."""
    model.eval()
    all_layers = None

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", max_length=max_len,
                            truncation=True, padding=False)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 10:
                continue

            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if all_layers is None:
                n_layers = len(hidden_states)
                all_layers = [[] for _ in range(n_layers)]

            for li, hs in enumerate(hidden_states):
                # Collect all positions (we'll handle pos-0 separately)
                all_layers[li].append(hs[0].cpu().float())

    return all_layers


def compute_sink_direction(hidden_list):
    """Compute sink direction as mean of position-0 vectors, normalized."""
    pos0_vecs = torch.stack([h[0] for h in hidden_list])
    sink_dir = pos0_vecs.mean(dim=0)
    sink_dir = sink_dir / (sink_dir.norm() + 1e-10)
    return sink_dir


def desink_matrix(H, sink_dir):
    """Project out sink direction from data matrix H (N x d)."""
    proj = (H @ sink_dir.unsqueeze(-1)) * sink_dir.unsqueeze(0)
    return H - proj


def pca_analysis(H, n_components=10):
    """Compute PCA on centered matrix. Return eigenvalues, eigenvectors, explained variance."""
    H_c = H - H.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(H_c, full_matrices=False)
    eigenvalues = (S[:n_components] ** 2) / (S ** 2).sum()
    return eigenvalues.numpy(), Vt[:n_components]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    save_dir = "logs/exp_c1"
    os.makedirs(save_dir, exist_ok=True)

    models_to_test = [
        "gpt2",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
    ]

    # Load text data
    print("Loading OpenWebText...")
    try:
        from datasets import load_dataset
        ds = load_dataset("stas/openwebtext-10k")
        texts = [t for t in ds['train']['text'] if len(t.strip()) > 100]
    except:
        print("Falling back to WikiText-2...")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        texts = [t for t in ds['train']['text'] if len(t.strip()) > 100]

    all_model_results = {}

    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine how many texts to use (target ~200K tokens)
        total_tokens = 0
        use_texts = []
        for t in texts:
            toks = tokenizer(t, truncation=True, max_length=256)['input_ids']
            total_tokens += len(toks)
            use_texts.append(t)
            if total_tokens >= 200000:
                break
        print(f"  Using {len(use_texts)} texts, ~{total_tokens} tokens")

        # Collect hidden states
        print("  Collecting hidden states...")
        all_layers = collect_hidden_states(model, tokenizer, use_texts, device, max_len=256)
        n_layers = len(all_layers)
        d = all_layers[0][0].shape[-1]
        print(f"  {n_layers} layers, dim={d}")

        # Get lm_head and its SVD
        print("  Computing lm_head SVD...")
        W_lm = get_lm_head(model)  # (vocab, d)
        U_lm, S_lm, Vt_lm = torch.linalg.svd(W_lm, full_matrices=False)
        # Vt_lm[i] are the right singular vectors (directions in hidden space)
        # S_lm[i] are singular values
        lm_top_dirs = Vt_lm[:20]  # top 20 directions
        lm_sv_energy = (S_lm[:20] ** 2) / (S_lm ** 2).sum()
        print(f"  lm_head shape: {W_lm.shape}")
        print(f"  Top 5 SV energy: {lm_sv_energy[:5].numpy()}")

        # Per-layer analysis
        layer_results = []
        for li in range(n_layers):
            hidden_list = all_layers[li]

            # Flatten to (N, d) — all tokens from all sequences
            H_all = torch.cat(hidden_list, dim=0)  # (total_tokens, d)

            # Sink direction
            sink_dir = compute_sink_direction(hidden_list)

            # Raw PCA
            raw_evals, raw_evecs = pca_analysis(H_all, n_components=10)

            # De-sinked PCA
            H_ds = desink_matrix(H_all, sink_dir)
            ds_evals, ds_evecs = pca_analysis(H_ds, n_components=10)

            # Alignment: de-sinked PC1 vs lm_head top singular vectors
            ds_pc1 = ds_evecs[0]  # (d,)
            alignments = []
            for k in range(min(20, lm_top_dirs.shape[0])):
                cos_sim = F.cosine_similarity(
                    ds_pc1.unsqueeze(0).float(),
                    lm_top_dirs[k].unsqueeze(0).float()
                ).item()
                alignments.append(abs(cos_sim))

            # Also check: raw PC1 vs lm_head (for comparison)
            raw_pc1 = raw_evecs[0]
            raw_alignments = []
            for k in range(min(20, lm_top_dirs.shape[0])):
                cos_sim = abs(F.cosine_similarity(
                    raw_pc1.unsqueeze(0).float(),
                    lm_top_dirs[k].unsqueeze(0).float()
                ).item())
                raw_alignments.append(cos_sim)

            # Sink direction alignment with lm_head
            sink_alignments = []
            for k in range(min(20, lm_top_dirs.shape[0])):
                cos_sim = abs(F.cosine_similarity(
                    sink_dir.unsqueeze(0).float(),
                    lm_top_dirs[k].unsqueeze(0).float()
                ).item())
                sink_alignments.append(cos_sim)

            result = {
                'layer': li,
                'raw_e1_pct': float(raw_evals[0] * 100),
                'desink_e1_pct': float(ds_evals[0] * 100),
                'raw_top5_pct': float(raw_evals[:5].sum() * 100),
                'desink_top5_pct': float(ds_evals[:5].sum() * 100),
                'desink_pc1_vs_lmhead': alignments,
                'desink_pc1_max_align': float(max(alignments)),
                'desink_pc1_best_sv': int(np.argmax(alignments)),
                'raw_pc1_vs_lmhead': raw_alignments,
                'raw_pc1_max_align': float(max(raw_alignments)),
                'sink_vs_lmhead': sink_alignments,
                'sink_max_align': float(max(sink_alignments)),
            }
            layer_results.append(result)

            # Print for key layers
            if li == 0 or li == n_layers - 1 or li % max(1, n_layers // 6) == 0:
                print(f"  L{li}: raw_E1={raw_evals[0]*100:.1f}% → ds_E1={ds_evals[0]*100:.1f}%"
                      f"  ds_PC1↔lm_head: max={max(alignments):.3f} (SV{np.argmax(alignments)})"
                      f"  sink↔lm_head: max={max(sink_alignments):.3f}")

        # Summary for this model
        final = layer_results[-1]
        print(f"\n  --- Final layer (L{final['layer']}) summary ---")
        print(f"  Raw E1: {final['raw_e1_pct']:.1f}%")
        print(f"  De-sinked E1: {final['desink_e1_pct']:.1f}%")
        print(f"  De-sinked PC1 best alignment with lm_head: "
              f"{final['desink_pc1_max_align']:.3f} (SV{final['desink_pc1_best_sv']})")
        print(f"  Raw PC1 best alignment with lm_head: {final['raw_pc1_max_align']:.3f}")
        print(f"  Sink dir best alignment with lm_head: {final['sink_max_align']:.3f}")

        # Interpretation
        if final['desink_pc1_max_align'] > 0.5:
            print(f"  → STRONG alignment: residual structure directly serves prediction")
        elif final['desink_pc1_max_align'] > 0.2:
            print(f"  → MODERATE alignment: partial link to prediction")
        else:
            print(f"  → WEAK alignment: residual structure is orthogonal to prediction")

        all_model_results[model_name] = {
            'n_layers': n_layers,
            'd': d,
            'lm_head_shape': list(W_lm.shape),
            'lm_sv_energy_top20': lm_sv_energy.tolist(),
            'layer_results': layer_results,
        }

        # Cleanup
        del model, tokenizer, all_layers, W_lm, U_lm, S_lm, Vt_lm
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ============================================================
    # Cross-model summary
    # ============================================================
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY: Final Layer Decomposition")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'raw_E1%':>8} {'ds_E1%':>8} {'ds_PC1↔lm':>10} {'sink↔lm':>10}")

    for model_name, data in all_model_results.items():
        final = data['layer_results'][-1]
        short = model_name.split('/')[-1]
        print(f"{short:<25} {final['raw_e1_pct']:>7.1f}% {final['desink_e1_pct']:>7.1f}% "
              f"{final['desink_pc1_max_align']:>9.3f} {final['sink_max_align']:>9.3f}")

    # Save
    out_path = os.path.join(save_dir, "c1_final_layer_decomposition.json")
    with open(out_path, 'w') as f:
        json.dump(all_model_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


# Need this import for cosine_similarity used in the loop
import torch.nn.functional as F

if __name__ == "__main__":
    main()
