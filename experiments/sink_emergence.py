"""Section 7 supporting data: sink formation dynamics across Pythia-70M training checkpoints."""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_pca_e1(tokens):
    if tokens.shape[0] < 3:
        return 0
    centered = tokens - tokens.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    eigs = S ** 2 / len(centered)
    total = eigs.sum()
    if total < 1e-12:
        return 100.0
    return eigs[0] / total * 100


def compute_effective_rank(tokens):
    if tokens.shape[0] < 3:
        return 1.0
    centered = tokens - tokens.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    eigs = S ** 2 / len(centered)
    total = eigs.sum()
    if total < 1e-12:
        return 1.0
    p = eigs / total
    return float(np.exp(-np.sum(p * np.log(p + 1e-10))))


def analyze_checkpoint(model, tokenizer, device, test_tokens, seq_len=256):
    """Analyze a single checkpoint for sink properties."""
    model.eval()

    # Prepare batches
    n = len(test_tokens) // seq_len * seq_len
    chunks = torch.tensor(test_tokens[:n].reshape(-1, seq_len), dtype=torch.long)
    n_batches = min(len(chunks) // 32, 20)  # up to 20 batches of 32

    all_hidden = None
    all_attentions = None

    with torch.no_grad():
        for i in range(n_batches):
            batch = chunks[i*32:(i+1)*32].to(device)
            out = model(batch, output_hidden_states=True, output_attentions=True)

            if all_hidden is None:
                n_layers = len(out.hidden_states)
                all_hidden = [[] for _ in range(n_layers)]
                all_attentions = [[] for _ in range(len(out.attentions))]

            for li, hs in enumerate(out.hidden_states):
                all_hidden[li].append(hs.float().cpu().numpy())

            for li, attn in enumerate(out.attentions):
                # Average attention to pos-0 from positions 10+
                attn_to_p0 = attn[:, :, 10:, 0].float().mean().item()
                all_attentions[li].append(attn_to_p0)

    # Concatenate
    for li in range(len(all_hidden)):
        all_hidden[li] = np.concatenate(all_hidden[li], axis=0)

    # LM head alignment
    head_weight = model.get_output_embeddings().weight.float().cpu().detach().numpy()
    _, S_head, Vt_head = np.linalg.svd(head_weight, full_matrices=False)
    K = min(50, len(S_head))
    V_topK = Vt_head[:K].T

    # Per-layer analysis
    layer_results = []
    for li in range(len(all_hidden)):
        hs = all_hidden[li]  # (n_seqs, seq_len, dim)
        all_tok = hs.reshape(-1, hs.shape[-1])
        no_p0 = hs[:, 1:, :].reshape(-1, hs.shape[-1])

        e1_all = compute_pca_e1(all_tok)
        e1_no_p0 = compute_pca_e1(no_p0)
        sink_ratio = e1_all / e1_no_p0 if e1_no_p0 > 0 else float('inf')

        er_all = compute_effective_rank(all_tok)
        er_no_p0 = compute_effective_rank(no_p0)

        # Norm ratio
        norms = np.linalg.norm(hs, axis=-1)  # (n_seqs, seq_len)
        p0_norm = norms[:, 0].mean()
        other_norm = norms[:, 1:].mean()
        norm_ratio = p0_norm / other_norm if other_norm > 0 else 0

        # LM head alignment
        centered = all_tok - all_tok.mean(axis=0)
        proj = centered @ V_topK
        var_in = np.sum(proj ** 2)
        var_total = np.sum(centered ** 2)
        align = var_in / var_total * 100 if var_total > 0 else 0

        # Attention to pos-0
        attn_p0 = np.mean(all_attentions[li]) if li < len(all_attentions) else 0

        layer_results.append({
            'layer': li,
            'e1_all': float(e1_all),
            'e1_no_p0': float(e1_no_p0),
            'sink_ratio': float(sink_ratio),
            'er_all': float(er_all),
            'er_no_p0': float(er_no_p0),
            'norm_ratio': float(norm_ratio),
            'align': float(align),
            'attn_to_p0': float(attn_p0),
        })

    return layer_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    save_dir = "logs/sink_emergence"
    os.makedirs(save_dir, exist_ok=True)

    # Load tokenizer
    model_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare test data
    print("Loading test data...")
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    all_tokens = []
    for ex in ds:
        toks = tokenizer.encode(ex['text'])
        all_tokens.extend(toks)
        if len(all_tokens) >= 200_000:
            break
    test_tokens = np.array(all_tokens[:200_000], dtype=np.int64)
    print(f"  {len(test_tokens)} test tokens")

    # Pythia checkpoint steps
    # Available: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, ..., 143000
    checkpoint_steps = [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000
    ]

    all_results = {
        'model': model_name,
        'n_test_tokens': len(test_tokens),
        'checkpoints': []
    }

    for step in checkpoint_steps:
        print(f"\n{'='*60}")
        print(f"Checkpoint: step {step}")
        print(f"{'='*60}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=f"step{step}",
                torch_dtype=torch.float16
            ).to(device)
        except Exception as e:
            print(f"  Failed to load step {step}: {e}")
            continue

        # Quick perplexity estimate
        model.eval()
        with torch.no_grad():
            sample = torch.tensor(test_tokens[:256], dtype=torch.long).unsqueeze(0).to(device)
            out = model(sample, labels=sample)
            ppl = torch.exp(out.loss).item()
            ppl = min(ppl, 1e6)  # cap for display

        print(f"  PPL (sample): {ppl:.1f}")

        layer_results = analyze_checkpoint(model, tokenizer, device, test_tokens)

        # Print summary
        print(f"  {'Layer':<6} {'E1_all':>7} {'E1_noP0':>8} {'ratio':>7} {'norm_r':>7} {'attn_p0':>8} {'align':>6}")
        for lr in layer_results:
            li = lr['layer']
            if li % 2 == 0 or li == len(layer_results) - 1:
                print(f"  L{li:<5} {lr['e1_all']:>6.1f}% {lr['e1_no_p0']:>7.1f}% "
                      f"{lr['sink_ratio']:>6.2f}x {lr['norm_ratio']:>6.2f}x "
                      f"{lr['attn_to_p0']:>7.4f} {lr['align']:>5.1f}%")

        checkpoint_result = {
            'step': step,
            'ppl': float(ppl),
            'layers': layer_results
        }
        all_results['checkpoints'].append(checkpoint_result)

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Summary: Sink emergence over training
    # ============================================================
    print(f"\n{'='*70}")
    print("SINK EMERGENCE OVER TRAINING")
    print(f"{'='*70}")

    # Pick representative layers
    if all_results['checkpoints']:
        n_layers = len(all_results['checkpoints'][0]['layers'])
        mid_layer = n_layers // 2
        late_layer = n_layers - 2  # second to last

        print(f"\n--- Middle layer (L{mid_layer}) ---")
        print(f"{'Step':>8} {'PPL':>10} {'sink_ratio':>11} {'norm_ratio':>11} {'attn_p0':>9}")
        for cp in all_results['checkpoints']:
            lr = cp['layers'][mid_layer]
            print(f"{cp['step']:>8} {cp['ppl']:>10.1f} {lr['sink_ratio']:>10.2f}x "
                  f"{lr['norm_ratio']:>10.2f}x {lr['attn_to_p0']:>8.4f}")

        print(f"\n--- Late layer (L{late_layer}) ---")
        print(f"{'Step':>8} {'PPL':>10} {'sink_ratio':>11} {'norm_ratio':>11} {'attn_p0':>9}")
        for cp in all_results['checkpoints']:
            lr = cp['layers'][late_layer]
            print(f"{cp['step']:>8} {cp['ppl']:>10.1f} {lr['sink_ratio']:>10.2f}x "
                  f"{lr['norm_ratio']:>10.2f}x {lr['attn_to_p0']:>8.4f}")

    # Find when sink emerges (norm_ratio > 2x)
    print(f"\n--- Sink emergence detection ---")
    for cp in all_results['checkpoints']:
        max_norm = max(lr['norm_ratio'] for lr in cp['layers'])
        max_sink = max(lr['sink_ratio'] for lr in cp['layers'])
        max_attn = max(lr['attn_to_p0'] for lr in cp['layers'] if lr['layer'] < len(cp['layers']) - 1)
        flag = " ← SINK" if max_norm > 2.0 else ""
        print(f"  Step {cp['step']:>6}: max_norm={max_norm:.2f}x max_E1_ratio={max_sink:.2f}x "
              f"max_attn_p0={max_attn:.4f}{flag}")

    with open(os.path.join(save_dir, "b4_v3_pythia_checkpoints.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {save_dir}/b4_v3_pythia_checkpoints.json")


if __name__ == "__main__":
    main()
