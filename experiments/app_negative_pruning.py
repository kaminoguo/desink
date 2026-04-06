"""
Paper 1 Exp F: De-sinked Layer Pruning
========================================
Prove sink distortion causes wrong layer pruning decisions.

1. Compute Block Influence (BI) score per layer: BI(i) = 1 - cos(mean(X_i), mean(X_{i+1}))
2. Compute de-sinked BI score
3. Compare rankings → prune 25% layers by each ranking → measure PPL
"""
import torch
import numpy as np
import json
import os
import time
from collections import defaultdict


def desink(X):
    """GPU-accelerated desink."""
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).cuda()
    X_c = X - X.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    d = Vt[0]
    d = d / (d.norm() + 1e-10)
    return (X - (X @ d).unsqueeze(-1) * d.unsqueeze(0)).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model selection — prefer Pythia-6.9B, fallback to GPT-2-XL
    MODEL = "EleutherAI/pythia-6.9b"
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
        model.eval()
        n_layers = model.config.num_hidden_layers
        print(f"Loaded {MODEL}: {n_layers} layers")
    except Exception as e:
        print(f"Failed {MODEL}: {e}, trying GPT-2-XL")
        MODEL = "gpt2-xl"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
        model.eval()
        n_layers = model.config.num_hidden_layers
        print(f"Loaded {MODEL}: {n_layers} layers")

    # Load calibration data
    print("Loading calibration data...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:200]
    print(f"  {len(texts)} calibration texts")

    # Step 1: Extract hidden states at each layer
    print("\nExtracting hidden states...")
    layer_hidden = defaultdict(list)  # layer_idx -> list of arrays

    for idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        for li in range(n_layers + 1):  # +1 for embedding layer
            h = out.hidden_states[li][0].float().cpu().numpy()
            layer_hidden[li].append(h)

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(texts)}")

    # Concatenate
    for li in layer_hidden:
        layer_hidden[li] = np.concatenate(layer_hidden[li], axis=0)
    print(f"  Tokens per layer: {layer_hidden[0].shape[0]}")

    # Step 2: Compute Block Influence scores
    print("\nComputing Block Influence scores...")

    def compute_bi(hidden_dict, do_desink=False):
        """BI(i) = 1 - cos(mean(X_i), mean(X_{i+1}))"""
        scores = {}
        for li in range(n_layers):
            X_in = hidden_dict[li]
            X_out = hidden_dict[li + 1]

            if do_desink:
                X_in = desink(X_in)
                X_out = desink(X_out)

            mean_in = X_in.mean(axis=0)
            mean_out = X_out.mean(axis=0)

            cos = np.dot(mean_in, mean_out) / (np.linalg.norm(mean_in) * np.linalg.norm(mean_out) + 1e-10)
            scores[li] = 1.0 - cos
        return scores

    bi_raw = compute_bi(layer_hidden, do_desink=False)
    bi_ds = compute_bi(layer_hidden, do_desink=True)

    # Rankings (lower BI = more redundant = prune first)
    rank_raw = sorted(bi_raw.keys(), key=lambda k: bi_raw[k])
    rank_ds = sorted(bi_ds.keys(), key=lambda k: bi_ds[k])

    n_prune = n_layers // 4  # 25%

    prune_raw = set(rank_raw[:n_prune])
    prune_ds = set(rank_ds[:n_prune])

    print(f"\n{'Layer':<8} {'BI_raw':>10} {'BI_ds':>10} {'rank_raw':>10} {'rank_ds':>10}")
    print("-" * 50)
    for li in range(n_layers):
        rr = rank_raw.index(li)
        rd = rank_ds.index(li)
        marker = ""
        if li in prune_raw and li not in prune_ds:
            marker = " ← RAW-ONLY prune"
        elif li not in prune_raw and li in prune_ds:
            marker = " ← DS-ONLY prune"
        elif li in prune_raw and li in prune_ds:
            marker = " ← BOTH prune"
        print(f"L{li:<7} {bi_raw[li]:>9.6f} {bi_ds[li]:>9.6f} {rr:>10} {rd:>10}{marker}")

    overlap = len(prune_raw & prune_ds)
    print(f"\nPrune {n_prune} layers (25%)")
    print(f"Raw prune set:  {sorted(prune_raw)}")
    print(f"De-sinked set:  {sorted(prune_ds)}")
    print(f"Overlap: {overlap}/{n_prune} ({overlap/n_prune:.0%})")
    print(f"Different layers: {sorted(prune_raw.symmetric_difference(prune_ds))}")

    # Step 3: Evaluate PPL with pruned models
    print("\nEvaluating pruned model PPL...")

    # Load eval data
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if len(t.strip()) > 0])
    eval_encodings = tokenizer(eval_text, return_tensors="pt")

    def eval_ppl(skip_layers=set(), max_tokens=2048):
        """Evaluate PPL by skipping specified layers via hooks."""
        class SkipHook:
            def __init__(self):
                self.handle = None
            def __call__(self, module, input, output):
                # Return input unchanged (skip this layer's transformation)
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
            def register(self, module):
                self.handle = module.register_forward_hook(self)
            def remove(self):
                if self.handle:
                    self.handle.remove()

        # Get transformer layers
        if hasattr(model, 'gpt_neox'):
            layers = list(model.gpt_neox.layers)
        elif hasattr(model, 'transformer'):
            layers = list(model.transformer.h)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = list(model.model.layers)
        else:
            raise RuntimeError("Cannot find layers")

        hooks = []
        for li in skip_layers:
            h = SkipHook()
            h.register(layers[li])
            hooks.append(h)

        # Compute PPL
        input_ids = eval_encodings.input_ids[:, :max_tokens].to(model.device)
        nlls = []
        stride = 512
        for i in range(0, input_ids.size(1) - 1, stride):
            end = min(i + stride, input_ids.size(1) - 1)
            ids = input_ids[:, i:end + 1]
            target = ids.clone()
            target[:, :-1] = -100

            with torch.no_grad():
                out = model(ids, labels=target)
                nlls.append(out.loss.item())

        for h in hooks:
            h.remove()

        return np.exp(np.mean(nlls))

    ppl_full = eval_ppl(skip_layers=set())
    ppl_raw_pruned = eval_ppl(skip_layers=prune_raw)
    ppl_ds_pruned = eval_ppl(skip_layers=prune_ds)

    print(f"\n{'='*60}")
    print("LAYER PRUNING RESULTS")
    print(f"{'='*60}")
    print(f"Full model PPL:        {ppl_full:.2f}")
    print(f"Raw BI pruned PPL:     {ppl_raw_pruned:.2f} (pruned {sorted(prune_raw)})")
    print(f"De-sinked BI pruned PPL: {ppl_ds_pruned:.2f} (pruned {sorted(prune_ds)})")
    print(f"\nPPL increase from raw pruning:     {ppl_raw_pruned - ppl_full:.2f}")
    print(f"PPL increase from de-sinked pruning: {ppl_ds_pruned - ppl_full:.2f}")

    if ppl_ds_pruned < ppl_raw_pruned:
        improvement = ppl_raw_pruned - ppl_ds_pruned
        print(f"\n→ DE-SINKED PRUNING IS BETTER by {improvement:.2f} PPL points")
        print(f"  Sink distortion caused suboptimal pruning. De-sinking fixes it.")
    elif ppl_ds_pruned > ppl_raw_pruned:
        print(f"\n→ Raw pruning is better (negative result)")
    else:
        print(f"\n→ No difference")

    # Save
    os.makedirs("logs/exp_F_pruning", exist_ok=True)
    out = {
        "model": MODEL, "n_layers": n_layers, "n_pruned": n_prune,
        "bi_raw": {str(k): float(v) for k, v in bi_raw.items()},
        "bi_desink": {str(k): float(v) for k, v in bi_ds.items()},
        "prune_raw": sorted(prune_raw), "prune_ds": sorted(prune_ds),
        "ppl_full": float(ppl_full), "ppl_raw_pruned": float(ppl_raw_pruned), "ppl_ds_pruned": float(ppl_ds_pruned),
    }
    with open("logs/exp_F_pruning/layer_pruning.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved. Total time: {time.time()-t0:.0f}s")


t0 = time.time()
if __name__ == "__main__":
    main()
