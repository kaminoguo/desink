#!/usr/bin/env python3
"""
Exp 95: Position-0 Matched-Random PPL Control
===============================================
Exp91: removing carrier from position-0 → up to 70x PPL per layer.
But is carrier direction special, or does ANY same-size perturbation to position-0 break it?

For each layer:
  1. Sink-only carrier ablation (remove carrier proj from pos-0 in residual)
  2. Sink-only matched-random ablation (remove random dir, same magnitude)
  3. Compare PPL

If carrier >> random: carrier direction encodes specific KEY info.
If carrier ≈ random: position-0 is just fragile to any perturbation.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time, os, gc, json

DEVICE = "cuda"
# os.environ["HF_HOME"] = ...  # set if needed
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "exp95_output.log")

MODEL_NAME = "EleutherAI/pythia-410m"
N_RANDOM_TRIALS = 5

EVAL_TEXTS = [
    "The capital of France is Paris, a city known for its art and culture",
    "In 1969, the first humans landed on the moon during the Apollo 11 mission",
    "Machine learning is a subset of artificial intelligence that focuses on data",
    "The history of artificial intelligence began in antiquity with myths and stories",
    "Quantum mechanics is a fundamental theory in physics that describes nature at small scales",
    "Climate change refers to long-term shifts in temperatures and weather patterns globally",
    "The structure of DNA was discovered by Watson and Crick in nineteen fifty three",
    "Philosophy is the study of general and fundamental questions about existence and knowledge",
    "Neural networks are computing systems inspired by biological neural networks in brains",
    "The Renaissance was a period of cultural rebirth that began in Italy around thirteen hundred",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy for growth",
    "The theory of relativity was developed by Albert Einstein in the early twentieth century",
]


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def get_layers(model):
    for chain in ['gpt_neox.layers', 'model.layers', 'transformer.h']:
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
    raise ValueError("No layers")


def get_carrier_directions(model, tokenizer, texts, device):
    layers = get_layers(model)
    n_layers = len(layers)

    per_layer = {i: [] for i in range(n_layers)}
    hooks = []
    for i, layer in enumerate(layers):
        ln = layer.input_layernorm
        def _h(idx):
            def fn(mod, inp, out):
                h = inp[0] if isinstance(inp, tuple) else inp
                per_layer[idx].append(h.detach().float().cpu().squeeze(0))
            return fn
        hooks.append(ln.register_forward_hook(_h(i)))

    model.eval()
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(device)
            model(**ids)
    for h in hooks:
        h.remove()

    carriers = {}
    pos0_proj_mags = {}
    for i in range(n_layers):
        H = torch.cat(per_layer[i], dim=0)
        H_c = H - H.mean(0, keepdim=True)
        _, S, Vt = torch.linalg.svd(H_c, full_matrices=False)
        pc1 = Vt[0]
        carriers[i] = pc1.to(device)
        carriers[f"energy_{i}"] = (S[0] ** 2).item() / (S ** 2).sum().item()

        # Compute mean |projection| of position-0 tokens onto carrier
        pos0_tokens = torch.stack([per_layer[i][j][0] for j in range(len(per_layer[i]))])
        pos0_proj = (pos0_tokens @ pc1).abs().mean().item()
        pos0_proj_mags[i] = pos0_proj

        del H, H_c, S, Vt

    del per_layer
    gc.collect()
    return carriers, pos0_proj_mags


def compute_ppl(model, tokenizer, texts, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(device)
            out = model(ids.input_ids, labels=ids.input_ids)
            n = ids.input_ids.shape[1] - 1
            total_loss += out.loss.item() * n
            total_tokens += n
    return float(np.exp(total_loss / total_tokens))


def main():
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 70)
    log("EXP 95: POSITION-0 MATCHED-RANDOM PPL CONTROL")
    log("=" * 70)
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Model: {MODEL_NAME}")
    log(f"Question: Is carrier direction special for position-0, or any perturbation breaks it?")
    log(f"Random trials: {N_RANDOM_TRIALS}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    layers = get_layers(model)
    n_layers = len(layers)

    log(f"\n  Computing carrier directions + pos-0 projection magnitudes...")
    carriers, pos0_mags = get_carrier_directions(model, tokenizer, EVAL_TEXTS, DEVICE)

    log(f"  Baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, EVAL_TEXTS, DEVICE)
    log(f"  Baseline PPL: {baseline_ppl:.2f}")

    # Show pos-0 projection magnitudes
    log(f"\n  Position-0 carrier projection magnitudes:")
    for i in range(0, n_layers, 4):
        log(f"    L{i}: |proj|={pos0_mags[i]:.2f}  CE={carriers.get(f'energy_{i}',0)*100:.1f}%")

    # ================================================================
    # Sink-only carrier ablation (per layer)
    # ================================================================
    log(f"\n{'=' * 70}")
    log(f"SINK-ONLY CARRIER ABLATION (per layer)")
    log(f"{'=' * 70}")

    carrier_results = []
    for li in range(n_layers):
        cd = carriers[li]

        def _ablate_sink_carrier(cd_):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] > 0:
                    sink = h[:, 0:1, :]
                    proj = (sink @ cd_).unsqueeze(-1) * cd_.unsqueeze(0).unsqueeze(0)
                    h_new = h.clone()
                    h_new[:, 0:1, :] = sink - proj
                    if isinstance(out, tuple):
                        return (h_new,) + out[1:]
                    return h_new
                return out
            return fn

        handle = layers[li].register_forward_hook(_ablate_sink_carrier(cd))
        ppl = compute_ppl(model, tokenizer, EVAL_TEXTS, DEVICE)
        handle.remove()

        ratio = ppl / baseline_ppl
        carrier_results.append({"layer": li, "ppl": ppl, "ratio": ratio})
        log(f"  L{li:>2d}: PPL={ppl:>10.1f}  ratio={ratio:>8.1f}x")

    # ================================================================
    # Sink-only matched-random ablation (per layer, N trials)
    # ================================================================
    log(f"\n{'=' * 70}")
    log(f"SINK-ONLY MATCHED-RANDOM ABLATION (per layer, {N_RANDOM_TRIALS} trials)")
    log(f"{'=' * 70}")

    random_results = {li: [] for li in range(n_layers)}

    for trial in range(N_RANDOM_TRIALS):
        log(f"\n  Trial {trial + 1}/{N_RANDOM_TRIALS}...")
        for li in range(n_layers):
            cd = carriers[li]
            target_mag = pos0_mags[li]

            # Random direction
            rd = torch.randn(cd.shape[0], device=DEVICE)
            rd = rd / rd.norm()

            def _ablate_sink_random(rd_, target_mag_):
                def fn(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] > 0:
                        sink = h[:, 0:1, :]
                        raw_proj = (sink @ rd_)  # [1, 1]
                        raw_mag = raw_proj.abs().mean()
                        scale = target_mag_ / (raw_mag.item() + 1e-12)
                        removal = (raw_proj * scale).unsqueeze(-1) * rd_.unsqueeze(0).unsqueeze(0)
                        h_new = h.clone()
                        h_new[:, 0:1, :] = sink - removal
                        if isinstance(out, tuple):
                            return (h_new,) + out[1:]
                        return h_new
                    return out
                return fn

            handle = layers[li].register_forward_hook(_ablate_sink_random(rd, target_mag))
            ppl = compute_ppl(model, tokenizer, EVAL_TEXTS, DEVICE)
            handle.remove()

            random_results[li].append(ppl / baseline_ppl)

        # Print summary for this trial
        sample_layers = [5, 7, 9, 11, 14]
        parts = [f"L{li}={np.mean(random_results[li]):>.1f}x" for li in sample_layers]
        log(f"    {', '.join(parts)}")

    # ================================================================
    # SUMMARY
    # ================================================================
    log(f"\n{'=' * 70}")
    log(f"SUMMARY")
    log(f"{'=' * 70}")

    log(f"\n  Baseline PPL: {baseline_ppl:.2f}")
    log(f"\n  {'Layer':>6s}  {'Carrier':>9s}  {'Random':>14s}  {'C/R ratio':>10s}  {'CE':>6s}")
    log(f"  {'-' * 55}")

    carrier_special_layers = 0
    carrier_notspecial_layers = 0

    for li in range(n_layers):
        cr = carrier_results[li]["ratio"]
        rr_mean = np.mean(random_results[li])
        rr_std = np.std(random_results[li])
        ce = carriers.get(f"energy_{li}", 0)
        c_over_r = cr / (rr_mean + 1e-8)

        marker = ""
        if cr > 2.0:  # significant damage
            if c_over_r > 2.0:
                marker = " ← CARRIER SPECIAL"
                carrier_special_layers += 1
            elif c_over_r < 0.5:
                marker = " ← RANDOM WORSE"
                carrier_notspecial_layers += 1
            else:
                marker = " ← SIMILAR"
                carrier_notspecial_layers += 1

        log(f"  L{li:>4d}  {cr:>9.1f}x  {rr_mean:>7.1f}x±{rr_std:>4.1f}  {c_over_r:>10.1f}x  {ce*100:>5.1f}%{marker}")

    log(f"\n  VERDICT:")
    damaged_layers = [li for li in range(n_layers) if carrier_results[li]["ratio"] > 2.0]
    if damaged_layers:
        carrier_ratios = [carrier_results[li]["ratio"] for li in damaged_layers]
        random_ratios = [np.mean(random_results[li]) for li in damaged_layers]
        avg_c = np.mean(carrier_ratios)
        avg_r = np.mean(random_ratios)
        log(f"    Layers with >2x carrier damage: {len(damaged_layers)}")
        log(f"    Avg carrier PPL ratio: {avg_c:.1f}x")
        log(f"    Avg matched-random PPL ratio: {avg_r:.1f}x")
        log(f"    Carrier / Random: {avg_c / (avg_r + 1e-8):.1f}x")

        if avg_c / (avg_r + 1e-8) > 2.0:
            log(f"    >>> CARRIER IS SPECIAL: same-size random perturbation causes much less damage")
            log(f"    >>> Carrier direction encodes specific KEY information at position-0")
        elif avg_c / (avg_r + 1e-8) > 1.3:
            log(f"    >>> PARTIALLY SPECIAL: carrier is worse but random also damages")
        else:
            log(f"    >>> NOT SPECIAL: any same-size perturbation to position-0 causes similar damage")
            log(f"    >>> Position-0 is simply fragile — carrier is just the biggest direction")

    # Save
    json_path = os.path.join(LOG_DIR, "exp95_data.json")
    with open(json_path, "w") as f:
        json.dump({
            "baseline_ppl": baseline_ppl,
            "carrier_results": carrier_results,
            "random_results": {str(k): v for k, v in random_results.items()},
            "pos0_proj_mags": pos0_mags,
        }, f, indent=2)

    elapsed = time.time() - t0
    log(f"\nTotal: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    log("Done.")


if __name__ == "__main__":
    main()
