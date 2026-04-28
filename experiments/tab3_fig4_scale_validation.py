"""Table 3 + Figure 4: Scale validation on Pythia-6.9B and GPT-2-XL."""

import torch
import numpy as np
import json
import os
import time

MODELS = [
    ("EleutherAI/pythia-6.9b", [0, 4, 8, 12, 16, 20, 24, 28, 31]),
    ("gpt2-xl", [0, 6, 12, 18, 24, 30, 36, 42, 47]),  # 48 layers
]
N_TEXTS = 200


def compute_metrics(X):
    X_c = X - X.mean(axis=0, keepdims=True)
    cov = X_c.T @ X_c / max(X_c.shape[0] - 1, 1)
    vals = np.linalg.svdvals(cov)
    vals = np.maximum(vals, 0)
    total = vals.sum()
    e1 = vals[0] / total if total > 0 else 0
    p = vals / total if total > 0 else vals
    p = p[p > 1e-10]
    er = np.exp(-np.sum(p * np.log(p))) if len(p) > 0 else 1.0
    sv = np.sqrt(np.maximum(vals, 0))
    sv_total = sv.sum()
    if sv_total > 0:
        p_sv = sv / sv_total
        p_sv = p_sv[p_sv > 1e-10]
        rankme = np.exp(-np.sum(p_sv * np.log(p_sv)))
    else:
        rankme = 1.0
    return {"e1": float(e1), "er": float(er), "rankme": float(rankme)}


def desink(X):
    X_c = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    d = Vt[0]
    d = d / (np.linalg.norm(d) + 1e-10)
    return X - np.outer(X @ d, d)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:N_TEXTS]
    except Exception:
        texts = [f"The quick brown fox jumps. Sentence {i}." for i in range(N_TEXTS)]
    print(f"Texts: {len(texts)}")

    all_results = []
    t0 = time.time()

    for model_name, target_layers in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Target layers: {target_layers}")
        print(f"{'='*60}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        # Extract hidden states
        layer_acts = {li: [] for li in target_layers}

        for idx, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            for li in target_layers:
                if li + 1 < len(out.hidden_states):
                    h = out.hidden_states[li + 1][0].float().cpu().numpy()
                    layer_acts[li].append(h)

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(texts)}")

        # Compute metrics
        print(f"\n{'Layer':<8} {'E1_raw':>8} {'E1_ds':>8} {'ER_raw':>8} {'ER_ds':>8} {'RM_raw':>8} {'RM_ds':>8}")
        print("-" * 56)

        model_results = {"model": model_name, "layers": []}

        for li in target_layers:
            if not layer_acts[li]:
                continue
            acts = np.concatenate(layer_acts[li], axis=0)

            raw = compute_metrics(acts)
            ds = compute_metrics(desink(acts))

            print(f"L{li:<7} {raw['e1']:>7.3f} {ds['e1']:>7.3f} "
                  f"{raw['er']:>7.1f} {ds['er']:>7.1f} "
                  f"{raw['rankme']:>7.1f} {ds['rankme']:>7.1f}")

            model_results["layers"].append({
                "layer": li,
                "raw": raw,
                "desinked": ds,
            })

        # Check if E1 inflation exists
        e1_ratios = [r["raw"]["e1"] / (r["desinked"]["e1"] + 1e-10)
                     for r in model_results["layers"]]
        max_ratio = max(e1_ratios)
        max_layer = model_results["layers"][np.argmax(e1_ratios)]["layer"]

        print(f"\nMax E1 inflation ratio: {max_ratio:.1f}x at L{max_layer}")
        if max_ratio > 2.0:
            print("-> SINK ARTIFACT CONFIRMED at scale")
        else:
            print("-> Minimal sink effect at this scale")

        model_results["max_e1_ratio"] = float(max_ratio)
        model_results["max_e1_layer"] = int(max_layer)
        all_results.append(model_results)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("logs/scale_validation", exist_ok=True)
    with open("logs/scale_validation/scale_validation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll done. Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
