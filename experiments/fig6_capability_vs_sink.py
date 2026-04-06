"""
Paper 1 Exp B: Sink Emergence vs Capability Emergence Timeline
=================================================================
Align three timelines across Pythia-70M training checkpoints:
1. Sink strength (from B4 data: norm_ratio, E1_ratio, attn_to_p0)
2. De-sinked metric trajectory (from Exp A)
3. Probing accuracy (next-token prediction) at each checkpoint

Question: Do raw metric "phase transitions" align with sink emergence
or with genuine capability acquisition?
"""
import torch
import numpy as np
import json
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000
]

PROBE_LAYERS = [1, 3, 6]  # Pythia-70M has 6 layers
N_TEXTS = 200
TOP_K_CLASSES = 100  # Probe for top-100 most frequent tokens


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
    except Exception:
        texts = [f"The quick brown fox jumps over the lazy dog number {i}."
                 for i in range(N_TEXTS)]
    print(f"  {len(texts)} texts")

    # Load existing B4 sink data if available
    b4_path = os.path.join(os.path.dirname(__file__), "logs", "b4_v3_pythia_checkpoints.json")
    sink_data = {}
    if os.path.exists(b4_path):
        with open(b4_path) as f:
            b4 = json.load(f)
        for entry in b4 if isinstance(b4, list) else [b4]:
            step = entry.get("step")
            if step is not None:
                sink_data[step] = entry
        print(f"  Loaded B4 sink data for {len(sink_data)} checkpoints")

    t0 = time.time()
    all_results = []

    for ci, step in enumerate(CHECKPOINTS):
        print(f"\n[{ci+1}/{len(CHECKPOINTS)}] Step {step}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-70m", revision=f"step{step}")
            model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/pythia-70m", revision=f"step{step}",
                torch_dtype=torch.float32, device_map=device)
            model.eval()
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        # Extract hidden states + next-token labels
        layer_features = {li: [] for li in PROBE_LAYERS}
        all_labels = []

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            input_ids = inputs["input_ids"][0].cpu().numpy()

            for li in PROBE_LAYERS:
                if li + 1 < len(out.hidden_states):
                    h = out.hidden_states[li + 1][0].float().cpu().numpy()
                    # Use positions [:-1] as features, [1:] as labels
                    layer_features[li].append(h[:-1])

            all_labels.append(input_ids[1:])

        # Concatenate
        labels = np.concatenate(all_labels)

        # Filter to top-K most frequent tokens
        unique, counts = np.unique(labels, return_counts=True)
        top_k_tokens = unique[np.argsort(counts)[-TOP_K_CLASSES:]]
        mask = np.isin(labels, top_k_tokens)
        labels_filtered = labels[mask]

        # Remap labels to 0..K-1
        label_map = {t: i for i, t in enumerate(sorted(top_k_tokens))}
        labels_mapped = np.array([label_map[l] for l in labels_filtered])

        checkpoint_result = {"step": step, "layers": {}}

        for li in PROBE_LAYERS:
            if li not in layer_features or not layer_features[li]:
                continue
            features = np.concatenate(layer_features[li], axis=0)
            features_filtered = features[mask]

            if len(features_filtered) < 50:
                continue

            # Raw probe
            try:
                scores_raw = cross_val_score(
                    LogisticRegression(max_iter=200, solver='lbfgs', C=1.0),
                    features_filtered, labels_mapped,
                    cv=3, scoring='accuracy'
                )
                acc_raw = float(np.mean(scores_raw))
            except Exception:
                acc_raw = 0.0

            # De-sinked probe
            X = features_filtered - features_filtered.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            sink_dir = Vt[0]
            proj = features_filtered @ sink_dir
            features_ds = features_filtered - np.outer(proj, sink_dir)

            try:
                scores_ds = cross_val_score(
                    LogisticRegression(max_iter=200, solver='lbfgs', C=1.0),
                    features_ds, labels_mapped,
                    cv=3, scoring='accuracy'
                )
                acc_ds = float(np.mean(scores_ds))
            except Exception:
                acc_ds = 0.0

            delta = acc_ds - acc_raw

            checkpoint_result["layers"][li] = {
                "probe_raw": acc_raw,
                "probe_desinked": acc_ds,
                "probe_delta": delta,
                "n_samples": len(features_filtered),
            }
            print(f"  L{li}: probe {acc_raw:.3f}→{acc_ds:.3f} (Δ={delta:+.3f})")

        # Add sink data if available
        if step in sink_data:
            sd = sink_data[step]
            checkpoint_result["sink"] = {
                "norm_ratio": sd.get("norm_ratio"),
                "e1_ratio": sd.get("e1_ratio"),
                "attn_to_p0": sd.get("attn_to_p0"),
            }

        all_results.append(checkpoint_result)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print("SINK EMERGENCE vs CAPABILITY EMERGENCE TIMELINE")
    print(sep)

    for li in PROBE_LAYERS:
        print(f"\n--- Layer {li} ---")
        print(f"{'Step':<10} {'probe_raw':>10} {'probe_ds':>10} {'Δ':>8} "
              f"{'norm_r':>8} {'e1_r':>8}")
        print("-" * 55)
        for r in all_results:
            if li in r.get("layers", {}):
                lr = r["layers"][li]
                sink = r.get("sink", {})
                nr = f"{sink.get('norm_ratio', 0):.2f}" if sink.get("norm_ratio") else "?"
                er = f"{sink.get('e1_ratio', 0):.2f}" if sink.get("e1_ratio") else "?"
                print(f"{r['step']:<10} {lr['probe_raw']:>9.3f} "
                      f"{lr['probe_desinked']:>9.3f} {lr['probe_delta']:>+7.3f} "
                      f"{nr:>8} {er:>8}")

    os.makedirs("logs/desink_timeline", exist_ok=True)
    with open("logs/desink_timeline/desink_timeline.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved. Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
