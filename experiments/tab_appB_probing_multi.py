"""
Exp C2: Systematic Probing — De-Sink as a Prescriptive Tool
=============================================================
B3 showed de-sink improves probing at L6 (final) by +12.1% on Pythia-70M.
This was one model, one task, one dataset.

C2 scales this to:
  - Models: GPT-2, Pythia-70M, Pythia-160M
  - Task: next-token prediction probing (top-500 tokens)
  - Data: WikiText-2 (500 passages, max_len=128)
  - Seeds: 3 per condition
  - Controls: 5 random directions per layer

Key question: Is de-sink probing improvement systematic across models?
If yes → prescriptive claim: "de-sink before probing" as a methodology recommendation.
If no → the B3 result is model-specific, not generalizable.

Also measures: what fraction of total variance is captured by sink direction
at each layer, to show the sink is not just a PCA artifact but actively
interferes with linear readout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Data Collection
# ============================================================

def collect_hidden_states(model, tokenizer, texts, device, max_len=128):
    model.eval()
    all_hidden_layers = None
    all_input_ids = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", max_length=max_len,
                            truncation=True, padding=False)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 10:
                continue

            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if all_hidden_layers is None:
                n_layers = len(hidden_states)
                all_hidden_layers = [[] for _ in range(n_layers)]

            for li, hs in enumerate(hidden_states):
                all_hidden_layers[li].append(hs[0].cpu())

            all_input_ids.append(input_ids[0].cpu())

    return all_hidden_layers, all_input_ids


def prepare_probe_data(hidden_list, input_ids_list, top_k_tokens=500):
    """Prepare next-token prediction data for probing."""
    token_counts = {}
    for ids in input_ids_list:
        for t in ids[1:].tolist():
            token_counts[t] = token_counts.get(t, 0) + 1

    sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
    top_tokens = {t: idx for idx, (t, _) in enumerate(sorted_tokens[:top_k_tokens])}
    n_classes = len(top_tokens)

    features = []
    labels = []

    for hidden, ids in zip(hidden_list, input_ids_list):
        seq_len = min(hidden.shape[0], ids.shape[0])
        for i in range(seq_len - 1):
            next_tok = ids[i + 1].item()
            if next_tok in top_tokens:
                features.append(hidden[i])
                labels.append(top_tokens[next_tok])

    if len(features) == 0:
        return None, None, 0

    features = torch.stack(features).float()
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels, n_classes


# ============================================================
# Linear Probe
# ============================================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(features, labels, n_classes, seed=0, epochs=15, lr=1e-3, batch_size=512):
    torch.manual_seed(seed)
    n = len(features)
    perm = torch.randperm(n)
    n_train = int(0.8 * n)

    train_feat = features[perm[:n_train]]
    train_lab = labels[perm[:n_train]]
    test_feat = features[perm[n_train:]]
    test_lab = labels[perm[n_train:]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe = LinearProbe(features.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_feat = train_feat.to(device)
    train_lab = train_lab.to(device)
    test_feat = test_feat.to(device)
    test_lab = test_lab.to(device)

    dataset = TensorDataset(train_feat, train_lab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        test_logits = probe(test_feat)
        test_acc = (test_logits.argmax(1) == test_lab).float().mean().item() * 100

    return test_acc


def run_probes_multi_seed(features, labels, n_classes, seeds=[0, 1, 2], **kwargs):
    accs = []
    for s in seeds:
        acc = train_probe(features, labels, n_classes, seed=s, **kwargs)
        accs.append(acc)
    return np.mean(accs), np.std(accs)


# ============================================================
# De-sink Utilities
# ============================================================

def compute_sink_direction(hidden_list):
    """Mean of position-0 vectors, normalized."""
    pos0 = torch.stack([h[0] for h in hidden_list]).float()
    d = pos0.mean(dim=0)
    d = d / (d.norm() + 1e-10)
    return d


def desink(features, sink_dir):
    proj = (features @ sink_dir.unsqueeze(-1)) * sink_dir.unsqueeze(0)
    return features - proj


def random_direction(d, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    v /= np.linalg.norm(v)
    return torch.from_numpy(v)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    save_dir = "logs/exp_c2"
    os.makedirs(save_dir, exist_ok=True)

    models_to_test = [
        "gpt2",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
    ]

    # Load text data
    print("Loading WikiText-2...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = [t for t in ds['test']['text'] if len(t.strip()) > 50]
    if len(texts) < 100:
        texts = [t for t in ds['train']['text'] if len(t.strip()) > 50][:500]
    texts = texts[:500]
    print(f"  {len(texts)} text passages")

    seeds = [0, 1, 2]
    n_random_controls = 5
    all_model_results = {}

    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("  Collecting hidden states...")
        all_hidden_layers, all_input_ids = collect_hidden_states(
            model, tokenizer, texts, device, max_len=128
        )

        n_layers = len(all_hidden_layers)
        d = all_hidden_layers[0][0].shape[-1]
        n_seqs = len(all_input_ids)
        total_tokens = sum(ids.shape[0] for ids in all_input_ids)
        print(f"  {n_layers} layers, dim={d}")
        print(f"  {n_seqs} sequences, {total_tokens} total tokens")

        layer_results = []

        for li in range(n_layers):
            hidden_list = all_hidden_layers[li]

            features, labels, n_classes = prepare_probe_data(
                hidden_list, all_input_ids, top_k_tokens=500
            )

            if features is None or len(features) < 200:
                print(f"  L{li}: skipping (insufficient data)")
                continue

            # Sink direction
            sink_dir = compute_sink_direction(hidden_list)

            # Variance captured by sink
            total_var = features.var(dim=0).sum().item()
            proj_sink = (features @ sink_dir.unsqueeze(-1)) * sink_dir.unsqueeze(0)
            sink_var = proj_sink.var(dim=0).sum().item()
            sink_var_pct = sink_var / total_var * 100 if total_var > 0 else 0

            # Raw probing
            raw_mean, raw_std = run_probes_multi_seed(features, labels, n_classes, seeds=seeds)

            # De-sinked probing
            features_ds = desink(features, sink_dir)
            ds_mean, ds_std = run_probes_multi_seed(features_ds, labels, n_classes, seeds=seeds)

            # Random direction controls
            rand_accs = []
            for ri in range(n_random_controls):
                rd = random_direction(d, seed=100 * li + ri)
                features_rd = desink(features, rd)
                rd_mean, _ = run_probes_multi_seed(features_rd, labels, n_classes, seeds=seeds)
                rand_accs.append(rd_mean)

            rand_mean = np.mean(rand_accs)
            rand_std = np.std(rand_accs)

            delta_ds = ds_mean - raw_mean
            delta_rand = rand_mean - raw_mean

            result = {
                'layer': li,
                'n_samples': len(features),
                'n_classes': n_classes,
                'sink_var_pct': float(sink_var_pct),
                'raw_test_mean': float(raw_mean),
                'raw_test_std': float(raw_std),
                'desink_test_mean': float(ds_mean),
                'desink_test_std': float(ds_std),
                'rand_test_mean': float(rand_mean),
                'rand_test_std': float(rand_std),
                'delta_desink': float(delta_ds),
                'delta_rand': float(delta_rand),
            }
            layer_results.append(result)

            marker = ""
            if delta_ds > 1.0:
                marker = " ← DE-SINK HELPS"
            elif delta_ds < -1.0:
                marker = " ← DE-SINK HURTS"

            print(f"  L{li}: raw={raw_mean:.1f}% ds={ds_mean:.1f}% "
                  f"(Δ={delta_ds:+.1f}%) rand={rand_mean:.1f}% sink_var={sink_var_pct:.1f}%{marker}")

        # Model summary
        print(f"\n  --- {model_name} Summary ---")
        print(f"  {'Layer':<8} {'sink_var%':>10} {'raw':>8} {'desink':>8} {'Δ_ds':>8} {'Δ_rand':>8}")
        for lr in layer_results:
            print(f"  L{lr['layer']:<7} {lr['sink_var_pct']:>9.1f}% "
                  f"{lr['raw_test_mean']:>7.1f}% {lr['desink_test_mean']:>7.1f}% "
                  f"{lr['delta_desink']:>+7.1f}% {lr['delta_rand']:>+7.1f}%")

        all_model_results[model_name] = {
            'n_layers': n_layers,
            'd': d,
            'n_sequences': n_seqs,
            'total_tokens': total_tokens,
            'layer_results': layer_results,
        }

        # Cleanup
        del model, tokenizer, all_hidden_layers, all_input_ids
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ============================================================
    # Cross-model summary
    # ============================================================
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY: De-Sink Probing Improvement")
    print(f"{'='*70}")

    for model_name, data in all_model_results.items():
        short = model_name.split('/')[-1]
        lrs = data['layer_results']
        if not lrs:
            continue

        # Find layers where de-sink helps most
        best = max(lrs, key=lambda x: x['delta_desink'])
        worst = min(lrs, key=lambda x: x['delta_desink'])
        final = lrs[-1]

        print(f"\n  {short}:")
        print(f"    Final layer (L{final['layer']}): Δ = {final['delta_desink']:+.1f}%")
        print(f"    Best improvement: L{best['layer']} Δ = {best['delta_desink']:+.1f}%")
        print(f"    Worst: L{worst['layer']} Δ = {worst['delta_desink']:+.1f}%")

        # Count layers where de-sink helps vs hurts
        helps = sum(1 for lr in lrs if lr['delta_desink'] > 0.5)
        hurts = sum(1 for lr in lrs if lr['delta_desink'] < -0.5)
        neutral = len(lrs) - helps - hurts
        print(f"    Layers helped/neutral/hurt: {helps}/{neutral}/{hurts}")

    # Verdict
    print(f"\n  VERDICT:")
    all_final_deltas = []
    for data in all_model_results.values():
        if data['layer_results']:
            all_final_deltas.append(data['layer_results'][-1]['delta_desink'])

    if all_final_deltas:
        avg_final = np.mean(all_final_deltas)
        if avg_final > 1.0:
            print(f"    De-sink consistently improves final-layer probing (avg Δ={avg_final:+.1f}%)")
            print(f"    → Prescriptive recommendation: de-sink before linear probing")
        elif avg_final > 0:
            print(f"    De-sink mildly improves final-layer probing (avg Δ={avg_final:+.1f}%)")
            print(f"    → Modest evidence for prescriptive recommendation")
        else:
            print(f"    De-sink does NOT consistently improve probing (avg Δ={avg_final:+.1f}%)")
            print(f"    → B3 result may be model-specific")

    # Save
    out_path = os.path.join(save_dir, "c2_probing_multi_model.json")
    with open(out_path, 'w') as f:
        json.dump(all_model_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
