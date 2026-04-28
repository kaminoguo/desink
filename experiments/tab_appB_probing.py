"""Appendix B: Linear probing with vs without de-sinking (large-scale verification on Pythia-70M)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# De-sink Utilities
# ============================================================

def random_direction(d, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    v /= np.linalg.norm(v)
    return torch.from_numpy(v)


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
        train_logits = probe(train_feat)
        train_acc = (train_logits.argmax(1) == train_lab).float().mean().item() * 100
        test_logits = probe(test_feat)
        test_acc = (test_logits.argmax(1) == test_lab).float().mean().item() * 100

    return train_acc, test_acc


def run_probes_multi_seed(features, labels, n_classes, seeds=[0, 1, 2], **kwargs):
    """Run probe with multiple seeds, return mean and std."""
    train_accs, test_accs = [], []
    for s in seeds:
        tr, te = train_probe(features, labels, n_classes, seed=s, **kwargs)
        train_accs.append(tr)
        test_accs.append(te)
    return np.mean(train_accs), np.std(train_accs), np.mean(test_accs), np.std(test_accs)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    save_dir = "logs/exp_b3_large"
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model_name = "EleutherAI/pythia-70m"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load real text data
    print("Loading WikiText-2...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        # Filter non-empty, non-header lines
        texts = [t for t in ds['test']['text'] if len(t.strip()) > 50]
        print(f"  Got {len(texts)} text passages from WikiText-2 test set")
    except Exception as e:
        print(f"  Failed to load WikiText-2: {e}")
        print("  Falling back to train split or synthetic...")
        texts = []

    if len(texts) < 100:
        # Fallback: use validation or generate
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            texts = [t for t in ds['train']['text'] if len(t.strip()) > 50][:500]
            print(f"  Using {len(texts)} passages from train set")
        except:
            print("  ERROR: Cannot load any text data")
            return

    # Cap at 500 passages for memory
    texts = texts[:500]

    print(f"\nCollecting hidden states from {len(texts)} passages (max_len=128)...")
    all_hidden_layers, all_input_ids = collect_hidden_states(
        model, tokenizer, texts, device, max_len=128
    )

    n_layers = len(all_hidden_layers)
    d = all_hidden_layers[0][0].shape[-1]
    n_seqs = len(all_input_ids)
    total_tokens = sum(ids.shape[0] for ids in all_input_ids)
    print(f"  {n_layers} layers, dim={d}")
    print(f"  {n_seqs} sequences, {total_tokens} total tokens")

    results = {'layers': [], 'model': model_name, 'd': d, 'n_layers': n_layers,
               'n_sequences': n_seqs, 'total_tokens': total_tokens}

    seeds = [0, 1, 2]

    print(f"\nAnalyzing all {n_layers} layers with {len(seeds)} seeds each...\n")

    for li in range(n_layers):
        print(f"{'='*60}")
        print(f"Layer {li}")
        print(f"{'='*60}")

        hidden_list = all_hidden_layers[li]

        features_raw, labels, n_classes = prepare_probe_data(
            hidden_list, all_input_ids, top_k_tokens=500
        )

        if features_raw is None or len(features_raw) < 200:
            print(f"  Skipping: insufficient data ({0 if features_raw is None else len(features_raw)})")
            continue

        print(f"  {len(features_raw)} samples, {n_classes} classes")

        # Sink direction
        pos0_tokens = torch.stack([h[0] for h in hidden_list]).float()
        sink_dir = pos0_tokens.mean(dim=0)
        sink_dir = sink_dir / (sink_dir.norm() + 1e-10)

        # De-sinked features
        proj_sink = (features_raw @ sink_dir.unsqueeze(-1)) * sink_dir.unsqueeze(0)
        features_desinked = features_raw - proj_sink

        # Variance stats
        total_var = features_raw.var(dim=0).sum().item()
        sink_var = proj_sink.var(dim=0).sum().item()
        sink_var_pct = sink_var / total_var * 100 if total_var > 0 else 0

        # Multiple random controls
        rand_deltas = []
        for rs in range(5):
            rd = random_direction(d, seed=100*li + rs)
            pr = (features_raw @ rd.unsqueeze(-1)) * rd.unsqueeze(0)
            feat_r = features_raw - pr
            _, _, te_r, _ = run_probes_multi_seed(feat_r, labels, n_classes, seeds=seeds)
            rand_deltas.append(te_r)

        # Probes with multi-seed
        tr_raw_m, tr_raw_s, te_raw_m, te_raw_s = run_probes_multi_seed(
            features_raw, labels, n_classes, seeds=seeds)
        tr_ds_m, tr_ds_s, te_ds_m, te_ds_s = run_probes_multi_seed(
            features_desinked, labels, n_classes, seeds=seeds)

        rand_test_mean = np.mean(rand_deltas)
        rand_test_std = np.std(rand_deltas)

        delta_desink = te_ds_m - te_raw_m
        delta_rand = rand_test_mean - te_raw_m

        print(f"  Sink var: {sink_var_pct:.1f}%")
        print(f"  Raw:       {te_raw_m:.1f}% ± {te_raw_s:.1f}%")
        print(f"  De-sinked: {te_ds_m:.1f}% ± {te_ds_s:.1f}%  (delta: {delta_desink:+.1f}%)")
        print(f"  Rand-ctrl: {rand_test_mean:.1f}% ± {rand_test_std:.1f}%  (delta: {delta_rand:+.1f}%)")

        layer_result = {
            'layer': li,
            'n_samples': len(features_raw),
            'n_classes': n_classes,
            'sink_var_pct': float(sink_var_pct),
            'raw_test_mean': float(te_raw_m), 'raw_test_std': float(te_raw_s),
            'desink_test_mean': float(te_ds_m), 'desink_test_std': float(te_ds_s),
            'rand_test_mean': float(rand_test_mean), 'rand_test_std': float(rand_test_std),
            'delta_desink': float(delta_desink),
            'delta_rand': float(delta_rand),
        }
        results['layers'].append(layer_result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (large-scale verification)")
    print(f"{'='*70}")
    print(f"Data: {n_seqs} WikiText-2 passages, {total_tokens} tokens, {len(seeds)} seeds")
    print(f"{'Layer':<8} {'sink_var%':>10} {'raw_test':>12} {'desink_test':>14} {'Δ_desink':>10} {'Δ_rand':>10}")

    for lr in results['layers']:
        print(f"L{lr['layer']:<7} {lr['sink_var_pct']:>9.1f}% "
              f"{lr['raw_test_mean']:>7.1f}±{lr['raw_test_std']:<4.1f} "
              f"{lr['desink_test_mean']:>8.1f}±{lr['desink_test_std']:<4.1f} "
              f"{lr['delta_desink']:>+9.1f}% {lr['delta_rand']:>+9.1f}%")

    with open(os.path.join(save_dir, "probing_results_large.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_dir}/probing_results_large.json")


if __name__ == "__main__":
    main()
