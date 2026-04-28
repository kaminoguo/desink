"""Figure 5(b): Per-layer departure profile (inverted-U) measured on a 6-layer model."""

import torch
import torch.nn as nn
import numpy as np
import math, os, json, time

DEVICE = "cuda"
# os.environ["HF_HOME"] = ...  # set if needed
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "exp116_output.log")

VOCAB_SIZE = 50257
MAX_SEQ_LEN = 512
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
BATCH_SIZE = 16
TOTAL_STEPS = 3000
MEASURE_EVERY = 200
MAX_LR = 3e-4

MEASURE_TEXTS = [
    "The history of artificial intelligence began in antiquity with myths",
    "In mathematics a proof is an inferential argument for a mathematical statement",
    "The global economy is interconnected through trade finance and technology",
    "Quantum mechanics is a fundamental theory in physics that describes nature",
    "Machine learning algorithms build a model based on sample data",
    "Climate change refers to long-term shifts in temperatures and weather",
    "The structure of DNA was discovered by Watson and Crick in 1953",
    "Philosophy is the study of general and fundamental questions about existence",
]


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class SimpleGPT(nn.Module):
    def __init__(self, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        d_ff = d_model * 4
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        mask = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))
        n_params = sum(p.numel() for p in self.parameters())
        log(f"  Model: {n_params/1e6:.1f}M params")

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device))
        for layer in self.layers:
            x = layer(x, self.mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, VOCAB_SIZE),
                labels[:, 1:].contiguous().view(-1)
            )
        return logits, loss

    def get_all_hiddens(self, input_ids):
        """Return hidden states after EVERY layer."""
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device))
        hiddens = []
        for layer in self.layers:
            x = layer(x, self.mask)
            hiddens.append(x.clone())
        return hiddens


def get_data_iterator(tokenizer, batch_size, seq_len):
    from datasets import load_dataset
    ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
    buffer = []
    for example in ds:
        tokens = tokenizer(example["text"], truncation=False)["input_ids"]
        buffer.extend(tokens)
        while len(buffer) >= batch_size * seq_len:
            batch = torch.tensor(buffer[:batch_size * seq_len]).reshape(batch_size, seq_len)
            buffer = buffer[batch_size * seq_len:]
            yield batch


def measure_all_layers(model, tokenizer, device):
    """Measure lm100 alignment at every layer simultaneously."""
    model.eval()
    # Collect hidden states at all layers for all texts
    all_hiddens = [[] for _ in range(model.n_layers)]

    with torch.no_grad():
        for t in MEASURE_TEXTS:
            ids = tokenizer(t, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN)["input_ids"].to(device)
            hiddens = model.get_all_hiddens(ids)
            for i, h in enumerate(hiddens):
                all_hiddens[i].append(h.float().cpu().squeeze(0))

    # SVD of lm_head (shared across all layers)
    lm_w = model.lm_head.weight.data.float().cpu()
    _, S_lm, Vt_lm = torch.linalg.svd(lm_w, full_matrices=False)

    results = {}
    for layer_idx in range(model.n_layers):
        nosink = [h[1:] for h in all_hiddens[layer_idx] if h.shape[0] > 1]
        H = torch.cat(nosink, dim=0)
        H_c = H - H.mean(0, keepdim=True)

        total_var = (H_c ** 2).sum().item()
        n_svs = min(100, len(S_lm))
        lm_var = 0.0
        for k in range(n_svs):
            proj = H_c @ Vt_lm[k]
            lm_var += (proj ** 2).sum().item()
        lm100 = lm_var / total_var if total_var > 0 else 0

        S_h = torch.linalg.svdvals(H_c)
        p_h = S_h / S_h.sum()
        p_h = p_h[p_h > 1e-10]
        ns_er = torch.exp(-torch.sum(p_h * torch.log(p_h))).item()

        results[layer_idx] = {"lm100": lm100, "nosink_er": ns_er}

    model.train()
    return results


def main():
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 70)
    log("EXP 116: PER-LAYER DEPARTURE PROFILE")
    log("Resolves exp115 confound: is departure = distance from lm_head?")
    log("=" * 70)
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"6L model, d={D_MODEL}, constant LR={MAX_LR}, {TOTAL_STEPS} steps")
    log(f"Measure alignment at ALL 6 layers every {MEASURE_EVERY} steps")
    t0 = time.time()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SimpleGPT(D_MODEL, N_LAYERS, N_HEADS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)
    data_iter = get_data_iterator(tokenizer, BATCH_SIZE, MAX_SEQ_LEN)

    all_results = []  # list of {step, loss, layer_0: {lm100, er}, layer_1: ...}
    losses = []

    # Header
    layer_cols = "  ".join([f"  L{i}" for i in range(N_LAYERS)])
    log(f"\n  {'Step':>5s}  {'Loss':>7s}  {layer_cols}   (lm100 %)")
    log(f"  {'-' * 60}")

    # Step 0
    layer_results = measure_all_layers(model, tokenizer, DEVICE)
    record = {"step": 0, "loss": float('nan')}
    for i in range(N_LAYERS):
        record[f"layer_{i}"] = layer_results[i]
    all_results.append(record)
    vals = "  ".join([f"{layer_results[i]['lm100']*100:>5.1f}" for i in range(N_LAYERS)])
    log(f"  {0:>5d}  {'N/A':>7s}  {vals}")

    for step in range(1, TOTAL_STEPS + 1):
        batch = next(data_iter).to(DEVICE)
        _, loss = model(batch, labels=batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % MEASURE_EVERY == 0:
            avg_loss = np.mean(losses[-MEASURE_EVERY:])
            layer_results = measure_all_layers(model, tokenizer, DEVICE)
            record = {"step": step, "loss": avg_loss}
            for i in range(N_LAYERS):
                record[f"layer_{i}"] = layer_results[i]
            all_results.append(record)
            vals = "  ".join([f"{layer_results[i]['lm100']*100:>5.1f}" for i in range(N_LAYERS)])
            log(f"  {step:>5d}  {avg_loss:>7.3f}  {vals}")

    # Analysis
    log(f"\n{'=' * 70}")
    log(f"PER-LAYER DEPARTURE ANALYSIS")
    log(f"{'=' * 70}")

    log(f"\n  {'Layer':>5s}  {'Dist':>4s}  {'Peak Step':>10s}  {'Peak%':>7s}  {'Final%':>7s}  {'Depart':>7s}")
    log(f"  {'-' * 50}")

    departures = []
    for i in range(N_LAYERS):
        dist_to_output = N_LAYERS - 1 - i  # layers between this and lm_head
        peak_r = max(all_results, key=lambda r: r[f"layer_{i}"]["lm100"])
        final_r = all_results[-1]
        peak_val = peak_r[f"layer_{i}"]["lm100"]
        final_val = final_r[f"layer_{i}"]["lm100"]
        dep = peak_val / final_val if final_val > 0 else 0
        departures.append(dep)
        log(f"  L{i:>4d}  {dist_to_output:>4d}  {peak_r['step']:>10d}  {peak_val*100:>6.1f}%  {final_val*100:>6.1f}%  {dep:>6.3f}x")

    # Check monotonicity with distance
    distances = list(range(N_LAYERS - 1, -1, -1))  # [5,4,3,2,1,0] for layers [0,1,2,3,4,5]
    log(f"\n  Distance from lm_head vs departure:")
    for i in range(N_LAYERS):
        log(f"    L{i} (dist={distances[i]}): departure={departures[i]:.3f}x")

    # Is it monotonic with distance?
    monotonic = all(departures[i] >= departures[i+1] for i in range(N_LAYERS - 1))
    log(f"\n  Departure monotonically decreasing L0→L5? {'YES' if monotonic else 'NO'}")

    if monotonic:
        log(f"  >>> TRIVIAL POSITIONAL EFFECT: departure = f(distance from lm_head)")
        log(f"  >>> Exp115 'inter-layer' conclusion was confounded")
    else:
        log(f"  >>> NON-TRIVIAL: departure profile is not simply distance-based")
        max_dep_layer = max(range(N_LAYERS), key=lambda i: departures[i])
        log(f"  >>> Strongest departure at L{max_dep_layer} (dist={distances[max_dep_layer]})")

    # NsER profile
    log(f"\n  Final NsER by layer:")
    for i in range(N_LAYERS):
        final_er = all_results[-1][f"layer_{i}"]["nosink_er"]
        log(f"    L{i}: NsER={final_er:.1f}")

    # Save
    json_path = os.path.join(LOG_DIR, "exp116_data.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    log(f"\nTotal: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log("Done.")


if __name__ == "__main__":
    main()
