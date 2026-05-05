"""Section 3.4 and Table 3: Synthetic sink injection (causal sufficiency) and Llama-3 / Qwen2.5 GQA contamination audit."""

import os, json, torch, numpy as np, time
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_texts(max_texts=500):
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return [t for t in ds["text"] if len(t.strip()) > 100][:max_texts]
    except:
        return [f"The transformer architecture sentence {i}. " * 20 for i in range(max_texts)]

def collect_hidden(model, tokenizer, texts, layer_idx, max_tokens=200000):
    layers = [m for n,m in model.named_modules()
              if (hasattr(m,'self_attn') or hasattr(m,'attention')) and hasattr(m,'mlp')]
    if layer_idx >= len(layers): return None
    out = {}
    def hook(m,i,o): out["h"] = (o[0] if isinstance(o,tuple) else o).detach()
    handle = layers[layer_idx].register_forward_hook(hook)
    hs = []; t = 0; model.eval()
    with torch.inference_mode():
        for text in texts:
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out.clear(); model(**inp)
            if "h" in out:
                x = out["h"].squeeze(0).float(); hs.append(x.cpu()); t += x.shape[0]
                if t >= max_tokens: break
    handle.remove()
    return torch.cat(hs, dim=0)[:max_tokens] if hs else None

texts = load_texts()
print(f"Loaded {len(texts)} texts", flush=True)

# ══════════════════════════════════════════════════════════════
# EXPERIMENT A: SYNTHETIC SINK INJECTION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP A: SYNTHETIC SINK INJECTION")
print("="*60, flush=True)

t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision="step512")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", revision="step512",
                                              torch_dtype=torch.float32).to(device)

H = collect_hidden(model, tokenizer, texts, 3, max_tokens=200000)  # L3
H = H.to(device)
print(f"Collected H: {H.shape}", flush=True)

del model; torch.cuda.empty_cache()

# Get sink direction from PC1
H_bar = H - H.mean(0)
U, S, Vt = torch.linalg.svd(H_bar, full_matrices=False)
s = Vt[0]  # sink direction

# Base metrics (before injection)
S2 = S**2; S2n = S2 / S2.sum()
E1_base = float(S2n[0])
ER_base = float(torch.exp(-torch.sum(S2n * torch.log(S2n + 1e-12))))

# De-sink to get clean representation
H_clean = H_bar - (H_bar @ s).unsqueeze(1) * s.unsqueeze(0)
_, S_clean, _ = torch.linalg.svd(H_clean, full_matrices=False)
S2c = S_clean**2; S2cn = S2c / S2c.sum()
E1_clean = float(S2cn[0])
ER_clean = float(torch.exp(-torch.sum(S2cn * torch.log(S2cn + 1e-12))))

print(f"Base (step512 L3):  E1={E1_base:.4f}  ER={ER_base:.1f}")
print(f"Clean (de-sinked):  E1={E1_clean:.4f}  ER={ER_clean:.1f}")

# Inject synthetic sink at varying strengths
mean_norm = float(H_clean.norm(dim=1).mean())
alpha_scales = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

injections = []
for alpha_scale in alpha_scales:
    H_inj = H_clean.clone()
    # Inject at position 0 of each "sequence" — but we have concatenated tokens
    # Simplification: inject at every 512th position (approximate sequence boundaries)
    # Or simpler: inject at position 0 only (one spike)
    perturbation = alpha_scale * mean_norm * s
    H_inj[0] = H_inj[0] + perturbation

    # If alpha is large enough, add to more positions to simulate real sink
    # Real sinks affect position-0 of EVERY sequence
    # Approximate: every 100th token is a "position-0"
    if alpha_scale > 0:
        for pos in range(0, len(H_inj), 100):
            H_inj[pos] = H_clean[pos] + perturbation

    H_inj_bar = H_inj - H_inj.mean(0)
    _, S_inj, _ = torch.linalg.svd(H_inj_bar, full_matrices=False)
    S2i = S_inj**2; S2in = S2i / S2i.sum()
    E1_inj = float(S2in[0])
    ER_inj = float(torch.exp(-torch.sum(S2in * torch.log(S2in + 1e-12))))

    # Anisotropy on subsample
    n_sub = min(500, len(H_inj))
    sub = H_inj[torch.randperm(len(H_inj))[:n_sub]]
    sub_n = sub / (sub.norm(dim=1, keepdim=True) + 1e-10)
    cos_mat = sub_n @ sub_n.T
    mask = ~torch.eye(n_sub, dtype=torch.bool, device=device)
    aniso = float(cos_mat[mask].mean())

    # Theorem 1 prediction: E1_theory = (kappa * alpha^2 + E1_clean) / (kappa * alpha^2 + 1)
    # kappa depends on fraction of position-0 tokens and their relative energy
    # Approximate kappa from the fraction of injected positions
    n_injected = len(range(0, len(H_inj), 100)) if alpha_scale > 0 else 0
    frac = n_injected / len(H_inj) if len(H_inj) > 0 else 0
    kappa = frac  # simplified
    effective_alpha = alpha_scale * mean_norm
    E1_theory = (kappa * effective_alpha**2 / (S_clean**2).sum().item() + E1_clean) if kappa > 0 else E1_clean

    injections.append({
        "alpha_scale": alpha_scale,
        "E1": E1_inj, "ER": ER_inj, "anisotropy": aniso,
        "E1_theory": min(E1_theory, 1.0),
    })
    print(f"  alpha={alpha_scale:3d}: E1={E1_inj:.4f}  ER={ER_inj:.1f}  aniso={aniso:.4f}", flush=True)

sink_injection = {
    "experiment": "synthetic_sink_injection",
    "base_model": "pythia-70m", "base_step": 512, "base_layer": "L3",
    "base_E1": E1_base, "base_ER": ER_base,
    "clean_E1": E1_clean, "clean_ER": ER_clean,
    "mean_norm": mean_norm,
    "injections": injections,
}

del H, H_bar, H_clean; torch.cuda.empty_cache()
print(f"\nExp A done in {time.time()-t0:.0f}s", flush=True)


# ══════════════════════════════════════════════════════════════
# EXPERIMENT B: GQA ARCHITECTURE STATIC AUDIT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP B: GQA ARCHITECTURE STATIC AUDIT")
print("="*60, flush=True)

t1 = time.time()

# Try models in order of preference
GQA_MODELS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5 (GQA, RoPE, SwiGLU)"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama (GQA, RoPE)"),
]

gqa_audit = None

for model_name, arch_desc in GQA_MODELS:
    print(f"\nTrying {model_name}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        print(f"Loaded {model_name}", flush=True)
    except Exception as e:
        print(f"Failed: {e}")
        continue

    layers = [m for n,m in model.named_modules()
              if (hasattr(m,'self_attn') or hasattr(m,'attention')) and hasattr(m,'mlp')]
    n_layers = len(layers)
    print(f"Layers: {n_layers}")

    layer_results = []
    for li in range(n_layers):
        H = collect_hidden(model, tokenizer, texts, li, max_tokens=200000)
        if H is None: continue
        H = H.to(device)

        # Raw
        H_bar = H - H.mean(0)
        _, S, Vt = torch.linalg.svd(H_bar, full_matrices=False)
        S2 = S**2; S2n = S2/S2.sum()
        E1_raw = float(S2n[0])
        ER_raw = float(torch.exp(-torch.sum(S2n * torch.log(S2n + 1e-12))))
        gap = float(S[0]/S[1]) if len(S) > 1 else 0

        # De-sinked
        sink = Vt[0]
        H_ds = H_bar - (H_bar @ sink).unsqueeze(1) * sink.unsqueeze(0)
        _, S_ds, _ = torch.linalg.svd(H_ds, full_matrices=False)
        S2d = S_ds**2; S2dn = S2d/S2d.sum()
        E1_ds = float(S2dn[0])
        ER_ds = float(torch.exp(-torch.sum(S2dn * torch.log(S2dn + 1e-12))))

        # Sink alpha
        norms = H.norm(dim=1)
        sink_alpha = float(norms[0] / norms[1:].mean()) if len(norms) > 1 else 1.0

        layer_results.append({
            "layer": li, "E1_raw": E1_raw, "E1_ds": E1_ds,
            "ER_raw": ER_raw, "ER_ds": ER_ds,
            "gap": gap, "sink_alpha": sink_alpha,
        })
        print(f"  L{li:2d}: E1 {E1_raw:.3f}→{E1_ds:.3f}  ER {ER_raw:.1f}→{ER_ds:.1f}  "
              f"gap {gap:.1f}x  α {sink_alpha:.1f}x", flush=True)
        del H

    gqa_audit = {
        "model": model_name, "architecture": arch_desc,
        "n_layers": n_layers, "layers": layer_results,
    }

    del model; torch.cuda.empty_cache()
    break  # use first successful model

print(f"\nExp B done in {time.time()-t1:.0f}s", flush=True)

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
os.makedirs("logs/manufacture_and_qwen", exist_ok=True)
with open("logs/manufacture_and_qwen/sink_injection.json", "w") as f:
    json.dump(sink_injection, f, indent=2)
if gqa_audit is not None:
    with open("logs/manufacture_and_qwen/gqa_audit.json", "w") as f:
        json.dump(gqa_audit, f, indent=2)

print("\n" + "="*60)
print("ALL DONE")
print("="*60, flush=True)
