"""Section 8: Eigenvalue spectral gap analysis (justifies removing exactly one direction)."""

import os, json, torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_gap(H):
    H_bar = H - H.mean(0)
    U, S, Vt = torch.linalg.svd(H_bar, full_matrices=False)
    S2 = S**2; S2n = S2/S2.sum()
    return {"E1_raw": float(S2n[0]), "top10_sv": S[:10].tolist(),
            "gap_ratio": float(S[0]/S[1]) if len(S)>1 else 0}

def collect(model, tokenizer, texts, li, max_t=200000):
    layers = [m for n,m in model.named_modules()
              if (hasattr(m,'self_attn') or hasattr(m,'attention')) and hasattr(m,'mlp')]
    if li >= len(layers): return None
    out = {}
    def hook(m,i,o): out["h"] = (o[0] if isinstance(o,tuple) else o).detach()
    h = layers[li].register_forward_hook(hook)
    hs = []; t = 0
    model.eval()
    with torch.inference_mode():
        for text in texts:
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out.clear(); model(**inp)
            if "h" in out:
                x = out["h"].squeeze(0).float(); hs.append(x.cpu()); t += x.shape[0]
                if t >= max_t: break
    h.remove()
    return torch.cat(hs, dim=0)[:max_t] if hs else None

def load_texts(max_texts=500):
    """Natural-text corpus. Raises on failure; never substitutes synthetic text."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:max_texts]
    if len(texts) < max_texts // 2:
        raise RuntimeError("corpus load returned %d texts; refusing to proceed" % len(texts))
    print("loaded %d WikiText-2 passages" % len(texts))
    return texts


texts = load_texts()

results = []
for mn in ["EleutherAI/pythia-70m", "openai-community/gpt2"]:
    tok = AutoTokenizer.from_pretrained(mn)
    mod = AutoModelForCausalLM.from_pretrained(mn, torch_dtype=torch.float32).to(device)
    layers = [m for n,m in mod.named_modules()
              if (hasattr(m,'self_attn') or hasattr(m,'attention')) and hasattr(m,'mlp')]
    md = {"model": mn, "n_layers": len(layers), "layers": []}
    for li in range(len(layers)):
        H = collect(mod, tok, texts, li, 200000)
        if H is None: continue
        H = H.to(device); m = compute_gap(H)
        md["layers"].append({"layer": li, **m})
        print(f"{mn} L{li}: gap={m['gap_ratio']:.2f}x E1={m['E1_raw']:.3f}", flush=True)
        del H
    results.append(md); del mod; torch.cuda.empty_cache()

os.makedirs("logs/spectral_gap", exist_ok=True)
with open("logs/spectral_gap/eigenvalue_gap.json", "w") as f:
    json.dump(results, f, indent=2)
print("Spectral gap analysis done.")
