"""Section 8: All-but-the-Top vs de-sinking comparison."""

import os, json, torch, numpy as np
device = torch.device("cuda")
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_spectral(H):
    H_bar = H - H.mean(0)
    U, S, Vt = torch.linalg.svd(H_bar, full_matrices=False)
    S2 = S**2; S2n = S2/S2.sum()
    H_normed = H_bar / (H_bar.norm(dim=1, keepdim=True) + 1e-10)
    cos_mat = H_normed @ H_normed.T
    n = cos_mat.shape[0]; mask = ~torch.eye(n, dtype=torch.bool, device=cos_mat.device)
    return {"E1": float(S2n[0]),
            "ER": float(torch.exp(-torch.sum(S2n * torch.log(S2n + 1e-10)))),
            "anisotropy": float(cos_mat[mask].mean())}

def collect(model, tokenizer, texts, li, max_t=200000):
    layers = [m for n,m in model.named_modules()
              if (hasattr(m,'self_attn') or hasattr(m,'attention')) and hasattr(m,'mlp')]
    if li >= len(layers): return None
    out = {}
    def hook(m,i,o): out["h"] = (o[0] if isinstance(o,tuple) else o).detach()
    h = layers[li].register_forward_hook(hook)
    hs = []; t = 0; model.eval()
    with torch.inference_mode():
        for text in texts:
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out.clear(); model(**inp)
            if "h" in out: x = out["h"].squeeze(0).float(); hs.append(x.cpu()); t += x.shape[0]
            if t >= max_t: break
    h.remove()
    return torch.cat(hs, dim=0)[:max_t] if hs else None

texts = [f"The quick brown fox sentence {i}. Transformer models learn complex representations. " * 10 for i in range(200)]
results = []

for model_name, step, layer_idx in [
    ("EleutherAI/pythia-70m", 143000, 3),
    ("EleutherAI/pythia-70m", 4000, 3),
    ("EleutherAI/pythia-70m", 143000, 5),
]:
    rev = f"step{step}"
    tok = AutoTokenizer.from_pretrained(model_name, revision=rev)
    mod = AutoModelForCausalLM.from_pretrained(model_name, revision=rev, torch_dtype=torch.float32).to(device)
    H = collect(mod, tok, texts, layer_idx, 200000)
    if H is None: del mod; torch.cuda.empty_cache(); continue
    H = H.to(device); H_bar = H - H.mean(0)
    U, S, Vt = torch.linalg.svd(H_bar, full_matrices=False)

    methods = {}
    methods["original"] = compute_spectral(H)

    for pc_idx, name in [(0, "desink_pc1"), (1, "remove_pc2"), (2, "remove_pc3")]:
        sv = Vt[pc_idx]
        Hp = H_bar - (H_bar @ sv).unsqueeze(1) * sv.unsqueeze(0)
        methods[name] = compute_spectral(Hp + H.mean(0))

    rr = []
    for seed in range(5):
        torch.manual_seed(seed); r = torch.randn(H.shape[1], device=device); r = r/r.norm()
        Hr = H_bar - (H_bar @ r).unsqueeze(1) * r.unsqueeze(0)
        rr.append(compute_spectral(Hr + H.mean(0)))
    methods["remove_random_mean"] = {k: float(np.mean([x[k] for x in rr])) for k in ["E1","ER","anisotropy"]}
    methods["remove_random_std"] = {k: float(np.std([x[k] for x in rr])) for k in ["E1","ER","anisotropy"]}

    results.append({"model": model_name, "step": step, "layer": layer_idx, "methods": methods})
    print(f"\n{model_name} step={step} L{layer_idx}:")
    for mn, vals in methods.items():
        if "E1" in vals: print(f"  {mn:20s}: E1={vals['E1']:.4f} ER={vals['ER']:.1f} aniso={vals['anisotropy']:.4f}")
    del mod, H, H_bar; torch.cuda.empty_cache()

os.makedirs("logs/abt_comparison", exist_ok=True)
with open("logs/abt_comparison/abt_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nABT vs de-sinking comparison done.")
