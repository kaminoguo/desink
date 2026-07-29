"""One (model, checkpoint) -> spectral metrics for every layer x every seed.

Three streaming passes over the shared corpus (no hidden-state caching, so
memory is O(d^2) not O(n*d)):

  pass 1  mean mu, per-position-0 / non-position-0 norm sums          O(n d)
  pass 2  centred covariance C = (1/n) sum (h-mu)(h-mu)^T             O(n d^2)
          -> eigendecomposition gives lambda_1..lambda_d and s
  pass 3  anisotropy sums (needs mu) and de-sinked anisotropy sums
          (needs s), plus the position-0 share of the sink-direction
          energy                                                      O(n d)

De-sinking is the rank-1 projection P = I - s s^T with s the top eigenvector of
the centred covariance.  Because C s = lambda_1 s exactly, P C P = C - lambda_1 s s^T,
so the de-sinked spectrum is exactly lambda_2..lambda_d — no second SVD needed.
Anisotropy is NOT a spectral functional (row-wise normalisation), so it does
need the explicit projection in pass 3.

fp32 model weights and activations, TF32 disabled: a TF32 matmul has a 10-bit
mantissa and would corrupt the small eigenvalues that ER / RankMe depend on.
Accumulators are fp64.

Self-check: sum_i <h_i - mu, s>^2 must equal n * lambda_1.  `proj_check` is that
ratio and is written into every record; anything off 1.0 means the covariance
and the projection disagree.
"""
import os, sys, json, time, argparse, traceback

os.environ.setdefault("HF_HOME", "/workspace/desink/hf")

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SEQ_LEN = 512
EPS_RANKME = 1e-7


# ──────────────────────────────────────────────────────────────────────────────
def forward_hidden(backbone, ids):
    with torch.no_grad():
        out = backbone(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states                      # tuple len = n_layers+1, [B,S,d]


def iter_batches(corpus, rows, batch_size, device):
    for i in range(0, len(rows), batch_size):
        yield torch.from_numpy(corpus[rows[i:i + batch_size]].astype(np.int64)).to(device)


def analyze_block(backbone, corpus, rows, device, batch_size, L, d):
    """rows: 1-D np index array of corpus sequences.  Returns list of L metric dicts."""
    n = len(rows) * SEQ_LEN
    f64 = dict(dtype=torch.float64, device=device)

    # ── pass 1 ───────────────────────────────────────────────────────────────
    sum_h = torch.zeros(L, d, **f64)
    sum_np0 = torch.zeros(L, **f64)
    sum_nrest = torch.zeros(L, **f64)
    for ids in iter_batches(corpus, rows, batch_size, device):
        for li, h in enumerate(forward_hidden(backbone, ids)):
            hf = h.float()
            sum_h[li] += hf.reshape(-1, d).sum(0).double()
            nrm = hf.norm(dim=-1)                              # [B,S]
            sum_np0[li] += nrm[:, 0].sum().double()
            sum_nrest[li] += nrm[:, 1:].sum().double()
    mu64 = sum_h / n
    mu = mu64.float()                                          # [L,d]
    cnt_p0 = len(rows)
    cnt_rest = n - cnt_p0

    # ── pass 2 ───────────────────────────────────────────────────────────────
    C = torch.zeros(L, d, d, **f64)
    for ids in iter_batches(corpus, rows, batch_size, device):
        for li, h in enumerate(forward_hidden(backbone, ids)):
            hb = h.float().reshape(-1, d) - mu[li]
            C[li] += (hb.T @ hb).double()
    C /= n

    lam = torch.zeros(L, d, **f64)
    svec = torch.zeros(L, d, dtype=torch.float32, device=device)
    for li in range(L):
        ev, evec = torch.linalg.eigh(C[li])                    # ascending
        lam[li] = ev.flip(0).clamp_min(0.0)
        svec[li] = evec[:, -1].float()
    del C

    # ── pass 3 ───────────────────────────────────────────────────────────────
    sum_u = torch.zeros(L, d, **f64)
    sum_uds = torch.zeros(L, d, **f64)
    sum_u_unc = torch.zeros(L, d, **f64)      # uncentred = the literature's
    sum_uds_unc = torch.zeros(L, d, **f64)    # anisotropy (Ethayarajh / RepDeg)
    p0_sq = torch.zeros(L, **f64)
    tot_sq = torch.zeros(L, **f64)
    unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    for ids in iter_batches(corpus, rows, batch_size, device):
        for li, h in enumerate(forward_hidden(backbone, ids)):
            s = svec[li]
            hf = h.float()                                     # [B,S,d]
            hb = hf - mu[li]
            proj = hb @ s                                      # [B,S]
            p0_sq[li] += (proj[:, 0] ** 2).sum().double()
            tot_sq[li] += (proj ** 2).sum().double()
            sum_u[li] += unit(hb).reshape(-1, d).sum(0).double()
            hds = hb - proj.unsqueeze(-1) * s
            sum_uds[li] += unit(hds).reshape(-1, d).sum(0).double()
            sum_u_unc[li] += unit(hf).reshape(-1, d).sum(0).double()
            hds_unc = hf - (hf @ s).unsqueeze(-1) * s
            sum_uds_unc[li] += unit(hds_unc).reshape(-1, d).sum(0).double()

    # ── metrics ──────────────────────────────────────────────────────────────
    out = []
    for li in range(L):
        lm = lam[li]
        tot = lm.sum()
        lm_ds = lm[1:]
        tot_ds = lm_ds.sum()

        def ent_exp(v):
            p = v / v.sum()
            p = p[p > 0]
            return float(torch.exp(-(p * p.log()).sum()))

        def rankme(v):
            sig = v.clamp_min(0).sqrt()
            q = sig / sig.sum() + EPS_RANKME
            return float(torch.exp(-(q * q.log()).sum()))

        aniso = lambda su: float((su.dot(su) - n) / (n * (n - 1.0)))

        out.append({
            "layer": li,
            "n_tokens": int(n), "d": int(d),
            "E1_raw": float(lm[0] / tot),
            "E1_ds": float(lm[1] / tot_ds),
            "ER_raw": ent_exp(lm), "ER_ds": ent_exp(lm_ds),
            "RankMe_raw": rankme(lm), "RankMe_ds": rankme(lm_ds),
            "aniso_raw": aniso(sum_u[li]), "aniso_ds": aniso(sum_uds[li]),
            "aniso_uncentered_raw": aniso(sum_u_unc[li]),
            "aniso_uncentered_ds": aniso(sum_uds_unc[li]),
            "gap_ratio": float((lm[0] / lm[1]).sqrt()),
            "sink_alpha": float((sum_np0[li] / cnt_p0) / (sum_nrest[li] / cnt_rest)),
            "p0_mass": float(p0_sq[li] / tot_sq[li]),
            "top20_eigenvalues": [float(x) for x in lm[:20]],
            "mu_norm": float(mu64[li].norm()),
            "mean_token_norm": float((sum_np0[li] + sum_nrest[li]) / n),
            "proj_check": float(tot_sq[li] / (n * lm[0])),
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="/workspace/desink/corpus/corpus.npy")
    ap.add_argument("--mode", default="e1", choices=["e1", "e3"])
    ap.add_argument("--batch-tokens", type=int, default=8192)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    device = torch.device("cuda")
    corpus = np.load(args.corpus)
    meta = json.load(open(args.corpus.replace(".npy", "_meta.json")))
    nsps, nseeds = meta["n_seq_per_seed"], meta["n_seeds"]
    assert corpus.shape[1] == SEQ_LEN

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32).to(device).eval()
    backbone = model.base_model
    assert backbone is not model, "base_model resolved to the LM head wrapper"
    d = model.config.hidden_size
    L = model.config.num_hidden_layers + 1
    batch_size = max(1, args.batch_tokens // SEQ_LEN)
    print(f"[{args.revision}] loaded d={d} L={L} bs={batch_size} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # held-out loss probe: a checkpoint that failed to differentiate shows up here
    probe = torch.from_numpy(corpus[nsps * nseeds:].astype(np.int64)).to(device)
    with torch.no_grad():
        loss = float(model(input_ids=probe, labels=probe).loss)
    print(f"[{args.revision}] probe_loss={loss:.4f}", flush=True)

    if args.mode == "e1":
        blocks = {f"seed{k}": np.arange(k * nsps, (k + 1) * nsps) for k in range(nseeds)}
    else:                                   # E3: nested prefixes, 8K..512K tokens
        blocks = {f"n{t}": np.arange(t // SEQ_LEN) for t in (8192, 32768, 131072, 524288)}

    res = {"model": args.model, "revision": args.revision, "step": args.step,
           "mode": args.mode, "d": d, "n_layers": model.config.num_hidden_layers,
           "probe_loss": loss, "corpus_sha256_16": meta["sha256_16"],
           "dtype": "float32", "tf32": False, "blocks": {}}

    for name, rows in blocks.items():
        tb = time.time()
        res["blocks"][name] = analyze_block(backbone, corpus, rows, device,
                                            batch_size, L, d)
        bad = [r["layer"] for r in res["blocks"][name] if abs(r["proj_check"] - 1) > 1e-3]
        m = res["blocks"][name]
        print(f"[{args.revision}] {name} {time.time()-tb:.0f}s  "
              f"L0 ER={m[0]['ER_raw']:.1f}/{m[0]['ER_ds']:.1f}  "
              f"Lmid ER={m[L//2]['ER_raw']:.1f}/{m[L//2]['ER_ds']:.1f} "
              f"sink={m[L//2]['sink_alpha']:.2f}x  projfail={bad}", flush=True)
        if bad:
            raise RuntimeError(f"proj_check failed on layers {bad}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"[{args.revision}] DONE {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
