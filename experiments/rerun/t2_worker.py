"""T2 rerun: is the sink direction special, or is de-sinking just "remove any one
direction"?  (exp_t1_all.py's T2-1 / T2-2 read the same poisoned `texts` variable
as T1, so that data is void as well.)

For every layer of one checkpoint, compute the spectrum and anisotropy after
removing each of:
    pc1   the top eigenvector of the centred covariance  (= de-sinking = ABTT-1)
    pc2, pc3
    rand0..rand4   fixed random unit directions
    (and the untouched baseline)

Removing an eigenvector v_k of C is exact in the spectrum: P C P = C - lambda_k v_k v_k^T,
so the resulting eigenvalues are lambda with lambda_k deleted.  Removing a random r is not,
so P C P is formed explicitly and re-diagonalised; its zero eigenvalue is dropped
so every variant is compared at rank d-1.

Anisotropy is not a spectral functional, so pass 3 accumulates a unit-vector sum
per variant (centred and uncentred).
"""
import os, sys, json, time, argparse, traceback

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SEQ_LEN, EPS_RANKME, N_RAND = 512, 1e-7, 5
VARIANTS = ["baseline", "pc1", "pc2", "pc3"] + [f"rand{i}" for i in range(N_RAND)]


def fwd(backbone, ids):
    with torch.no_grad():
        return backbone(input_ids=ids, output_hidden_states=True,
                        use_cache=False).hidden_states


def batches(corpus, rows, bs, device):
    for i in range(0, len(rows), bs):
        yield torch.from_numpy(corpus[rows[i:i + bs]].astype(np.int64)).to(device)


def spec(lam):
    lam = lam.clamp_min(0)
    tot = lam.sum()
    p = lam / tot
    p = p[p > 0]
    sig = lam.sqrt()
    q = sig / sig.sum() + EPS_RANKME
    return {"E1": float(lam[0] / tot),
            "ER": float(torch.exp(-(p * p.log()).sum())),
            "RankMe": float(torch.exp(-(q * q.log()).sum())),
            "gap_ratio": float((lam[0] / lam[1]).sqrt())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="/workspace/desink/corpus/corpus.npy")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    device = torch.device("cuda")
    corpus = np.load(args.corpus)
    meta = json.load(open(args.corpus.replace(".npy", "_meta.json")))
    rows = np.arange(meta["n_seq_per_seed"])          # seed 0 only — T2 is a
    n = len(rows) * SEQ_LEN                            # per-layer comparison, not a curve

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32).to(device).eval()
    backbone = model.base_model
    d = model.config.hidden_size
    L = model.config.num_hidden_layers + 1
    bs = max(1, args.batch_tokens // SEQ_LEN)
    f64 = dict(dtype=torch.float64, device=device)
    print(f"[t2 {args.model}@{args.revision}] d={d} L={L} n={n}", flush=True)

    t0 = time.time()
    sum_h = torch.zeros(L, d, **f64)
    for ids in batches(corpus, rows, bs, device):
        for li, h in enumerate(fwd(backbone, ids)):
            sum_h[li] += h.float().reshape(-1, d).sum(0).double()
    mu = (sum_h / n).float()

    C = torch.zeros(L, d, d, **f64)
    for ids in batches(corpus, rows, bs, device):
        for li, h in enumerate(fwd(backbone, ids)):
            hb = h.float().reshape(-1, d) - mu[li]
            C[li] += (hb.T @ hb).double()
    C /= n
    print(f"[t2] covariance done {time.time()-t0:.0f}s", flush=True)

    # directions + spectra per layer
    dirs = torch.zeros(L, len(VARIANTS), d, dtype=torch.float32, device=device)
    spectra = [{} for _ in range(L)]
    g = torch.Generator(device="cpu").manual_seed(1234)
    rand_dirs = torch.nn.functional.normalize(torch.randn(N_RAND, d, generator=g), dim=1).to(device)
    for li in range(L):
        ev, evec = torch.linalg.eigh(C[li])
        lam = ev.flip(0).clamp_min(0)
        V = evec.flip(1)                               # columns, descending
        spectra[li]["baseline"] = spec(lam)
        for k in (1, 2, 3):
            spectra[li][f"pc{k}"] = spec(torch.cat([lam[:k - 1], lam[k:]]))
            dirs[li, k] = V[:, k - 1].float()
        for j in range(N_RAND):
            r = rand_dirs[j].double()
            P = torch.eye(d, **f64) - torch.outer(r, r)
            evj = torch.linalg.eigvalsh(P @ C[li] @ P).flip(0).clamp_min(0)
            spectra[li][f"rand{j}"] = spec(evj[:-1])   # drop the induced zero
            dirs[li, 4 + j] = rand_dirs[j]
    del C
    print(f"[t2] spectra done {time.time()-t0:.0f}s", flush=True)

    # pass 3: anisotropy per variant
    V_ = len(VARIANTS)
    su = torch.zeros(L, V_, d, **f64)
    su_unc = torch.zeros(L, V_, d, **f64)
    unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    for ids in batches(corpus, rows, bs, device):
        for li, h in enumerate(fwd(backbone, ids)):
            hf = h.float().reshape(-1, d)
            hb = hf - mu[li]
            for vi in range(V_):
                if vi == 0:
                    a, b = hb, hf
                else:
                    v = dirs[li, vi]
                    a = hb - torch.outer(hb @ v, v)
                    b = hf - torch.outer(hf @ v, v)
                su[li, vi] += unit(a).sum(0).double()
                su_unc[li, vi] += unit(b).sum(0).double()

    aniso = lambda s: float((s.dot(s) - n) / (n * (n - 1.0)))
    res = {"model": args.model, "revision": args.revision, "step": args.step,
           "d": d, "n_layers": model.config.num_hidden_layers, "n_tokens": n,
           "corpus_sha256_16": meta["sha256_16"], "variants": VARIANTS, "layers": []}
    for li in range(L):
        rec = {"layer": li}
        for vi, name in enumerate(VARIANTS):
            rec[name] = dict(spectra[li][name],
                             aniso=aniso(su[li, vi]),
                             aniso_uncentered=aniso(su_unc[li, vi]))
        res["layers"].append(rec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"))
    mid = L // 2
    b, p1 = res["layers"][mid]["baseline"], res["layers"][mid]["pc1"]
    rm = np.mean([res["layers"][mid][f"rand{j}"]["ER"] for j in range(N_RAND)])
    print(f"[t2] L{mid}: ER base={b['ER']:.1f} pc1={p1['ER']:.1f} "
          f"pc2={res['layers'][mid]['pc2']['ER']:.1f} rand={rm:.1f}", flush=True)
    print(f"[t2] DONE {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
