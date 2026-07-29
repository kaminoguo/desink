"""Collapse the per-checkpoint result JSONs into paper-ready artifacts.

  t1_rerun_long.csv        one row per (exp, model, step, block, layer), scalars only
  t1_rerun_summary.json    mean/std over the 3 seeds, per (model, step, layer)
  t1_rerun_eigs.npz        top-20 eigenvalues, keyed "<exp>|<model>|<step>|<block>|<layer>"
and prints the headline trajectory: raw vs de-sinked ER at the peak-sink layer.
"""
import os, json, glob, csv
import numpy as np

RES = "/workspace/desink/results"
OUT = "/workspace/desink/agg"
SCALARS = ["E1_raw", "E1_ds", "ER_raw", "ER_ds", "RankMe_raw", "RankMe_ds",
           "aniso_raw", "aniso_ds", "aniso_uncentered_raw", "aniso_uncentered_ds",
           "gap_ratio", "sink_alpha", "p0_mass", "mu_norm", "mean_token_norm",
           "proj_check", "n_tokens", "d"]


def load():
    rows, eigs = [], {}
    for p in sorted(glob.glob(f"{RES}/e[123]_*.json")):
        d = json.load(open(p))
        exp = os.path.basename(p).split("_")[0]
        model = d["model"].split("/")[-1]
        for block, recs in d["blocks"].items():
            for r in recs:
                rows.append(dict(exp=exp, model=model, step=d["step"],
                                 revision=d["revision"], block=block,
                                 layer=r["layer"], n_layers=d["n_layers"],
                                 probe_loss=d["probe_loss"],
                                 **{k: r[k] for k in SCALARS}))
                eigs[f"{exp}|{model}|{d['step']}|{block}|{r['layer']}"] = \
                    np.asarray(r["top20_eigenvalues"])
    return rows, eigs


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, eigs = load()
    if not rows:
        raise RuntimeError("no results found")
    keys = list(rows[0])
    with open(f"{OUT}/t1_rerun_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(f"{OUT}/t1_rerun_eigs.npz", **eigs)

    # mean/std over seed blocks (E1/E2 only; E3 blocks are different n, not seeds)
    summ = {}
    for r in rows:
        if not r["block"].startswith("seed"):
            continue
        k = f"{r['exp']}|{r['model']}|{r['step']}|{r['layer']}"
        summ.setdefault(k, []).append(r)
    out = {}
    for k, rs in summ.items():
        out[k] = {"n_seeds": len(rs), "probe_loss": rs[0]["probe_loss"]}
        for c in SCALARS:
            v = np.array([x[c] for x in rs], dtype=float)
            out[k][c] = [float(v.mean()), float(v.std())]
    json.dump(out, open(f"{OUT}/t1_rerun_summary.json", "w"))

    # headline: at each model's peak-sink layer, raw vs de-sinked ER over training
    print(f"\nwrote {len(rows)} rows, {len(eigs)} eigenspectra, {len(out)} summary cells\n")
    for model in sorted({r["model"] for r in rows if r["exp"] in ("e1", "e2")}):
        sub = [r for r in rows if r["model"] == model and r["exp"] in ("e1", "e2")
               and r["block"].startswith("seed")]
        if not sub:
            continue
        last = max(r["step"] for r in sub)
        fin = [r for r in sub if r["step"] == last]
        peak = max(fin, key=lambda r: r["sink_alpha"])["layer"]
        print(f"=== {model}  peak-sink layer L{peak} (of {fin[0]['n_layers']}) ===")
        print(f"{'step':>7} {'loss':>6} {'ER_raw':>8} {'ER_ds':>8} {'E1_raw':>7} "
              f"{'E1_ds':>7} {'RkM_raw':>8} {'RkM_ds':>8} {'ani_u_raw':>9} "
              f"{'ani_u_ds':>9} {'sink':>6} {'p0mass':>7}")
        for st in sorted({r["step"] for r in sub}):
            g = [r for r in sub if r["step"] == st and r["layer"] == peak]
            if not g:
                continue
            m = lambda c: np.mean([x[c] for x in g])
            print(f"{st:7d} {g[0]['probe_loss']:6.2f} {m('ER_raw'):8.1f} {m('ER_ds'):8.1f} "
                  f"{m('E1_raw'):7.3f} {m('E1_ds'):7.3f} {m('RankMe_raw'):8.1f} "
                  f"{m('RankMe_ds'):8.1f} {m('aniso_uncentered_raw'):9.4f} "
                  f"{m('aniso_uncentered_ds'):9.4f} {m('sink_alpha'):6.2f} {m('p0_mass'):7.4f}")
        print()


if __name__ == "__main__":
    main()
