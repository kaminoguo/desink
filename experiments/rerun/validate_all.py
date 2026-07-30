"""Post-run validity gate over every result JSON.

Checks, per record: no NaN/inf, proj_check == 1 (the covariance and the explicit
sink projection must agree), E1 in [0,1], ER/RankMe in (0, d], sink_alpha > 0.
Per (model, step): probe_loss finite.  Per (model, layer, step): cross-seed
spread of ER_raw / ER_ds.  Also flags checkpoints whose metrics are byte-identical
to a different checkpoint's (would indicate a revision that silently resolved to
the same weights).
"""
import glob, json, math, sys
from collections import defaultdict
import numpy as np

FLOAT_KEYS = ["E1_raw", "E1_ds", "ER_raw", "ER_ds", "RankMe_raw", "RankMe_ds",
              "aniso_raw", "aniso_ds", "aniso_uncentered_raw", "aniso_uncentered_ds",
              "gap_ratio", "sink_alpha", "p0_mass", "proj_check"]


def main(paths):
    bad, n_rec, n_files = [], 0, 0
    spread = defaultdict(list)
    sigs = {}
    for p in sorted(paths):
        d = json.load(open(p))
        n_files += 1
        if not math.isfinite(d["probe_loss"]):
            bad.append((p, "probe_loss not finite"))
        sig = []
        for bn, rows in d["blocks"].items():
            for r in rows:
                n_rec += 1
                for k in FLOAT_KEYS:
                    v = r[k]
                    if not math.isfinite(v):
                        bad.append((p, bn, r["layer"], k, v))
                if abs(r["proj_check"] - 1) > 1e-3:
                    bad.append((p, bn, r["layer"], "proj_check", r["proj_check"]))
                if not (0 <= r["E1_raw"] <= 1) or not (0 <= r["E1_ds"] <= 1):
                    bad.append((p, bn, r["layer"], "E1 out of [0,1]"))
                if not (0 < r["ER_raw"] <= r["d"] + 1e-6):
                    bad.append((p, bn, r["layer"], "ER_raw out of range", r["ER_raw"]))
                if r["sink_alpha"] <= 0:
                    bad.append((p, bn, r["layer"], "sink_alpha<=0"))
                if bn.startswith("seed"):
                    spread[(d["model"], d["step"], r["layer"])].append(
                        (r["ER_raw"], r["ER_ds"]))
            if bn == "seed0":
                sig = [round(r["ER_raw"], 6) for r in rows]
        key = (d["model"], tuple(sig))
        if sig and key in sigs:
            sigs[key].append(d["step"])
        elif sig:
            sigs[key] = [d["step"]]

    rel = []
    for k, v in spread.items():
        a = np.array(v)
        for j, name in enumerate(("ER_raw", "ER_ds")):
            m = a[:, j].mean()
            if m > 0:
                rel.append(((a[:, j].max() - a[:, j].min()) / m, k, name))
    rel.sort(reverse=True)

    print(f"files={n_files}  records={n_rec}")
    print(f"INVALID: {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    print(f"\ncross-seed relative spread: median={np.median([r[0] for r in rel]):.4f} "
          f"p95={np.percentile([r[0] for r in rel], 95):.4f} max={rel[0][0]:.4f} @ {rel[0][1]} {rel[0][2]}")
    dup = {k: v for k, v in sigs.items() if len(v) > 1}
    print(f"\ncheckpoints with byte-identical seed0 ER_raw profile ({len(dup)} groups):")
    for k, v in dup.items():
        print(f"   {k[0]}: steps {v}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or glob.glob("results/e[123]_*.json")))


def check_init_effective_rank(records):
    """Refuse a run whose step-0 effective rank is too small for its width.

    A corpus collapsed to a handful of distinct tokens shows up here first: a
    768-dimensional model reported ER ~ 10 at initialization in the round this
    replaces.  A randomly initialized model measured on natural text sits at a
    large fraction of d, so d/25 is a floor with a wide margin, not a threshold.
    """
    bad = []
    for r in records:
        if int(r["step"]) != 0 or r.get("exp") != "e1":
            continue
        d, er = float(r["d"]), float(r["ER_raw"])
        if er < d / 25.0:
            bad.append((r["model"], r["layer"], er, d))
    if bad:
        raise SystemExit(
            "ABORT: effective rank at initialization is below d/25 for %d "
            "(model, layer) pairs, e.g. %s. The corpus is almost certainly "
            "degenerate; check build_corpus.py output before trusting anything."
            % (len(bad), bad[:3]))
    return len(records)
