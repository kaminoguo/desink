"""Dump one result JSON as a per-layer table + validity checks (NaN/inf, ranges,
proj_check, cross-seed spread).  Used as the post-run sanity gate."""
import json, math, sys
import numpy as np

COLS = ["E1_raw", "E1_ds", "ER_raw", "ER_ds", "RankMe_raw", "RankMe_ds",
        "aniso_raw", "aniso_ds", "aniso_uncentered_raw", "aniso_uncentered_ds",
        "gap_ratio", "sink_alpha", "p0_mass", "proj_check"]
FMT = {"E1_raw": "7.4f", "E1_ds": "7.4f", "ER_raw": "7.1f", "ER_ds": "7.1f",
       "RankMe_raw": "8.1f", "RankMe_ds": "8.1f", "aniso_raw": "8.4f",
       "aniso_ds": "8.4f", "aniso_uncentered_raw": "8.4f",
       "aniso_uncentered_ds": "8.4f", "gap_ratio": "8.2f", "sink_alpha": "8.2f",
       "p0_mass": "8.4f", "proj_check": "8.4f"}


def main(path):
    d = json.load(open(path))
    print(f"{d['model']} {d['revision']} mode={d['mode']} d={d['d']} "
          f"n_layers={d['n_layers']} probe_loss={d['probe_loss']:.4f} "
          f"corpus={d['corpus_sha256_16']}")
    names = list(d["blocks"])
    bad = []
    for bn, rows in d["blocks"].items():
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    bad.append((bn, r["layer"], k))
            if abs(r["proj_check"] - 1) > 1e-3:
                bad.append((bn, r["layer"], "proj_check", r["proj_check"]))
    print("INVALID:", bad if bad else "none")

    b0 = d["blocks"][names[0]]
    print(f"\nblock={names[0]}  n_tokens={b0[0]['n_tokens']}")
    print(" L " + " ".join(f"{c:>8}" for c in COLS))
    for r in b0:
        print(f"{r['layer']:2d} " + " ".join(format(r[c], FMT[c]) for c in COLS))

    if len(names) > 1:
        print(f"\ncross-block spread (max |rel range| over {names}):")
        for c in ("ER_raw", "ER_ds", "E1_raw", "sink_alpha"):
            rel = []
            for li in range(len(b0)):
                v = np.array([d["blocks"][n][li][c] for n in names])
                rel.append((v.max() - v.min()) / (abs(v.mean()) + 1e-12))
            print(f"  {c:12s} max={max(rel):.3f} at L{int(np.argmax(rel))}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
