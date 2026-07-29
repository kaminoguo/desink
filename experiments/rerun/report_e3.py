"""E3: does the token budget change the metrics?  Print each metric vs n at the
peak-sink layer, plus the max deviation from the largest-n value."""
import json, glob, sys
import numpy as np

COLS = ["ER_raw", "ER_ds", "E1_raw", "E1_ds", "RankMe_raw", "RankMe_ds",
        "aniso_uncentered_raw", "sink_alpha", "p0_mass"]


def main(paths):
    for p in sorted(paths):
        d = json.load(open(p))
        blocks = list(d["blocks"])
        n_layers = len(d["blocks"][blocks[0]])
        big = d["blocks"][blocks[-1]]
        L = max(range(n_layers), key=lambda i: big[i]["sink_alpha"])
        print(f"=== {d['model'].split('/')[-1]} {d['revision']}  L{L} (peak sink) ===")
        print(f"{'n_tokens':>9} " + " ".join(f"{c:>10}" for c in COLS))
        vals = {}
        for bn in blocks:
            r = d["blocks"][bn][L]
            vals[r["n_tokens"]] = [r[c] for c in COLS]
            print(f"{r['n_tokens']:>9} " + " ".join(f"{r[c]:10.4f}" for c in COLS))
        ref = vals[max(vals)]
        dev = {c: max(abs(vals[n][i] - ref[i]) / (abs(ref[i]) + 1e-12) for n in vals)
               for i, c in enumerate(COLS)}
        print("  max |rel dev| vs largest n: " +
              "  ".join(f"{c}={dev[c]*100:.1f}%" for c in COLS))
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or glob.glob("results/e3_*.json"))
