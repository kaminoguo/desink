#!/usr/bin/env python3
"""Regenerate every figure and every number reported in the paper.

Usage:  python build.py            # writes figs/*.pdf and numbers.txt

All inputs live in data/ and are the raw outputs of the scripts in experiments/.
Nothing here re-runs a model; this file only reads, checks and plots.  Every
number quoted in the manuscript is printed to numbers.txt next to the file it
came from, so a reader can diff the manuscript against the data.
"""

import csv
import json
import os
import sys
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

RAW = "#D55E00"      # Okabe-Ito vermillion
DS = "#0072B2"       # Okabe-Ito blue
THIRD = "#009E73"    # Okabe-Ito green
PINK = "#CC79A7"
GREY = "#666666"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
})

OUT = []
PEAK = {"pythia-160m": (4, 12, 768), "pythia-410m": (9, 24, 1024), "pythia-1.4b": (5, 24, 2048)}
NICE = {"pythia-160m": "Pythia-160M", "pythia-410m": "Pythia-410M", "pythia-1.4b": "Pythia-1.4B"}
OLMO = "OLMo-2-0425-1B-early-training"


def rec(line=""):
    OUT.append(line)
    print(line)


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def m_predict(share, m_res):
    """Propositions 2 and 3:  M(H) = exp(h_b(t)) * M(C) ** (1 - t)."""
    return np.exp(hb(share)) * m_res ** (1.0 - share)


def implied_share(m_raw, m_res):
    """Invert the identity for the share variable t."""
    if not (np.isfinite(m_raw) and np.isfinite(m_res) and m_res > 1.001):
        return np.nan
    try:
        return brentq(lambda t: m_predict(t, m_res) - m_raw, 1e-9, 1 - 1e-9)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- load sources
ROWS = list(csv.DictReader(open(os.path.join(DATA, "rerun_long.csv"))))
for r in ROWS:
    for k in list(r):
        if k not in ("exp", "model", "revision", "block"):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                r[k] = np.nan

corpus = load("rerun_corpus_pythia.json")
corpus_olmo = load("rerun_corpus_olmo.json")
scale = load("scale_validation.json")
qwen = load("qwen25_gqa_audit.json")
inject = load("sink_injection.json")
a5_gpt2 = load("audit_gpt2.json")
a5_p70 = load("audit_pythia70m.json")
cka = load("cka_heatmaps.json")
ident = load("sink_identification.json")
gap_g2 = load("gap_gpt2.json")
probe = load("probing_pythia70m.json")
t2 = {"pythia-160m": load("rerun_t2_160m.json"), "pythia-410m": load("rerun_t2_410m.json")}
traj70 = load("pythia70m_trajectory.json")


def sel(exp=None, model=None, layer=None, step=None):
    out = ROWS
    if exp is not None:
        out = [r for r in out if r["exp"] == exp]
    if model is not None:
        out = [r for r in out if r["model"] == model]
    if layer is not None:
        out = [r for r in out if int(r["layer"]) == layer]
    if step is not None:
        out = [r for r in out if int(r["step"]) == step]
    return out


def traj(model, layer, keys, exp="e1"):
    """Seed-mean and seed-std trajectory at one layer."""
    by = collections.defaultdict(list)
    for r in sel(exp, model, layer):
        by[int(r["step"])].append(r)
    steps = sorted(by)
    mean = {k: np.array([np.mean([g[k] for g in by[s]]) for s in steps]) for k in keys}
    std = {k: np.array([np.std([g[k] for g in by[s]]) for s in steps]) for k in keys}
    return np.array(steps), mean, std


# ---------------------------------------------------------------------- Fig 1
def fig_identity():
    e1 = np.array([r["E1_raw"] for r in ROWS])
    err = np.array([r["ER_raw"] for r in ROWS])
    erd = np.array([r["ER_ds"] for r in ROWS])
    ok = np.isfinite(e1) & np.isfinite(err) & np.isfinite(erd) & (e1 < 1) & (erd > 1)
    relER = np.abs(m_predict(e1[ok], erd[ok]) - err[ok]) / err[ok]

    p = e1
    r_ = np.array([implied_share(r["RankMe_raw"], r["RankMe_ds"]) for r in ROWS])
    good = np.isfinite(p) & np.isfinite(r_)

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    fam = {"pythia-160m": ("o", DS), "pythia-410m": ("s", THIRD),
           "pythia-1.4b": ("^", RAW), OLMO: ("D", PINK)}

    lo, hi = 0.7, 4000
    ax[0].plot([lo, hi], [lo, hi], color=GREY, lw=0.8, zorder=1)
    for m, (mk, c) in fam.items():
        idx = [i for i in range(len(ROWS)) if ok[i] and ROWS[i]["model"] == m]
        if not idx:
            continue
        ax[0].scatter(err[idx], m_predict(e1[idx], erd[idx]), s=5, marker=mk,
                      facecolor="none", edgecolor=c, linewidth=0.5,
                      label=NICE.get(m, "OLMo-2 1B"), zorder=3, alpha=0.55)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel(r"measured $\mathrm{ER}(\mathbf{H})$")
    ax[0].set_ylabel(r"predicted by Prop. 2")
    ax[0].legend(frameon=False, loc="upper left", handletextpad=0.1,
                 borderpad=0.1, markerscale=1.6)
    ax[0].set_title("(a) %d records, every layer and step" % ok.sum(), loc="left")
    ax[0].text(0.97, 0.06, "max rel. error %.0e" % relER.max(),
               transform=ax[0].transAxes, ha="right", fontsize=7, color=GREY)

    for m, (mk, c) in fam.items():
        idx = [i for i in range(len(ROWS)) if good[i] and ROWS[i]["model"] == m]
        if not idx:
            continue
        ax[1].scatter(p[idx], r_[idx], s=5, marker=mk, facecolor="none",
                      edgecolor=c, linewidth=0.5, zorder=3, alpha=0.55)
    xs = np.linspace(0, 1, 50)
    ax[1].plot(xs, xs, color=GREY, lw=0.8, zorder=1)
    ax[1].text(0.60, 0.70, r"$r=p$", fontsize=7, color=GREY)
    ax[1].set_xlabel(r"$p = E_1$, the share that drives effective rank")
    ax[1].set_ylabel(r"$r = \sigma_1/\sum_i\sigma_i$" "\nthe share that drives RankMe")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_title(r"(b) $r \leq p$ in all %d records" % good.sum(), loc="left")

    fig.tight_layout(w_pad=1.8)
    fig.savefig(os.path.join(FIGS, "fig1_identity.pdf"))
    plt.close(fig)

    rec("[Fig 1] Prop. 2 on %d records: median rel. err %.1e, p99 %.1e, max %.1e"
        % (ok.sum(), np.median(relER), np.percentile(relER, 99), relER.max()))
    rec("        r <= p violations: %d of %d; median p/r %.1f; max p/r %.1f"
        % (int((r_[good] > p[good] + 1e-9).sum()), int(good.sum()),
           np.median(p[good] / r_[good]), (p[good] / r_[good]).max()))


# ---------------------------------------------------------------------- Fig 2
def fig_dynamics():
    keys = ["ER_raw", "ER_ds", "RankMe_raw", "RankMe_ds", "E1_raw", "E1_ds",
            "sink_alpha", "p0_mass"]
    fig, ax = plt.subplots(2, 3, figsize=(6.6, 3.7), sharex=True)
    for j, (m, (L, nl, d)) in enumerate(PEAK.items()):
        steps, mu, sd = traj(m, L, keys)
        x = np.maximum(steps, 1)
        for i, (kr, kd, lab) in enumerate([("ER_raw", "ER_ds", "effective rank"),
                                           ("RankMe_raw", "RankMe_ds", "RankMe")]):
            a = ax[i][j]
            a.axvspan(16, 1000, color=THIRD, alpha=0.10, lw=0)
            a.plot(x, mu[kr], "o-", ms=2.5, lw=1.1, color=RAW, label="raw")
            a.plot(x, mu[kd], "s-", ms=2.5, lw=1.1, color=DS, label="residual")
            a.fill_between(x, mu[kr] - sd[kr], mu[kr] + sd[kr], color=RAW, alpha=0.3, lw=0)
            a.fill_between(x, mu[kd] - sd[kd], mu[kd] + sd[kd], color=DS, alpha=0.3, lw=0)
            a.set_xscale("log")
            if i == 0:
                a.set_yscale("log")
                a.set_title(r"%s L%d, $d=%d$" % (NICE[m], L, d), loc="left")
            if j == 0:
                a.set_ylabel(lab)
            if i == 1:
                a.set_xlabel("training step")
            if i == 0 and j == 0:
                a.legend(frameon=False, loc="lower left")

        i8 = int(np.where(steps == 8000)[0][0])
        rec("[Fig 2] %-12s L%-2d  ER raw %6.1f -> %5.1f (%.0fx down) | residual %6.1f -> %6.1f (%.2fx up)"
            % (NICE[m], L, mu["ER_raw"][0], mu["ER_raw"][-1],
               mu["ER_raw"][0] / mu["ER_raw"][-1], mu["ER_ds"][0], mu["ER_ds"][-1],
               mu["ER_ds"][-1] / mu["ER_ds"][0]))
        rec("             RankMe raw %6.1f -> %6.1f (%+.1f%%) | residual %6.1f -> %6.1f (%+.1f%%)"
            % (mu["RankMe_raw"][0], mu["RankMe_raw"][-1],
               100 * (mu["RankMe_raw"][-1] / mu["RankMe_raw"][0] - 1),
               mu["RankMe_ds"][0], mu["RankMe_ds"][-1],
               100 * (mu["RankMe_ds"][-1] / mu["RankMe_ds"][0] - 1)))
        rec("             RankMe after the sink forms (8k -> 143k): raw %+.1f%%, residual %+.1f%%"
            % (100 * (mu["RankMe_raw"][-1] / mu["RankMe_raw"][i8] - 1),
               100 * (mu["RankMe_ds"][-1] / mu["RankMe_ds"][i8] - 1)))
        rec("             E1 raw %.3f -> %.3f | residual %.3f -> %.3f (max %.3f at step %d)"
            % (mu["E1_raw"][0], mu["E1_raw"][-1], mu["E1_ds"][0], mu["E1_ds"][-1],
               mu["E1_ds"].max(), steps[int(np.argmax(mu["E1_ds"]))]))
        rec("             alpha %.2f -> %.2f | p0 mass %.3f -> %.3f | residual ER dips to %.1f at step %d"
            % (mu["sink_alpha"][0], mu["sink_alpha"][-1], mu["p0_mass"][0],
               mu["p0_mass"][-1], mu["ER_ds"].min(), steps[int(np.argmin(mu["ER_ds"]))]))
    fig.tight_layout(w_pad=1.2, h_pad=0.7)
    fig.savefig(os.path.join(FIGS, "fig2_dynamics.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------- Fig 3
def fig_control():
    keys = ["E1_raw", "E1_ds", "sink_alpha", "ER_raw", "ER_ds", "p0_mass"]
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.1))
    cols = {"pythia-160m": DS, "pythia-410m": THIRD, "pythia-1.4b": RAW}

    for m, (L, nl, d) in PEAK.items():
        steps, mu, sd = traj(m, L, keys)
        x = np.maximum(steps, 1)
        ax[0].plot(x, mu["E1_raw"], "-", lw=1.2, color=cols[m], label=NICE[m])
        ax[0].plot(x, mu["E1_ds"], "--", lw=1.0, color=cols[m], alpha=0.85)
        ax[1].plot(x, mu["sink_alpha"], "-", lw=1.2, color=cols[m])
    so, muo, sdo = traj(OLMO, 3, keys, exp="e2")
    xo = np.maximum(so, 1)
    ax[1].plot(xo, muo["sink_alpha"], "-", lw=1.2, color=PINK, label="OLMo-2 1B")

    ax[0].set_xscale("log")
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel(r"$E_1$")
    ax[0].set_title("(a) raw (solid), residual (dashed)", loc="left")
    ax[0].legend(frameon=False, loc="upper left")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("training step")
    ax[1].set_ylabel(r"sink strength $\alpha$")
    ax[1].axhline(1.0, color=GREY, lw=0.6, ls=":")
    ax[1].set_title("(b) sink strength", loc="left")
    ax[1].legend(frameon=False, loc="upper left")

    ax[2].plot(xo, muo["ER_raw"], "o-", ms=2.5, lw=1.1, color=RAW, label="raw")
    ax[2].plot(xo, muo["ER_ds"], "s-", ms=2.5, lw=1.1, color=DS, label="residual")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("training step")
    ax[2].set_ylabel("effective rank")
    ax[2].set_title("(c) OLMo-2 1B L3, no sink", loc="left")
    ax[2].legend(frameon=False, loc="upper left")

    fig.tight_layout(w_pad=1.4)
    fig.savefig(os.path.join(FIGS, "fig3_control.pdf"))
    plt.close(fig)

    e2 = sel("e2")
    gaps = [abs(r["ER_ds"] / r["ER_raw"] - 1) for r in e2 if r["ER_raw"] > 0]
    rec("[Fig 3] OLMo-2 1B, %d revisions x %d layers: max alpha %.3f over ALL layers, max p0 mass %.4f"
        % (len(set(r["revision"] for r in e2)), int(max(r["layer"] for r in e2)) + 1,
           max(r["sink_alpha"] for r in e2), max(r["p0_mass"] for r in e2)))
    rec("        |ER residual / ER raw - 1|: median %.1f%%, p95 %.1f%%, max %.1f%%"
        % (100 * np.median(gaps), 100 * np.percentile(gaps, 95), 100 * max(gaps)))
    rec("        L3 ER raw %.1f (init) -> %.1f (step 1000 dip, alpha %.2f) -> %.1f (step 36000)"
        % (muo["ER_raw"][0], muo["ER_raw"][1], muo["sink_alpha"][1], muo["ER_raw"][-1]))


# ---------------------------------------------------------------------- Fig 4
def fig_cka():
    e = [x for x in cka if x["model"] == "gpt2"][0]
    A = np.array(e["raw_cka"])
    B = np.array(e["desinked_cka"])
    D = B - A
    n = A.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.15))
    for a, M, ttl, cmap, vlim in [(ax[0], A, "(a) raw CKA", "viridis", (0, 1)),
                                  (ax[1], B, "(b) residual CKA", "viridis", (0, 1)),
                                  (ax[2], D, "(c) residual minus raw", "RdBu_r", (-0.6, 0.6))]:
        im = a.imshow(M, cmap=cmap, vmin=vlim[0], vmax=vlim[1], origin="lower")
        a.set_title(ttl, loc="left")
        a.set_xticks(range(0, n, 2))
        a.set_yticks(range(0, n, 2))
        a.set_xlabel("layer")
        if a is ax[0]:
            a.set_ylabel("layer")
        a.spines[:].set_visible(False)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIGS, "fig4_cka.pdf"))
    plt.close(fig)
    m = ~np.eye(n, dtype=bool)
    rec("[Fig 4] GPT-2 CKA off-diagonal mean %.3f -> %.3f; most negative change %.3f at (L%d,L%d)"
        % (A[m].mean(), B[m].mean(), D.min(),
           np.unravel_index(D.argmin(), D.shape)[0], np.unravel_index(D.argmin(), D.shape)[1]))
    p = [x for x in cka if "pythia" in x["model"]][0]
    Ap, Bp = np.array(p["raw_cka"]), np.array(p["desinked_cka"])
    mp = ~np.eye(Ap.shape[0], dtype=bool)
    Dp = Bp - Ap
    rec("        Pythia-70M CKA off-diagonal mean %.3f -> %.3f; largest change %+.3f"
        % (Ap[mp].mean(), Bp[mp].mean(), Dp.flatten()[np.abs(Dp).argmax()]))



# ---------------------------------------------------------------------- Fig 5
def fig_shared_factor():
    """Proposition 5 assumes a band of layers shares one left factor. Test it."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.0))
    cols = {"pythia-160m": DS, "pythia-410m": THIRD, "pythia-1.4b": RAW}
    for m, (L, nl, d) in PEAK.items():
        g = collections.defaultdict(list)
        for r in sel("e1", m, step=143000):
            g[int(r["layer"])].append(r)
        Ls = sorted(g)
        frac = [x / max(Ls) for x in Ls]
        ax[0].plot(frac, [np.mean([q["p0_mass"] for q in g[x]]) for x in Ls],
                   "o-", ms=2.5, lw=1.1, color=cols[m], label=NICE[m])
        ax[1].plot(frac, [np.mean([q["E1_raw"] for q in g[x]]) for x in Ls],
                   "o-", ms=2.5, lw=1.1, color=cols[m])
    ax[0].axhline(0.2, color=GREY, lw=0.6, ls=":")
    ax[0].set_ylabel("position-0 mass of $\\mathbf{s}$")
    ax[1].set_ylabel(r"$E_1$")
    for a, t in zip(ax, ["(a) the leading direction sits on position 0",
                         r"(b) $E_1$ over the same layers"]):
        a.set_xlabel("relative depth")
        a.set_title(t, loc="left")
    ax[0].legend(frameon=False, loc="center right")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(FIGS, "fig5_shared_factor.pdf"))
    plt.close(fig)

    rec("[Fig 5] position-0 mass of the leading direction by layer, step 143000:")
    for m, (L, nl, d) in PEAK.items():
        g = collections.defaultdict(list)
        for r in sel("e1", m, step=143000):
            g[int(r["layer"])].append(r["p0_mass"])
        v = {x: np.mean(y) for x, y in g.items()}
        band = [x for x in sorted(v) if v[x] > 0.20]
        inband = [v[x] for x in band]
        rec("  %-12s band L%d..L%d (%d of %d layers), p0 mass %.2f to %.2f, spread within band %.3f"
            % (NICE[m], band[0], band[-1], len(band), len(v),
               min(inband), max(inband), np.std(inband)))
        rec("               outside the band: " + " ".join(
            "L%d %.2f" % (x, v[x]) for x in sorted(v) if x not in band))


# ------------------------------------------------------- attention to position 0
def attention_evidence():
    d = load("pythia70m_sink_emergence.json")
    ck = d["checkpoints"]
    a = np.array([[x for x in c["layers"] if x["layer"] == 3][0]["attn_to_p0"] for c in ck])
    nr = np.array([[x for x in c["layers"] if x["layer"] == 3][0]["norm_ratio"] for c in ck])
    from scipy.stats import spearmanr
    rec("[attention] Pythia-70M L3, %d checkpoints, %d tokens: attn to position 0 %.4f -> %.4f (%.0fx)"
        % (len(ck), d["n_test_tokens"], a[0], a[-1], a[-1] / a[0]))
    rec("            Spearman(attn to position 0, alpha) = %.3f (p = %.1e)"
        % spearmanr(a, nr))
    rec("            per checkpoint: " + " ".join("%d:%.3f" % (c["step"], v) for c, v in zip(ck, a)))


# ------------------------------------------------------------------- numbers
def tables():
    rec()
    rec("=" * 78)
    rec("CORPUS (data/rerun_corpus_*.json)")
    rec("=" * 78)
    for tag, c in [("Pythia", corpus), ("OLMo-2", corpus_olmo)]:
        rec("  %-7s %s" % (tag, c["source"]))
        rec("          %d tokens, %d sequences of %d, %d blocks of %d tokens"
            % (c["n_tokens"], c["n_seq"], c["seq_len"], c["n_seeds"],
               c["n_seq_per_seed"] * c["seq_len"]))
        rec("          sha256_16 %s | %d distinct token ids | %d duplicate sequences"
            % (c["sha256_16"], c["checks"]["n_unique_token_ids"],
               c["checks"]["n_duplicate_sequences"]))

    rec()
    rec("=" * 78)
    rec("THE METRIC SPLIT at the final checkpoint (seed mean)")
    rec("=" * 78)
    rec("  %-12s %-4s %6s %8s %6s %8s %9s %9s %11s"
        % ("model", "L", "p=E1", "r", "p/r", "D_ER", "D_RankMe", "ER_res", "RankMe_res"))
    for m, (L, nl, d) in PEAK.items():
        g = sel("e1", m, L, 143000)
        p = np.mean([x["E1_raw"] for x in g])
        r_ = np.mean([implied_share(x["RankMe_raw"], x["RankMe_ds"]) for x in g])
        rec("  %-12s L%-3d %6.3f %8.4f %6.1f %8.1f %9.2f %9.1f %11.1f"
            % (NICE[m], L, p, r_, p / r_,
               np.mean([x["ER_ds"] / x["ER_raw"] for x in g]),
               np.mean([x["RankMe_ds"] / x["RankMe_raw"] for x in g]),
               np.mean([x["ER_ds"] for x in g]), np.mean([x["RankMe_ds"] for x in g])))
        ar = np.mean([x["aniso_uncentered_raw"] for x in g])
        ad = np.mean([x["aniso_uncentered_ds"] for x in g])
        rec("               uncentered anisotropy %.3f -> %.3f (%+.0f%%)"
            % (ar, ad, 100 * (ad / ar - 1)))

    rec()
    rec("=" * 78)
    rec("CROSS-SEED SPREAD (3 disjoint 200,704-token blocks), peak layers")
    rec("=" * 78)
    for m, (L, nl, d) in PEAK.items():
        steps, mu, sd = traj(m, L, ["ER_raw", "ER_ds", "E1_raw", "RankMe_ds", "sink_alpha"])
        rec("  %-12s " % NICE[m] + "  ".join(
            "%s %.2f%%" % (k, 100 * np.median(sd[k][mu[k] > 0] / mu[k][mu[k] > 0]))
            for k in ["ER_raw", "ER_ds", "E1_raw", "RankMe_ds", "sink_alpha"]))

    rec()
    rec("=" * 78)
    rec("SINK STRENGTH PREDICTS E1  (single corpus, 20 checkpoints, seed mean)")
    rec("=" * 78)
    for m, (L, nl, d) in PEAK.items():
        steps, mu, _ = traj(m, L, ["E1_raw", "sink_alpha"])
        a, e = mu["sink_alpha"], mu["E1_raw"]
        k = least_squares(lambda q: q[0] * a ** 2 / (q[0] * a ** 2 + 1) - e, [0.05]).x[0]
        pred = k * a ** 2 / (k * a ** 2 + 1)
        r2 = 1 - np.sum((e - pred) ** 2) / np.sum((e - e.mean()) ** 2)
        rec("  %-12s kappa %.5f  R2 %.4f  Pearson(x, a^2) %.4f (p %.1e)  n = %d"
            % (NICE[m], k, r2, *pearsonr(e / (1 - e), a ** 2), len(a)))

    rec()
    rec("=" * 78)
    rec("PC1 SPECIFICITY (data/rerun_t2_*.json, same corpus, 5 random directions)")
    rec("=" * 78)
    for m in ["pythia-160m", "pythia-410m"]:
        L = PEAK[m][0]
        lay = [x for x in t2[m]["layers"] if x["layer"] == L][0]
        rnd = [lay["rand%d" % i]["ER"] for i in range(5)]
        rec("  %-12s L%-2d ER: baseline %7.1f | PC1 %7.1f | PC2 %7.1f | PC3 %7.1f | random %7.1f +- %.2f"
            % (NICE[m], L, lay["baseline"]["ER"], lay["pc1"]["ER"], lay["pc2"]["ER"],
               lay["pc3"]["ER"], np.mean(rnd), np.std(rnd)))
    lay = [x for x in t2["pythia-160m"]["layers"] if x["layer"] == 6][0]
    g = sel("e1", "pythia-160m", 6, 143000)
    rec("  cross-implementation check, Pythia-160M L6: T2 gives %.1f / %.1f, T1 gives %.1f / %.1f"
        % (lay["baseline"]["ER"], lay["pc1"]["ER"],
           np.mean([x["ER_raw"] for x in g]), np.mean([x["ER_ds"] for x in g])))

    rec()
    rec("=" * 78)
    rec("SAMPLE-SIZE DEPENDENCE (exp e3, Pythia-1.4B L5)")
    rec("=" * 78)
    ns = [8192, 32768, 131072, 524288]
    for step in (512, 8000, 143000):
        rows = [r for r in ROWS if r["exp"] == "e3" and int(r["step"]) == step
                and int(r["layer"]) == 5]
        by = {r["block"]: r for r in rows}
        for k in ["E1_raw", "sink_alpha", "aniso_uncentered_raw", "ER_raw", "ER_ds", "RankMe_raw"]:
            v = [by["n%d" % n][k] for n in ns if ("n%d" % n) in by]
            if len(v) == len(ns):
                rec("  step %6d  %-20s " % (step, k) + "  ".join("%9.3f" % x for x in v)
                    + "   max deviation from largest n: %.1f%%"
                    % (100 * max(abs(np.array(v) / v[-1] - 1))))
        rec("")
    rows = [r for r in ROWS if r["exp"] == "e3" and int(r["step"]) == 143000
            and int(r["layer"]) == 5]
    by = {r["block"]: r for r in rows}
    rec("  raw-to-residual ER ratio at step 143000: "
        + "  ".join("n=%d %.0fx" % (n, by["n%d" % n]["ER_ds"] / by["n%d" % n]["ER_raw"])
                    for n in ns if ("n%d" % n) in by))

    rec()
    rec("=" * 78)
    rec("MEASUREMENTS RETAINED FROM THE FIRST SUBMISSION")
    rec("=" * 78)
    for tag, d in [("GPT-2", a5_gpt2), ("Pythia-70M", a5_p70)]:
        Ls = d["layers"]
        i = int(np.argmax([L["e1_raw"] for L in Ls]))
        rec("  %-11s peak-E1 L%d: E1 %.1f%% -> %.1f%% | rogue5 %.1f%% -> %.1f%% | "
            "RankMe %.0f -> %.0f | aniso %.3f -> %.3f"
            % (tag, Ls[i]["layer"], Ls[i]["e1_raw"], Ls[i]["e1_desink"],
               Ls[i]["rogue_top5_var_pct_raw"], Ls[i]["rogue_top5_var_pct_desink"],
               Ls[i]["rankme_raw"], Ls[i]["rankme_desink"],
               Ls[i]["aniso_raw"], Ls[i]["aniso_desink"]))
    ov = [L["rogue_sink_overlap"] for L in a5_gpt2["layers"][3:11]]
    rec("  GPT-2 sink mass in the top-5 rogue dims, L3 to L10: %.1f%% to %.1f%%" % (min(ov), max(ov)))
    for m in scale:
        name = {"EleutherAI/pythia-6.9b": "Pythia-6.9B", "gpt2-xl": "GPT-2-XL"}[m["model"]]
        Ls = m["layers"]
        i = int(np.argmax([L["desinked"]["er"] / L["raw"]["er"] for L in Ls]))
        L = Ls[i]
        rec("  %-11s L%-2d E1 %.3f -> %.3f, D = %.0fx, bound ER(C) = %.0f"
            % (name, L["layer"], L["raw"]["e1"], L["desinked"]["e1"],
               L["desinked"]["er"] / L["raw"]["er"], L["desinked"]["er"]))
    Ls = qwen["layers"]
    i = int(np.argmax([L["ER_ds"] / L["ER_raw"] for L in Ls]))
    L = Ls[i]
    rec("  %-11s L%-2d E1 %.3f -> %.3f, D = %.0fx, bound ER(C) = %.0f; peak gap %.0f, peak alpha %.0f"
        % ("Qwen2.5-1.5B", L["layer"], L["E1_raw"], L["E1_ds"], L["ER_ds"] / L["ER_raw"],
           L["ER_ds"], max(x["gap"] for x in Ls), max(x["sink_alpha"] for x in Ls)))
    rec("  GPT-2 spectral gap: " + "  ".join("L%d %.1f" % (L["layer"], L["gap_ratio"])
                                             for L in gap_g2["layers"]))
    for lay in ("3", "1"):
        rec("  Pythia-70M L%s (WikiText-2, %d tokens, 1 block): ER raw %.1f -> %.1f, residual %.1f -> %.1f"
            % (lay, traj70[0]["layers"][lay]["n_tokens"],
               traj70[0]["layers"][lay]["raw"]["er"], traj70[-1]["layers"][lay]["raw"]["er"],
               traj70[0]["layers"][lay]["desinked"]["er"], traj70[-1]["layers"][lay]["desinked"]["er"]))

    rec()
    rec("=" * 78)
    rec("SYNTHETIC INJECTION (data/sink_injection.json)")
    rec("=" * 78)
    rec("  base Pythia-70M step %d %s: residual E1 %.3f, residual ER %.1f"
        % (inject["base_step"], inject["base_layer"], inject["clean_E1"], inject["clean_ER"]))
    for r in inject["injections"]:
        rec("  alpha %3d: E1 %.3f  ER %7.2f   (Prop. 2 with ER(C) = %.1f predicts %7.2f)"
            % (r["alpha_scale"], r["E1"], r["ER"], inject["clean_ER"],
               m_predict(r["E1"], inject["clean_ER"])))

    rec()
    rec("=" * 78)
    rec("PROBING (data/probing_pythia70m.json)")
    rec("=" * 78)
    for L in probe["layers"]:
        rec("  L%d  sink var %5.1f%%  raw %.2f+-%.2f  residual %.2f+-%.2f  d_res %+.2f  d_rand %+.2f"
            % (L["layer"], L["sink_var_pct"], L["raw_test_mean"], L["raw_test_std"],
               L["desink_test_mean"], L["desink_test_std"], L["delta_desink"], L["delta_rand"]))

    rec()
    rec("=" * 78)
    rec("SINK IDENTIFICATION, first-submission run (data/sink_identification.json)")
    rec("=" * 78)
    for r in ident["results"]:
        rec("  %-14s %-4s step %6d  E1(H) %.3f  E1(C) %.3f  p0 mass %5.1f%%  alpha %5.2f"
            % (r["model"], r["layer"], r["step"], r["e1_H_bar"], r["e1_of_C"],
               100 * r["p0_concentration_of_sink_dir"], r["norm_ratio_p0_vs_rest"]))


def main():
    rec("# numbers.txt  regenerated by build.py; every value below appears in the manuscript")
    rec()
    fig_identity()
    fig_dynamics()
    fig_control()
    fig_cka()
    fig_shared_factor()
    attention_evidence()
    tables()
    with open(os.path.join(HERE, "numbers.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")
    print("\nwrote figs/*.pdf and numbers.txt")


if __name__ == "__main__":
    sys.exit(main())
