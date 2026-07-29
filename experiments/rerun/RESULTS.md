# T1 rerun — results

76 jobs (E1 60 + E3 3 + E2 13), 4743 records, **0 invalid**.
Validity gate `validate_all.py`: no NaN/inf; `proj_check == 1` everywhere;
E1∈[0,1]; ER∈(0,d]; cross-seed relative spread median 2.0% / p95 6.3%.
Corpus sha `66f376629b477962` reproduced bit-identically on three separate boxes;
pythia-160m step143000 metrics matched to every printed digit across boxes.
Only byte-identical checkpoint pair: step0 vs step1 (all 3 Pythia models) —
expected under the 1430-step LR warmup; safetensors md5s do differ.

## E1 — main claim holds, and is stronger than the old (void) numbers

At each model's peak-sink layer, step0 → step143000:

| model | layer | ER_raw | ER_ds | E1_raw | E1_ds | sink_alpha | p0_mass |
|---|---|---|---|---|---|---|---|
| pythia-1.4b | L5 (of 24) | 441.6 → **4.2** | 477.8 → **1023.2** | 0.040 → 0.853 | 0.038 → **0.033** | 1.22 → 25.8 | 0.001 → 0.498 |
| pythia-410m | L9 (of 24) | 278.5 → **2.4** | 294.7 → **553.8** | 0.039 → 0.910 | 0.036 → **0.028** | 1.20 → 43.3 | 0.001 → 0.563 |
| pythia-160m | L4 (of 12) | 237.1 → **11.1** | 250.3 → **356.1** | 0.040 → 0.696 | 0.041 → **0.048** | 1.23 → 19.5 | 0.001 → 0.441 |

The reported 105× "rank collapse" becomes a 2.1× **rank growth** once the top
eigendirection is projected out. `E1_ds` is flat from random init to convergence
in all three models — the entire E1 trajectory lives in one direction.
`p0_mass ≈ 0.5`: position-0 tokens are 1/512 = 0.2% of the corpus but carry half
the sink direction's energy (≈250× enrichment).

## Three findings that do NOT support the blanket claim

**1. RankMe never collapses, so it cannot be "hijacked".**
pythia-1.4b L5 `RankMe_raw`: 1430.5 → 1470.4 over all of training (+2.8%), on the
exact same tokens where `ER_raw` fell 105×. RankMe normalises singular values
σ=√λ, which compresses the dynamic range and is near-blind to one dominant
direction. De-sinking does raise it (1441.8 → 1820.0, +26%), so RankMe is
sink-*sensitive* — but there is no collapse to explain. The correct claim is that
the collapse is **metric-specific**: it appears in eigenvalue-entropy effective
rank, essentially not in RankMe.

**2. Anisotropy is only marginally explained by the sink.**
Uncentered mean pairwise cosine, final checkpoint, raw → de-sinked:
1.4b 0.537 → 0.480 (−11%), 410m 0.290 → 0.272 (−6%), 160m 0.252 → 0.196 (−22%).
Its own dynamics are sink-independent: it peaks at 0.62–0.73 around step 16, when
`sink_alpha` is still 1.0–1.1, then falls. Do not file it under "hijacked".
(The centred variant the spec asked for sits at 0.0005–0.009 by construction and
cannot show the effect at all; both are recorded.)

**3. There is a real early rank dip with no sink present.**
pythia-1.4b L5: `ER_raw` 420 → 149 between step 16 and step 256 while
`sink_alpha` stays 1.07–1.22, and de-sinking does not recover it (212 → 295).
It then climbs back to 577 by step 2000; the sink-driven collapse only starts
after step 4000. Two distinct phenomena. Replicated in OLMo (below).

## E2 — OLMo-2 forms no sink in its first 76B tokens, and shows no collapse

13 stage-1 revisions, step0 → step36000 (76B tokens ≈ 2% of the run).

| step | ER_raw | ER_ds | E1_raw | sink_alpha | p0_mass |
|---|---|---|---|---|---|
| 0 | 1008.4 | 1057.4 | 0.022 | 1.01 | 0.008 |
| 1000 | 111.0 | 134.8 | 0.110 | 1.04 | 0.0003 |
| 8000 | 937.2 | 961.8 | 0.015 | 1.10 | 0.002 |
| 36000 | 1272.4 | 1297.4 | 0.011 | 1.18 | 0.002 |

`sink_alpha` never exceeds **1.168 across all 17 layers** and `p0_mass ≤ 0.004`.
With no sink, raw and de-sinked metrics agree to ~2% and `ER_raw` *grows*
monotonically. This is the control the argument needs: the raw/de-sinked gap is a
function of sink strength, not of the de-sinking operation itself.

**Do not over-read this.** 36000 steps is ~2% of OLMo-2-1B's run. Pythia's sink at
L5 only became visible after step 4000/143000 ≈ 2.8%. OLMo's `sink_alpha` is
ticking up at the end (1.07 → 1.18). This shows *no sink in the first 76B tokens*,
**not** that OLMo never forms one. The `-early-training` repo stops at step 36000;
extending needs the main `allenai/OLMo-2-0425-1B` checkpoints.

Note the step0 → step1000 dip (1008 → 111) with `sink_alpha = 1.04`: the same
sink-independent early rank dip as Pythia, in a second architecture family.

## E3 — half of the metrics are NOT sample-size robust

pythia-1.4b, peak-sink layer, n ∈ {8192, 32768, 131072, 524288}; max relative
deviation from the largest-n value:

| metric | step512 | step8000 | step143000 |
|---|---|---|---|
| `E1_raw` | 10.4% | 5.1% | **0.5%** |
| `sink_alpha` | 1.1% | 2.4% | **0.8%** |
| `aniso_uncentered_raw` | 18.1% | 3.1% | **1.6%** |
| `ER_raw` | 13.6% | 24.3% | 7.7% |
| `ER_ds` | 13.5% | 24.3% | **27.1%** |
| `RankMe_raw` | 10.9% | 8.5% | 11.5% |

`ER`/`RankMe` rise monotonically with n and have **not converged at 524,288 tokens**
for d=2048 (step143000 L5 `ER_ds`: 755.7 → 938.7 → 986.6 → 1036.3). This is
Marchenko–Pastur sample-spectrum bias: small eigenvalues are underestimated at low
n, so entropy-based effective rank approaches its limit from below. `ER_ds` is the
worst affected because it lives in the tail.

Consequence for the paper: every E1 point is at the same fixed n = 200,704, so all
comparisons are valid, and the direction is robust at every n (the raw/de-sinked
ratio is 194× at n=8192 and 246× at n=524288). But **absolute ER values must not be
reported as "the effective rank" of a layer** — they are n-dependent. State
comparisons at fixed n; put the non-convergence in limitations.
Ratio-type metrics (`E1`, `sink_alpha`, anisotropy) are n-robust and can be quoted
absolutely.

## T2 — de-sinking is not "remove any direction"

`exp_t1_all.py`'s T2-1/T2-2 read the same poisoned `texts` variable as T1, so that
data is void too. Rerun on the validated corpus (`t2_worker.py`), ER at the peak
layer:

| | pythia-70m L3 s143000 | pythia-70m L3 s4000 | pythia-160m L6 s143000 |
|---|---|---|---|
| baseline | 33.5 | 156.5 | 11.0 |
| remove PC1 (= de-sink) | **182.5** | **256.0** | **223.2** |
| remove PC2 | 32.5 | 159.1 | 10.1 |
| remove random dir (5 seeds) | 33.4 | 156.3 | 11.0 |

Cross-implementation check: T2's `pythia-160m` L6 baseline/PC1 (11.0 / 223.2)
equals T1's `ER_raw`/`ER_ds` to every printed digit, from independently written code.

## Artifacts

`results/*.json` (82 files) · `agg/t1_rerun_long.csv` (4743 rows) ·
`agg/t1_rerun_summary.json` (mean/std over 3 seeds) ·
`agg/t1_rerun_eigs.npz` (4743 top-20 eigenspectra) · `agg/corpus*_meta.json`
