# Training-dynamics rerun: Pythia 160M/410M/1.4B and OLMo-2 1B

Produces `data/rerun_long.csv`, the 4,743 records behind Figures 1 to 3 and Tables 1, 2, 3
and 7 of the paper.

These scripts replace two earlier ones whose data is void. Those two are not in this
repository; they were deleted from the working tree and remain in the git history. The
paths below assume the box layout `/workspace/desink`; change the constant at the top of
each file to run elsewhere.

## Why the old data is void

`exp_t1_all.py::load_texts` wrapped the dataset load in a bare `except` and fell back to
`["The quick brown fox … " * 10] * 200` — 200 byte-identical sequences. Every spectral
metric was then computed on a matrix with ~10 distinct tokens, so d=768 models reported
step-0 effective rank ≈ 10 instead of a few hundred. `exp_t1_2_fix.py` never attempted a
dataset load at all. Secondary bug: `sink_alpha = norms[0]/norms[1:].mean()` estimated the
numerator from a *single* token (sequence 0, position 0) and polluted the denominator with
every other sequence's position-0 token.

Refuted by construction in this rerun — `build_corpus.py` asserts
`n_unique_token_ids >= 5000`, `n_duplicate_sequences == 0`, `per_seq_unique_frac >= 0.30`,
prints decoded samples, and has no bare `except` anywhere in the pipeline. Measured:
28,552 unique token ids, 0 duplicate sequences, 0.524 mean unique fraction.
Confirmed against the failure signature: pythia-1.4b step0 mid-layer `ER_raw = 431`
(old broken value ≈ 10), `sink_alpha = 1.17` (no sink at init), `probe_loss = 11.10 ≈ ln(50304)`.

## Protocol

| item | value |
|---|---|
| models | `EleutherAI/pythia-{160m,410m,1.4b}` |
| checkpoints | 20 × `revision=step{0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,128000,143000}` |
| layers | all, incl. embedding output (`hidden_states[0..n_layers]`) |
| corpus | `Skylion007/openwebtext` @ `refs/convert/parquet` shard 0000, docs concatenated (no separator) then chunked — matches the pythia-70m protocol in `exp_b4_v3` |
| tokens | 602,112 = 3 disjoint seeds × 200,704 |
| seq len | 512 |
| dtype | float32, **TF32 disabled** (10-bit mantissa would corrupt the small eigenvalues ER/RankMe depend on) |
| corpus sha256[:16] | `66f376629b477962` (pythia) / see `corpus_olmo_meta.json` (OLMo) |

One corpus, tokenized once, memmapped by every job — cross-checkpoint curves are only
comparable if the corpus is bit-identical. Pythia shares one tokenizer across the suite.

## Metrics (per model × checkpoint × layer × seed)

H ∈ R^{n×d} = all tokens of one layer/seed; `C = (1/n)·H̄ᵀH̄` with `H̄ = H − μ`;
λ₁≥…≥λ_d and top eigenvector `s`.

| key | definition |
|---|---|
| `E1_raw` / `E1_ds` | λ₁/Σλ  /  λ₂/(Σλ−λ₁) |
| `ER_raw` / `ER_ds` | exp(entropy of normalized λ) over λ₁.. / λ₂.. |
| `RankMe_raw` / `RankMe_ds` | exp(entropy of normalized σ=√λ), ε=1e-7, over σ₁.. / σ₂.. |
| `aniso_raw` / `aniso_ds` | mean pairwise cosine of **centred** unit rows, before / after removing `s` |
| `aniso_uncentered_raw` / `_ds` | same on **uncentred** rows — this is the literature's anisotropy (Ethayarajh / RepDeg); the centred variant is ~0.005 by construction and cannot show the hijack |
| `gap_ratio` | σ₁/σ₂ |
| `sink_alpha` | mean‖h‖ over **all** position-0 tokens ÷ mean‖h‖ over **all** non-position-0 tokens |
| `p0_mass` | share of Σ⟨h̄ᵢ,s⟩² contributed by position-0 rows |
| `top20_eigenvalues`, `n_tokens`, `d`, `mu_norm`, `mean_token_norm`, `probe_loss` | |

De-sinking is the rank-1 projection `P = I − ssᵀ`. Because `Cs = λ₁s` exactly,
`PCP = C − λ₁ssᵀ`, so the de-sinked spectrum **is** λ₂..λ_d — no second SVD. Anisotropy is
not a spectral functional (row-wise normalization), so it does need the explicit projection.

## Implementation

Three streaming passes per (checkpoint, seed); memory is O(d²), not O(n·d):

1. μ, position-0 / non-position-0 norm sums — O(nd)
2. centred covariance C → eigendecomposition (fp64) — O(nd²)
3. anisotropy sums (needs μ) and de-sinked anisotropy sums (needs s), plus the
   position-0 share of sink-direction energy — O(nd)

**Self-check written into every record:** `proj_check = Σ⟨h̄ᵢ,s⟩² / (n·λ₁)` must be 1.
A job raises if any layer deviates by >1e-3. All runs so far: exactly 1.0000.

**Second self-check:** `probe_loss` on 8 held-out sequences. Distinguishes checkpoints that
failed to load distinct weights (step0 → 11.10 ≈ ln V, step143000 → 2.05).

## Files

| file | role |
|---|---|
| `build_corpus.py` | tokenize once, validate, save `corpus.npy` + meta |
| `t1_worker.py` | one (model, revision) → all layers × all seeds; `--mode e1\|e3` |
| `orchestrate.py` | 4-GPU work queue, longest-job-first, idempotent on results |
| `prefetch.py` | 158 GB weight prefetch, overlaps compute |
| `e2_setup.py` | OLMo corpus + weight prefetch |
| `e2_orchestrate.py` | 4-GPU queue for E2 |
| `watchdog.sh` | cron */3 — restarts anything dead, chains E2 behind E1, reaps weights |
| `reap.py` | delete cached weights whose result JSON exists |
| `inspect_result.py` | per-layer table + NaN/range/proj_check gate for one result |
| `aggregate.py` | → `t1_rerun_long.csv`, `t1_rerun_summary.json`, `t1_rerun_eigs.npz` |

## Sub-experiments

- **E1** (main) 3 models × 20 checkpoints × all layers × 3 seeds = 60 jobs
- **E3** pythia-1.4b at steps 512 / 8000 / 143000, n ∈ {8192, 32768, 131072, 524288}
  tokens — confirms the metrics are not sample-size artifacts. 3 jobs.
- **E2** `allenai/OLMo-2-0425-1B-early-training`, 13 stage-1 revisions, same protocol,
  OLMo-tokenized corpus from the same documents. Second architecture family.
