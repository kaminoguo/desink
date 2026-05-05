# De-Sinking

> **Phantom Phases: How Attention Sinks Generate Fictitious Training Dynamics**
> NeurIPS 2026 submission · [paper](paper/main.pdf)

Standard spectral metrics on transformer hidden states — $E_1$, effective rank, RankMe, CKA, anisotropy — are systematically contaminated by the **attention sink**, an input-invariant high-variance direction that emerges during training.

The "geometric phases" and "rank collapse" widely reported in the literature are largely measurement artifacts. *De-sinking*, a single rank-1 projection, removes the contamination and reveals a different story: monotonic rank **growth**, not collapse, and a previously invisible shortcut-to-full-rank transition. Distortions reach $721\times$ in effective rank at 6.9B parameters.

## Install & use

```bash
pip install -r requirements.txt
```

```python
from desink import desink, spectral_metrics

H_ds = desink(H)               # H: (n_tokens, d_model) at any layer
spectral_metrics(H)            # raw  -- contaminated
spectral_metrics(H_ds)         # de-sinked -- corrected
```

The whole method is one line: `H_ds = H - (H @ s)[:, None] * s`, where `s` is PC1 of the centered `H`.

## Reproducing the paper

Each experiment is a standalone script in `experiments/`. Run any single artifact directly:

```bash
python experiments/fig1_training_trajectory.py    # ~30 min on A100
python experiments/tab2_contamination_audit.py    # ~20 min
python experiments/tab3_fig4_scale_validation.py  # ≥40 GB GPU for Pythia-6.9B
```

All models are public Hugging Face checkpoints (GPT-2, Pythia, OLMo-2, Qwen2.5). Total compute for the full paper: **~10 GPU-hours on a single A100**.

<details>
<summary><strong>Full experiment ↔ paper map</strong></summary>

**Main paper**

| Script | Paper artifact |
|---|---|
| `fig1_training_trajectory.py`           | Figure 1(a,b) — $E_1$ / ER on 20 Pythia-70M checkpoints |
| `fig1c_theorem_fit.py`                  | Figure 1(c) — Theorem 1 fit, $R^2 = 0.978$ |
| `tab2_contamination_audit.py`           | Table 2 + Figure 2 — five published metrics audited |
| `fig3_cka_heatmap.py`                   | Figure 3 — layer × layer CKA heatmaps |
| `tab3_fig4_scale_validation.py`         | Table 3 + Figure 4 — scale validation, 70M → 6.9B |
| `sec34_manufacture_and_qwen_audit.py`   | §3.4 synthetic sink injection + Table 3 Qwen2.5 row |
| `sec5_dynamics_and_controls.py`         | §5 — Pythia-160M / 1.4B / OLMo-2 1B dynamics + ER↔probe correlation |
| `fig5a_alignment_peak.py`               | Figure 5(a) — Pythia alignment peak |
| `fig5a_olmo_alignment.py`               | Figure 5(a) — OLMo-2 1B validation |
| `fig5b_perlayer_departure.py`           | Figure 5(b) — inverted-U per-layer departure |
| `fig6_capability_vs_sink.py`            | Figure 6 — capability vs sink growth |
| `sec7_sink_emergence.py`                | §7 — sink formation across training |

**Controls and appendices**

| Script | Paper artifact |
|---|---|
| `controls_random_direction.py`          | §8 — matched-random direction control |
| `controls_abt_comparison.py`            | §8 — All-but-the-Top vs de-sinking |
| `controls_spectral_gap.py`              | §8 — eigenvalue gap analysis |
| `app_b_probing.py`                      | Appendix B — probing on Pythia-70M (large-scale) |
| `app_b_probing_multi.py`                | Appendix B — probing across three models |
| `app_d_sink_geometry.py`                | Appendix D — sink ↔ unembedding geometry |
| `app_negative_pruning.py`               | Appendix C — layer pruning (negative result) |
| `app_negative_lora.py`                  | Appendix C — LoRA layer selection (inconclusive) |
| `app_negative_cross_cka.py`             | Appendix C — cross-model CKA |
| `app_negative_distillation.py`          | Appendix C — distillation layer matching |

</details>

## Layout

```
desink/         core.py          # de-sinking + spectral metrics (~80 LoC)
experiments/                     # one script per paper artifact (see table above)
paper/          main.tex
                main.pdf
                neurips_2026.sty
                figs/            # vector PDF figures
```

---

Released under the [MIT License](LICENSE).
