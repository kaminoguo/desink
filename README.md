# De-Sinking: Removing the Attention Sink Artifact from Transformer Representations

Code and paper for *Phantom Phases: How Attention Sinks Generate Fictitious Training Dynamics*.

[[Paper]](paper/main.pdf)

## TL;DR

Standard spectral metrics (PCA, effective rank, CKA, RankMe, anisotropy) applied to transformer hidden states are systematically contaminated by the attention sink. De-sinking---projecting out the sink direction---eliminates phantom "geometric phases," reverses apparent rank collapse to rank *growth*, and reveals a hidden shortcut-to-full-rank training transition. Distortions reach 721x in effective rank at 6.9B parameters.

## Quick Start

```python
import torch
from desink import desink, spectral_metrics

# H: hidden states from any transformer layer, shape (n_tokens, d_model)
H_ds = desink(H)

# Or equivalently, in one line:
H_ds = H - (H @ s) * s  # where s is the first right singular vector of centered H

# Compare metrics before and after
print(spectral_metrics(H))     # raw (contaminated)
print(spectral_metrics(H_ds))  # de-sinked (corrected)
```

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```
desink/
  core.py                          # De-sinking implementation + spectral metrics
experiments/
  fig1_training_trajectory.py      # Fig 1a,b: E1/ER across 20 Pythia-70M checkpoints
  fig1c_theorem_fit.py             # Fig 1c: Theorem 1 fit (R^2 = 0.978)
  fig1b_rank_growth_and_correlation.py  # Rank growth + ER-probe anti-correlation
  tab2_contamination_audit.py      # Table 2: 5 metrics x 2 models
  fig3_cka_heatmap.py              # Fig 3: Raw vs de-sinked CKA heatmaps
  tab3_fig4_scale_validation.py    # Table 3 + Fig 4: 70M to 6.9B scale
  fig5a_alignment_peak.py          # Fig 5a: Shortcut-to-full-rank transition
  fig5a_olmo_alignment.py          # Fig 5a: OLMo-2 1B validation
  fig5b_perlayer_departure.py      # Fig 5b: Inverted-U departure profile
  fig6_capability_vs_sink.py       # Fig 6: Probe accuracy vs sink strength
  tab_appE_manufacture_and_qwen.py # Synthetic sink injection + Qwen2.5 GQA
  sink_emergence_checkpoints.py    # Sink formation across training
  controls_*.py                    # Random direction, ABT comparison, spectral gap
  tab_appB_*.py                    # Extended probing results
  tab_appD_sink_geometry.py        # Sink-unembedding geometry
  app_negative_*.py                # Negative results (pruning, LoRA, distillation)
paper/
  main.tex                         # NeurIPS 2026 submission
  neurips_2026.sty
  figs/                            # All figures (PDF vector graphics)
```

## Reproducing Results

Each experiment script is standalone. Example:

```bash
# Figure 1: Phantom phases in Pythia-70M
python experiments/fig1_training_trajectory.py

# Table 2: Contamination audit across 5 metrics
python experiments/tab2_contamination_audit.py

# Scale validation (requires ~40GB GPU memory for Pythia-6.9B)
python experiments/tab3_fig4_scale_validation.py
```

All experiments use publicly available models from Hugging Face (GPT-2, Pythia, OLMo-2, Qwen2.5). Total compute: ~4 GPU-hours on a single A100.

## Citation

```bibtex
@inproceedings{liu2026phantom,
  title={Phantom Phases: How Attention Sinks Generate Fictitious Training Dynamics},
  author={Liu, Yuchen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT
