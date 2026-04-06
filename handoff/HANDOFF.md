# Phantom Phases — Handoff to Claude Code

## 文件结构

```
handoff/
├── main.tex                 # 完整NeurIPS论文LaTeX源码（当前用简化style）
├── HANDOFF.md               # 本文件
├── figs_html/               # 5张图的standalone HTML
│   ├── fig1_hero.html       # Figure 1: E1 + ER trajectory + theorem fit (3 panels)
│   ├── fig2_metrics.html    # Figure 2: E1 + rogue + anisotropy bars (3 panels)
│   ├── fig3_cka.html        # Figure 3: CKA heatmaps 2×3 grid
│   ├── fig4_scale.html      # Figure 4: Scale validation ER bars (2 panels)
│   └── fig5_timeline.html   # Figure 5: Probe vs sink timeline (wrapfig size)
```

## TODO（按优先级）

### 1. 图表渲染 → PNG/PDF
每个HTML文件都是self-contained的Chart.js页面，浏览器直接打开就能看。需要转成图片嵌入LaTeX。

方法：
```bash
# 用puppeteer或playwright截图
npx puppeteer-screenshot fig1_hero.html fig1_hero.png --width=1400 --height=400
# 或者直接浏览器打开 → Cmd+Shift+4截图
# 或者用 html2canvas / playwright
```

推荐尺寸：
- fig1, fig2, fig4: 宽1400px，高380-400px（\textwidth三栏）
- fig3: 宽1000px，高~750px（2×3 grid）
- fig5: 宽560px，高380px（wrapfigure用）

### 2. 换NeurIPS官方template
当前main.tex用的是简化style `neurips_2024.sty`。需要：
- 下载官方NeurIPS 2026 template（https://neurips.cc/Conferences/2026/CallForPapers）
- 替换 `\usepackage{neurips_2024}` 为官方包
- 调整页面布局适配官方margin
- 确认正文 ≤ 9页（不含references和appendix）

### 3. 统一数字
main.tex abstract写了"592×"，但D4数据Pythia-6.9B L8实际是721×。
Table 2 (tab:scale) 已经用了721×。需要统一——要么abstract改成721×，要么改成"up to ~700×"。

### 4. 参考文献核实
以下citation需要核实exact信息：
- `li2024geometric`: Li et al. "The geometric phases of neural network training" — 确认正确title、是否发表、venue
- `chen2024condensation`: Chen & He "From condensation to rank collapse" — 确认
- `barbero2025round`: Barbero et al. — 确认title，这篇可能不是关于rotary PE的

### 5. Broader Impact Statement
NeurIPS要求。建议简短写：本文是方法论贡献，揭示测量工具的系统性偏差。正面影响是改善representation analysis的准确性。无明显负面社会影响。

### 6. 补一个 de-sinking 伪代码
论文Section 3.1定义了de-sinking但没给代码。建议加一个Algorithm环境：

```
Algorithm: De-sinking
Input: H ∈ R^{n×d} (hidden states)
1. X ← H - mean(H)           # center
2. U, S, V^T ← SVD(X)        # thin SVD
3. s ← V^T[0]                # first right singular vector = sink direction
4. H_ds ← H - (H·s) s^T     # project out sink
Return: H_ds
```

## 数学模型关键信息

Theorem 1:
$$E_1^{\text{raw}} = \frac{\kappa \alpha^2 + E_1^c}{\kappa \alpha^2 + 1}$$

- κ = 0.0242（fitted on Pythia-70M L3, 20 checkpoints）
- R² = 0.978
- 单一自由参数
- Phase 1 transient偏离理论曲线（因为content genuinely changes）——这支持Phase 1是真实的

ER模型（Corollary，不需要精确fit）:
$$\mathrm{ER}_{\text{raw}} = 1 + \frac{\mathrm{ER}_{\text{ds}} - 1}{1 + k_{\text{er}}(\alpha-1)^2}$$
- k_er = 0.2078, R² = 0.935

## 论文结构速查

| Section | Pages | 核心内容 | 核心figure/table |
|---------|-------|---------|-----------------|
| 1. Intro | 1.5 | Hook + punchline + contributions | — |
| 2. Background | 1 | Sink / metrics / rank collapse literature | — |
| 3. Theory | 2 | De-sinking定义 + Theorem 1 + 4 corollaries + validation | Fig 1 (hero) |
| 4. Scope | 2 | 5 metrics × 2 models + CKA heatmaps + scale validation | Fig 2, 3, 4; Tab 1, 2 |
| 5. True Dynamics | 1.5 | Rank growth + Phase 1 survives + timeline | Fig 5 |
| 6. Mechanism | 0.5 | Causal necessity + informational emptiness + C1 geometry | Tab 3 |
| 7. Controls | 0.3 | Sink vs random direction + two methods agree | — |
| 8. Limits | 0.5 | Engineering robust + limitations | — |
| 9. Discussion | 0.3 | Broader principle + recommendations | — |
| Appendix A | 1 | Full proof of Theorem 1 | — |
| Appendix B | 0.5 | Full probing table | Tab 4 |
| Appendix C | 0.5 | Negative results (F/G/H/I) | — |
| Appendix D | 0.5 | Full C1 geometry table | Tab 5 |

## 叙事核心（给Claude Code看的）

论文的narrative arc是：
1. 大家都信的故事（geometric phases, rank collapse）
2. 一个简单操作就让这个故事消失（de-sinking → phases vanish）
3. 数学证明这必须如此（Theorem 1, R² = 0.978）
4. 影响范围很大（5 metrics, 5 papers, 70M-6.9B）
5. 真实的故事更有趣（rank growth not collapse）
6. 但工程决策naturally robust（honest scoping）

对标论文：Schaeffer et al. "Are Emergent Abilities a Mirage?" (NeurIPS 2023 Outstanding Paper)

## 配色一致性

所有图表统一配色：
- Raw = #c0392b (红)
- De-sinked = #27ae60 (绿)
- Theory curve = #2980b9 (蓝)
- Probe = #8e44ad (紫)
- Sink α = #e67e22 (橙)
- Text primary = #2c2c2a
- Text secondary = #6b6a65
- Grid = rgba(0,0,0,0.06)
