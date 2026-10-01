# Figure contract

Backend: Python/Matplotlib only. No synthetic observations in research outputs.

Core conclusion under test: high native reliability, quality, or routing scores can coexist with negative conditional contribution in asymmetric or redundant tasks; these scores may therefore fail to identify avoidable harm from forced full fusion. AV-MNIST is retained as a boundary counterexample.

Evidence order:

1. **Figure 1 — Intro motivation:** conceptual distinction; cross-task high-reliability negative-Shapley rate; native-score harm AUROC; fusion-regret ECDF.
2. **Figure 2 — Cross-method HCR:** dataset panels comparing coalition concat, TMC, QMF, PDF, and I²MoE unified-feature adaptations.
3. **Figure 3 — Harm identification:** AUROC/AUPRC of each method's native score for detecting Shapley contribution below −0.01 nats.
4. **Figure 4 — Prediction harm:** regret threshold rates, negative flips/corrections, and oracle coalition composition.
5. **Figure 5 — Robustness:** epsilon sensitivity, deletion versus Shapley, calibration and feature-space pressure tests.

Statistical rules: fixed calibration-set q90 thresholds; five training seeds; CREMA-D actor clusters and MOSI/MOSEI source-video clusters retained in 10,000 paired bootstrap draws; no significance stars; low-coverage estimates are marked descriptive.

Export: **PNG only, 300 dpi, white background, colorblind-friendly Okabe–Ito palette.** Each figure has a matching CSV in `source_data/`. No SVG, PDF or TIFF is generated.
