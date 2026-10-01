# RCG-Fusion frozen main-experiment protocol (v1)

Frozen on 2026-09-30 for the four currently available datasets. UPMC FOOD-101
is explicitly outside this experiment round at the user's request.

## Confirmatory method

- Backbone: coalition-aware model with 50% full-coalition and 50% uniformly
  sampled non-empty incomplete-coalition training.
- Posterior objective: cross-entropy + 0.1 Brier loss.
- Posterior ensemble: seeds 11, 22, 33, 44, 55; ensemble temperature fitted
  only on the selection split.
- Analytic contribution tolerance: 0.01 nats.
- Final correction:
  `alpha(x) = 0.75 * pi(x)^2` and
  `p_final = (1-alpha) * p_full + alpha * q`.

## Fixed randomness and splits

- Task seeds: 11, 22, 33, 44, 55.
- Corruption seeds: 101, 202, 303.
- MOSI/MOSEI bootstrap unit: original video.
- CREMA-D split and bootstrap unit: actor.
- AV-MNIST bootstrap unit: sample, stratified by class.

## Primary confirmatory comparisons

1. Full coalition versus RCG-Fusion on NLL.
2. Full coalition versus RCG-Fusion on Accuracy and Macro-F1.
3. Analytic contribution versus the best native reliability score on harmful
   contribution AUROC.
4. Adaptive shrinkage versus fixed and confidence-based shrinkage at matched
   clipped-harm budget.

All comparisons use paired predictions. The family of primary comparisons is
Holm corrected. Figures are exported only as white-background, colorblind-safe,
300 dpi PNG files.

## Interpretation rules

- CREMA-D is the cross-task confirmation dataset; its five actor-held-out folds
  are pooled only after each actor has appeared in the test set exactly once.
- Components unsupported by ablation are removed from the claimed innovation.
- If only proper scoring rules improve, the claim is limited to probability
  quality and safer fusion rather than improved classification decisions.
- If the gap holds only on MOSI/MOSEI, the scope is limited to multimodal
  sentiment analysis.
