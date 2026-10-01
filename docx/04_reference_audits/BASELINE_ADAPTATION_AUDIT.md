# Strong-baseline adaptation audit

This audit records exactly what is preserved and changed in the unified frozen-feature diagnostic. The resulting numbers are not direct reproductions of the original benchmark tables.

| Method | Official source inspected | Preserved mechanism and native score | Necessary adaptation |
|---|---|---|---|
| TMC | Existing author-code adaptation in this repository | Nonnegative Dirichlet evidence, Dempster–Shafer combination, native score `1-K/sum(alpha)` | Dataset encoders replaced by frozen-feature MLP heads; explicit coalition masks |
| QMF | `external/QMF/text-image-classification` | Per-modality energy `logsumexp(logits)/10`, confidence-weighted logits, loss-ranking supervision | BERT/image encoders replaced by MLP projections; generalized to 2/3 modalities; coalition dropout |
| PDF | `external/PDF/src/models/latefusion_pdf.py` | ConfidNet Mono-Confidence, Holo-Confidence, Co-Belief, dynamic-uncertainty correction | Feature dimensions replaced by shared 128D projections; Holo-Confidence generalized from two to all available modalities; coalition dropout |
| I²MoE | `external/I2MoE/src/imoe/InteractionMoE.py` | Modality uniqueness experts plus synergy/redundancy experts, MLP routing, replacement-based interaction regularizers | Dataset fusion backbone replaced by coalition-aware MLP experts over frozen features; first uniqueness-route weights used as modality diagnostic scores |

## Common controls

- All methods receive identical normalized frozen representations and identical train/model-selection/calibration/test samples.
- Every method sees a complete coalition in 50% of training rows; the other 50% uniformly sample nonempty incomplete coalitions.
- All coalition logits are temperature-scaled only with the calibration split before loss, Shapley and regret calculation.
- Each method's native score threshold is the corresponding calibration-set 90th percentile. The score is not retuned on test corruptions.
- For the maximum-probability baseline, modality logits receive their own calibration temperatures. QMF, PDF, TMC and I²MoE retain their published native scoring forms.
- Test labels are used only after inference to calculate observed loss, contribution, Shapley value, regret and evaluation metrics.

## Interpretation limit

The comparison asks whether the mechanism represented by each native score identifies harmful modality addition under a controlled common feature protocol. It does not claim that the adaptation matches every architecture, optimizer or dataset-specific encoder used in the original paper.
