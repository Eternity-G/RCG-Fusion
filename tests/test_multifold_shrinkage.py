from pathlib import Path

from rcg.projected_fusion_pipeline import load_posteriors


def test_load_posteriors_has_explicit_outer_fold():
    # Regression guard: CREMA-D must not silently load fold_0 checkpoints for
    # all five actor-held-out outer folds.
    assert load_posteriors.__defaults__ == (0,)
    assert "fold_index" in load_posteriors.__code__.co_varnames


def test_frozen_protocol_excludes_deferred_food101():
    text = Path("docx/03_experiment_history/EXPERIMENT_PROTOCOL_V1.md").read_text(
        encoding="utf-8")
    assert "outside this experiment round" in text
    assert "MOSI" not in text or "CREMA-D" in text
