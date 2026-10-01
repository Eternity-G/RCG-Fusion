"""A synthetic fixture tests the software only; it is never a research result."""
import numpy as np

from rcg.analyze import analyze
from rcg.evaluate import evaluate_seed, METHODS
from rcg.io import load_json, write_json
from rcg.train import train_seed, load_models
from rcg.coalition import evaluate_coalitions, analyze_coalitions


def test_small_end_to_end_pipeline(tmp_path):
    import torch
    import pandas as pd
    torch.set_num_threads(1)
    cfg = load_json("config.json")
    cfg.update(epochs=2, patience=2, seeds=[7], corruption_seeds=[101], noise_levels=[.5],
               mask_levels=[.5], bootstrap=30, hidden=8)
    rng = np.random.default_rng(21)
    splits = {}
    for name in ("train", "selection", "calibration", "test"):
        y = np.tile([0, 1], 24)
        splits[name] = {"y": y, "id": np.array([f"{name}_{i//4}$_${i}" for i in range(48)]),
                        "video": np.array([f"{name}_{i//4}" for i in range(48)])}
        for m, dim in zip(("text", "audio", "vision"), (6, 3, 4)):
            splits[name][m] = (rng.normal(size=(48, dim)) + y[:, None]*.5).astype(np.float32)
    out = tmp_path / "seed_7"
    models, temps = train_seed(splits, cfg, 7, out, "cpu")
    loaded, loaded_temps = load_models(out / "checkpoint.pt", cfg, "cpu")
    assert temps == loaded_temps
    evaluate_seed(loaded, temps, splits, cfg, 7, out, "cpu")
    for method in METHODS:
        frame = pd.read_parquet(out / f"{method}.parquet")
        assert np.isfinite(frame.u).all()
        assert not ((frame.kind == "missing") & (frame.modality == frame.affected)).any()
        np.testing.assert_allclose(frame.u, frame.loss_without-frame.loss_full)
    write_json(tmp_path / "manifest.json", {"dataset": "synthetic_test_only", "config": cfg})
    analyze(tmp_path, cfg)
    assert (tmp_path / "REPORT.md").exists()
    intervals = pd.read_csv(tmp_path / "analysis" / "cluster_intervals.csv")
    assert intervals.n_videos.eq(12).all()
    assert intervals.descriptive_only.any()

    coalition_run = tmp_path / "coalition"
    evaluate_coalitions(loaded, temps, splits, cfg, "synthetic_test_only", 7,
                        coalition_run / "seed_7", "cpu")
    analyze_coalitions(coalition_run, cfg)
    coalition = pd.read_parquet(coalition_run / "seed_7" / "coalitions.parquet")
    clean = coalition[(coalition.family == "main") & (coalition.kind == "clean")]
    assert len(clean) == len(splits["test"]["y"])
    assert (clean.fusion_regret >= -1e-7).all()
    np.testing.assert_allclose(
        clean[["shapley_contribution_text", "shapley_contribution_audio", "shapley_contribution_vision"]].sum(1),
        clean.loss_EMPTY-clean.loss_TAV,
        atol=1e-6,
    )
    assert (coalition_run / "REPORT.md").exists()
