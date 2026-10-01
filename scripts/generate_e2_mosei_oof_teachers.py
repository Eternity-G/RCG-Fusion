"""Generate the four missing MOSEI OOF teachers for the E2 experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from rcg.rcg_fusion import CoalitionAwareBackbone, nonempty_coalitions
from rcg.rcg_fusion_pipeline import (_inner_split, _take, calibrate_temperatures,
                                     fit_backbone, load_dataset, predict_bundle,
                                     seed_all)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/rcg-fusion-mosei-shrinkage-v1/fold_0/oof_targets.npz"
OUTPUT = ROOT / "runs/formal-e2-mosei-oof"
TEACHER_SEEDS = (11, 22, 33, 44, 55)
FOLDS = 5


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds, names = load_dataset("mosei", ROOT / "data")
    if len(folds) != 1:
        raise ValueError("MOSEI E2 expects one standard outer split")
    split = folds[0]["train"]
    dims = [values.shape[1] for values in split["x"]]
    classes = int(np.max(split["y"])) + 1
    coalitions = nonempty_coalitions(len(names))
    n = len(split["y"])
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with np.load(SOURCE) as saved:
        original = {key: saved[key] for key in saved.files}
    if original["teacher_seeds"].tolist() != [11]:
        raise ValueError("source must contain exactly the existing seed-11 teacher")

    assignment = np.full(n, -1, dtype=np.int16)
    fold_splits = list(GroupKFold(FOLDS).split(
        np.arange(n), split["y"], split["group"]
    ))
    for fold, (_, holdout) in enumerate(fold_splits):
        assignment[holdout] = fold
    if not np.array_equal(assignment, original["fold"]):
        raise ValueError("reconstructed GroupKFold assignment differs from source")

    metadata = []
    for teacher_seed in TEACHER_SEEDS[1:]:
        for fold, (fit_indices, holdout_indices) in enumerate(fold_splits):
            chunk = OUTPUT / f"seed_{teacher_seed}_fold_{fold}.npz"
            meta_path = OUTPUT / f"seed_{teacher_seed}_fold_{fold}.json"
            if chunk.exists() and meta_path.exists():
                metadata.append(json.loads(meta_path.read_text(encoding="utf-8")))
                continue
            train_indices, selection_indices = _inner_split(
                np.asarray(fit_indices), split["group"], 20260927 + fold
            )
            train_split = _take(split, train_indices)
            selection_split = _take(split, selection_indices)
            holdout_split = _take(split, holdout_indices)
            actual_seed = teacher_seed + fold * 1000
            seed_all(actual_seed)
            model = CoalitionAwareBackbone(dims, classes).to(device)
            history = fit_backbone(
                model, train_split, selection_split, seed=actual_seed,
                epochs=100, batch_size=128, patience=10,
            )
            temperatures = calibrate_temperatures(model, selection_split, device)
            prediction = predict_bundle(model, holdout_split, device, temperatures)
            np.savez_compressed(
                chunk, holdout_indices=np.asarray(holdout_indices),
                probabilities=prediction["probabilities"].astype(np.float32),
                losses=prediction["losses"].astype(np.float32),
            )
            record = {
                "fold": fold, "teacher_seed": teacher_seed,
                "actual_seed": actual_seed, "n_train": len(train_indices),
                "n_selection": len(selection_indices), "n_holdout": len(holdout_indices),
                "epochs": len(history),
                "best_selection_nll": min(row["selection_full_nll"] for row in history),
                "device": device,
            }
            meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            metadata.append(record)
            print(
                f"teacher={teacher_seed} fold={fold} epochs={len(history)} "
                f"holdout={len(holdout_indices)}", flush=True,
            )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    probabilities = np.full(
        (len(TEACHER_SEEDS), n, len(coalitions), classes), np.nan, dtype=np.float32
    )
    losses = np.full(
        (len(TEACHER_SEEDS), n, len(coalitions)), np.nan, dtype=np.float32
    )
    probabilities[0] = original["probabilities"][0]
    losses[0] = original["losses"][0]
    for teacher_index, teacher_seed in enumerate(TEACHER_SEEDS[1:], start=1):
        for fold in range(FOLDS):
            with np.load(OUTPUT / f"seed_{teacher_seed}_fold_{fold}.npz") as saved:
                indices = saved["holdout_indices"]
                probabilities[teacher_index, indices] = saved["probabilities"]
                losses[teacher_index, indices] = saved["losses"]
    if not np.isfinite(probabilities).all() or not np.isfinite(losses).all():
        raise RuntimeError("additional OOF teacher coverage is incomplete")
    np.savez_compressed(
        OUTPUT / "oof_targets.npz", probabilities=probabilities, losses=losses,
        fold=assignment, teacher_seeds=np.asarray(TEACHER_SEEDS, dtype=np.int32),
    )
    (OUTPUT / "metadata.json").write_text(
        json.dumps({
            "dataset": "mosei", "source_teacher": str(SOURCE),
            "teacher_seeds": list(TEACHER_SEEDS), "folds": FOLDS,
            "protocol": "same GroupKFold and inner split as rcg_fusion_pipeline",
            "generated": metadata,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"saved {OUTPUT / 'oof_targets.npz'}", flush=True)


if __name__ == "__main__":
    main()
