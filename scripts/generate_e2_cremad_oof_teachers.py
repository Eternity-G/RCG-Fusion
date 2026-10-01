"""Generate missing multi-teacher OOF records inside each CREMA-D outer fold."""
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
SOURCE = ROOT / "runs/rcg-fusion-cremad-v1"
OUTPUT = ROOT / "runs/formal-e2-cremad-oof"
TEACHER_SEEDS = (11, 22, 33, 44, 55)
INNER_FOLDS = 5


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outer_folds, names = load_dataset("cremad", ROOT / "data")
    if len(outer_folds) != 5:
        raise ValueError("CREMA-D E2 expects five actor-held-out outer folds")
    metadata = []
    for outer_index, splits in enumerate(outer_folds):
        split = splits["train"]
        dims = [values.shape[1] for values in split["x"]]
        classes = int(np.max(split["y"])) + 1
        coalitions = nonempty_coalitions(len(names))
        n = len(split["y"])
        fold_output = OUTPUT / f"fold_{outer_index}"
        fold_output.mkdir(parents=True, exist_ok=True)
        with np.load(SOURCE / f"fold_{outer_index}/oof_targets.npz") as saved:
            original = {key: saved[key] for key in saved.files}
        if original["teacher_seeds"].tolist() != [11]:
            raise ValueError("source must contain exactly the existing seed-11 teacher")

        assignment = np.full(n, -1, dtype=np.int16)
        inner_splits = list(GroupKFold(INNER_FOLDS).split(
            np.arange(n), split["y"], split["group"]
        ))
        for inner_fold, (_, holdout) in enumerate(inner_splits):
            assignment[holdout] = inner_fold
        if not np.array_equal(assignment, original["fold"]):
            raise ValueError(f"outer fold {outer_index}: inner GroupKFold differs from source")

        for teacher_seed in TEACHER_SEEDS[1:]:
            for inner_fold, (fit_indices, holdout_indices) in enumerate(inner_splits):
                chunk = fold_output / f"seed_{teacher_seed}_inner_{inner_fold}.npz"
                meta_path = fold_output / f"seed_{teacher_seed}_inner_{inner_fold}.json"
                if chunk.exists() and meta_path.exists():
                    metadata.append(json.loads(meta_path.read_text(encoding="utf-8")))
                    continue
                train_indices, selection_indices = _inner_split(
                    np.asarray(fit_indices), split["group"],
                    20260927 + inner_fold,
                )
                train_split = _take(split, train_indices)
                selection_split = _take(split, selection_indices)
                holdout_split = _take(split, holdout_indices)
                actual_seed = teacher_seed + inner_fold * 1000
                seed_all(actual_seed)
                model = CoalitionAwareBackbone(dims, classes).to(device)
                history = fit_backbone(
                    model, train_split, selection_split, seed=actual_seed,
                    epochs=100, batch_size=64, patience=10,
                )
                temperatures = calibrate_temperatures(model, selection_split, device)
                prediction = predict_bundle(model, holdout_split, device, temperatures)
                np.savez_compressed(
                    chunk, holdout_indices=np.asarray(holdout_indices),
                    probabilities=prediction["probabilities"].astype(np.float32),
                    losses=prediction["losses"].astype(np.float32),
                )
                record = {
                    "outer_fold": outer_index, "inner_fold": inner_fold,
                    "teacher_seed": teacher_seed, "actual_seed": actual_seed,
                    "n_train": len(train_indices), "n_selection": len(selection_indices),
                    "n_holdout": len(holdout_indices), "epochs": len(history),
                    "best_selection_nll": min(
                        row["selection_full_nll"] for row in history
                    ),
                    "device": device,
                }
                meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                metadata.append(record)
                print(
                    f"outer={outer_index} teacher={teacher_seed} inner={inner_fold} "
                    f"epochs={len(history)} holdout={len(holdout_indices)}",
                    flush=True,
                )
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

        probabilities = np.full(
            (len(TEACHER_SEEDS), n, len(coalitions), classes),
            np.nan, dtype=np.float32,
        )
        losses = np.full(
            (len(TEACHER_SEEDS), n, len(coalitions)), np.nan, dtype=np.float32,
        )
        probabilities[0], losses[0] = original["probabilities"][0], original["losses"][0]
        for teacher_index, teacher_seed in enumerate(TEACHER_SEEDS[1:], start=1):
            for inner_fold in range(INNER_FOLDS):
                with np.load(
                    fold_output / f"seed_{teacher_seed}_inner_{inner_fold}.npz"
                ) as saved:
                    indices = saved["holdout_indices"]
                    probabilities[teacher_index, indices] = saved["probabilities"]
                    losses[teacher_index, indices] = saved["losses"]
        if not np.isfinite(probabilities).all() or not np.isfinite(losses).all():
            raise RuntimeError(f"outer fold {outer_index}: incomplete OOF coverage")
        np.savez_compressed(
            fold_output / "oof_targets.npz", probabilities=probabilities,
            losses=losses, fold=assignment,
            teacher_seeds=np.asarray(TEACHER_SEEDS, dtype=np.int32),
        )

    (OUTPUT / "metadata.json").write_text(
        json.dumps({
            "dataset": "cremad", "source": str(SOURCE),
            "teacher_seeds": list(TEACHER_SEEDS), "outer_folds": 5,
            "inner_folds": INNER_FOLDS,
            "grouping": "actors at both outer and inner split levels",
            "generated": metadata,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"saved five outer-fold OOF target files under {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
