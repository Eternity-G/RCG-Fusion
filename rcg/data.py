from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from . import MODALITIES
from .download import HASHES
from .io import sha256, write_json, load_json


def sample_id(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return "$_$".join(str(v) for v in value)
    return str(value)


def video_id(value):
    value = sample_id(value)
    if "$_$" in value:
        return value.split("$_$", 1)[0]
    if "[" in value and value.endswith("]"):
        return value.rsplit("[", 1)[0]
    raise ValueError(f"Cannot infer source video from ID {value!r}; provide a verified parser")


def pool_features(split, modality):
    x = np.asarray(split[modality])
    if not np.issubdtype(x.dtype, np.floating) or x.ndim != 3:
        raise ValueError(f"{modality} must be float [N,T,D], got {x.shape}/{x.dtype}")
    n, length, _ = x.shape
    source = ""
    # Text BERT attention mask is metadata, never use token IDs as feature vectors.
    bert = np.asarray(split.get("text_bert", []))
    if bert.ndim == 3 and bert.shape[0] == n and bert.shape[1] == 3 and bert.shape[2] == length:
        # The chosen file is word-aligned: the same time mask applies to all views.
        mask = bert[:, 1, :].astype(bool).copy()
        # Aligned audio/video have zero placeholders at [CLS]/[SEP]. Exclude these
        # positions in every view to avoid length-dependent dilution of their means.
        for i, row in enumerate(mask):
            indices = np.flatnonzero(row)
            if len(indices):
                if (bert[i, 0, indices[0]], bert[i, 0, indices[-1]]) != (101, 102):
                    raise ValueError("Expected BERT CLS/SEP endpoints (101/102); review alignment before pooling")
                row[indices[[0, -1]]] = False
        source = "aligned BERT attention mask, excluding first/last special-token positions"
    elif f"{modality}_lengths" in split:
        lengths = np.asarray(split[f"{modality}_lengths"]).reshape(-1)
        if len(lengths) != n or np.any((lengths < 0) | (lengths > length)):
            raise ValueError("Invalid sequence lengths")
        mask = np.arange(length)[None, :] < lengths[:, None]
        source = "explicit lengths (right padding)"
    else:
        # Aligned published features have zero padding; preserve interior zero timesteps.
        active = np.any(np.nan_to_num(x) != 0, axis=2)
        mask = np.zeros((n, length), dtype=bool)
        for i, row in enumerate(active):
            idx = np.flatnonzero(row)
            if len(idx):
                mask[i, idx[0]:idx[-1] + 1] = True
        source = "first-to-last nonzero frame fallback; verify padding convention in audit"
    valid = mask & np.isfinite(x).all(axis=2)
    counts = valid.sum(1)
    out = (np.where(valid[:, :, None], x, 0).astype(np.float32).sum(1)
           / np.maximum(counts, 1)[:, None])
    return out.astype(np.float32), {
        "shape": list(x.shape), "mask_source": source,
        "nonfinite_frames_excluded": int((mask & ~np.isfinite(x).all(axis=2)).sum()),
        "empty_sequences": int((counts == 0).sum()),
        "valid_length_min": int(counts.min()), "valid_length_max": int(counts.max()),
    }


def normalize(train, arrays):
    mean = train.mean(0, dtype=np.float64).astype(np.float32)
    std = train.std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return [(x - mean) / std for x in arrays], mean, std


def prepare(dataset, root, split_seed):
    root = Path(root) / dataset
    source = root / "aligned_50.pkl"
    digest = sha256(source)
    if digest != HASHES[dataset]:
        raise ValueError("Feature file is not the published MMSA aligned version; refusing pickle load")
    with open(source, "rb") as f:
        raw = pickle.load(f)
    pooled, audit = {}, {"dataset": dataset, "source_sha256": digest, "split_seed": split_seed}
    for name in ("train", "valid", "test"):
        split = raw[name]
        labels = np.asarray(split["regression_labels"]).reshape(-1)
        if not np.isfinite(labels).all():
            raise ValueError("Non-finite labels")
        keep = labels != 0
        ids = np.array([sample_id(v) for v in split["id"]])
        if len(ids) != len(labels) or len(set(ids)) != len(ids):
            raise ValueError("ID count mismatch or duplicate IDs")
        videos = np.array([video_id(v) for v in ids])
        values = {"y": (labels[keep] > 0).astype(np.int64), "id": ids[keep], "video": videos[keep]}
        report = {"original_n": len(labels), "neutral_excluded": int((~keep).sum()),
                  "n": int(keep.sum()), "videos": len(set(videos[keep])), "modalities": {}}
        for m in MODALITIES:
            x, info = pool_features(split, m)
            if len(x) != len(labels):
                raise ValueError("Feature-label count mismatch")
            values[m] = x[keep]
            report["modalities"][m] = info
        report["class_counts"] = np.bincount(values["y"], minlength=2).tolist()
        pooled[name], audit[name] = values, report
    del raw
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        for field in ("id", "video"):
            if set(pooled[a][field]) & set(pooled[b][field]):
                raise ValueError(f"{field} leakage between {a} and {b}")
    norm = {}
    for m in MODALITIES:
        values, mean, std = normalize(pooled["train"][m], [pooled[s][m] for s in pooled])
        norm[f"{m}_mean"], norm[f"{m}_std"] = mean, std
        for s, x in zip(pooled, values):
            pooled[s][m] = x.astype(np.float32)
    valid = pooled.pop("valid")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=split_seed)
    selection, calibration = next(splitter.split(valid["y"], groups=valid["video"]))
    for name, index in (("selection", selection), ("calibration", calibration)):
        pooled[name] = {k: v[index] for k, v in valid.items()}
        if len(np.unique(pooled[name]["y"])) != 2:
            raise ValueError(f"{name} has one class: cannot calibrate; review group split")
        audit[name] = {"n": len(index), "videos": len(set(valid["video"][index])),
                       "class_counts": np.bincount(valid["y"][index], minlength=2).tolist(),
                       "video_ids": sorted(set(valid["video"][index]))}
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    for name, split in pooled.items():
        np.savez_compressed(out / f"{name}.npz", **split)
    np.savez_compressed(out / "normalization.npz", **norm)
    audit["feature_dims"] = [pooled["train"][m].shape[1] for m in MODALITIES]
    audit["protocol"] = "Nonzero-label binary sentiment, fixed pooled features; normalization fit on train only"
    write_json(out / "audit.json", audit)
    print(f"Prepared {dataset}: " + ", ".join(f"{s}={len(v['y'])}" for s, v in pooled.items()), flush=True)
    return audit


def load_prepared(dataset, root):
    root = Path(root) / dataset / "prepared"
    splits = {}
    for name in ("train", "selection", "calibration", "test"):
        with np.load(root / f"{name}.npz", allow_pickle=False) as f:
            splits[name] = dict(f)
    return splits, load_json(root / "audit.json")
