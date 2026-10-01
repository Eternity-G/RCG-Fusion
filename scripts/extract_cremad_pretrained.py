"""Extract resumable frozen Wav2Vec2-Base and R3D-18 CREMA-D features.

The script writes one small NPZ per clip first, so interruption never destroys
completed work.  A final consolidated NPZ is created only after all 7,442
audio/video pairs have passed validation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
import torchaudio
from PIL import Image
from torchvision.models.video import R3D_18_Weights, r3d_18


EMOTIONS = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}


def inventory(root: Path):
    wav = {p.stem: p for p in (root / "AudioWAV").glob("*.wav")}
    video = {p.stem: p for p in (root / "VideoFlash").glob("*.flv")}
    ids = sorted(set(wav) & set(video))
    if len(ids) != 7442 or set(wav) != set(video):
        raise ValueError(f"Expected 7,442 name-matched pairs; wav={len(wav)}, video={len(video)}, pairs={len(ids)}")
    rows = []
    for sample_id in ids:
        fields = sample_id.split("_")
        if len(fields) != 4 or fields[2] not in EMOTIONS:
            raise ValueError(f"Unexpected CREMA-D filename: {sample_id}")
        rows.append((sample_id, int(fields[0]), EMOTIONS[fields[2]], wav[sample_id], video[sample_id]))
    if len({x[1] for x in rows}) != 91:
        raise ValueError("Expected 91 actors")
    return rows


def decode_uniform(path: Path, count: int = 16) -> torch.Tensor:
    """Decode a clip and return T,C,H,W uint8 frames sampled uniformly."""
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    width, height = meta["size"]
    frames = [np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy() for buf in reader]
    if not frames:
        raise ValueError(f"No frames decoded: {path}")
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
    sampled = np.stack([frames[i] for i in indices])
    return torch.from_numpy(sampled).permute(0, 3, 1, 2)


def audio_embedding(model, path: Path, device: str) -> np.ndarray:
    # Python's wave reader avoids optional TorchCodec/FFmpeg audio backends.
    with wave.open(str(path), "rb") as stream:
        sample_rate = stream.getframerate(); channels = stream.getnchannels()
        width = stream.getsampwidth(); raw = stream.readframes(stream.getnframes())
    if width == 2:
        signal = np.frombuffer(raw, dtype="<i2").astype(np.float32)/32768.0
    elif width == 1:
        signal = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32)-128)/128.0
    else:
        raise ValueError(f"Unsupported WAV sample width {width}: {path}")
    waveform = torch.from_numpy(signal.reshape(-1, channels).mean(1)).unsqueeze(0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    waveform = waveform.to(device)
    with torch.inference_mode():
        layers, lengths = model.extract_features(waveform)
        feature = layers[-1][0]
        length = int(lengths[0]) if lengths is not None else len(feature)
        pooled = feature[:length].mean(0)
    return pooled.float().cpu().numpy()


def video_embedding(model, transform, path: Path, device: str) -> np.ndarray:
    frames = decode_uniform(path)
    # Official weights transform: uint8 T,C,H,W -> normalized float C,T,H,W.
    tensor = transform(frames).unsqueeze(0).to(device)
    with torch.inference_mode():
        pooled = model(tensor)[0]
    return pooled.float().cpu().numpy()


def extract(repository: Path, output: Path, device: str, limit: int | None = None):
    rows = inventory(repository)
    if limit:
        rows = rows[:limit]
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "clips"
    cache.mkdir(exist_ok=True)
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"

    audio_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model().eval().to(device)
    weights = R3D_18_Weights.KINETICS400_V1
    video_model = r3d_18(weights=weights)
    video_model.fc = torch.nn.Identity()
    video_model = video_model.eval().to(device)
    transform = weights.transforms()

    failures = []
    for index, (sample_id, actor, label, wav, video) in enumerate(rows, 1):
        target = cache / f"{sample_id}.npz"
        if target.exists():
            try:
                with np.load(target) as old:
                    if old["audio"].shape == (768,) and old["visual"].shape == (512,):
                        continue
            except Exception:
                pass
        try:
            audio = audio_embedding(audio_model, wav, device)
            visual = video_embedding(video_model, transform, video, device)
            np.savez_compressed(target, audio=audio, visual=visual,
                                label=np.int16(label), actor=np.int16(actor))
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": repr(exc)})
        if index == 1 or index % 25 == 0:
            print(f"CREMA-D frozen features {index}/{len(rows)}; failures={len(failures)}", flush=True)
        if failures:
            (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    if failures:
        raise RuntimeError(f"Feature extraction had {len(failures)} failures; see {output/'failures.json'}")
    if limit:
        return
    records = []
    for sample_id, actor, label, _, _ in inventory(repository):
        with np.load(cache / f"{sample_id}.npz") as item:
            records.append((sample_id, actor, label, item["audio"], item["visual"]))
    np.savez_compressed(
        output / "cremad_wav2vec2_r3d18.npz",
        sample_id=np.array([x[0] for x in records]),
        actor=np.array([x[1] for x in records], dtype=np.int16),
        label=np.array([x[2] for x in records], dtype=np.int8),
        audio=np.stack([x[3] for x in records]).astype(np.float32),
        visual=np.stack([x[4] for x in records]).astype(np.float32),
    )
    audit = {
        "samples": len(records), "actors": len({x[1] for x in records}),
        "audio": "torchaudio WAV2VEC2_BASE (LibriSpeech 960h), final transformer layer mean, 768D",
        "visual": "torchvision R3D_18 KINETICS400_V1, 16 uniform frames, official transform, 512D",
        "label": "six intended-emotion classes parsed from filename",
        "device": device,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("data/cremad/repository"))
    parser.add_argument("--output", type=Path, default=Path("data/cremad/pretrained"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    extract(args.repository, args.output, args.device, args.limit)


if __name__ == "__main__":
    main()
