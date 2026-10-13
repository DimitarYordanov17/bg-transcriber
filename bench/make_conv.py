"""Build one synthetic 3-speaker conversation from clean FLEURS clips.

Each speaker gets their own recording condition (loud and close, quiet and far, phone line),
then cafeteria noise from DEMAND is laid under the whole thing.

Writes conv01.wav (degraded), conv01_clean.wav (same timing, no degradation), conv01.txt and conv01.json.

Usage: python bench/make_conv.py --sources ../samples/synthetic/sources --out ../samples/synthetic
"""
import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore", "Possible clipped samples")

SR = 16000
SRC = Path("../samples/synthetic/sources")
OUT = Path("../samples/synthetic")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
rng = np.random.default_rng(7)
meter = pyloudnorm.Meter(SR)

SPEAKERS = {"A": "near", "B": "far", "C": "phone"}
LOUDNESS = {"near": -18.0, "far": -36.0, "phone": -25.0}  # LUFS each speaker ends up at
NOISE_LUFS = -40.0

# speaker, source clip, which sentence (None = whole clip), pause before the turn (negative = overlap)
TURNS = [
    ("A", "10145399168975604452", 0, 0.0),
    ("B", "10607669002366733860", None, 0.5),
    ("C", "11290732792982580381", None, 0.35),
    ("A", "10145399168975604452", 1, -0.3),
]


def load(name: str) -> np.ndarray:
    audio, sr = sf.read(SRC / name, dtype="float32")
    assert sr == SR, (name, sr)
    return audio if audio.ndim == 1 else audio[:, 0]


def frame_db(x: np.ndarray, frame: int) -> np.ndarray:
    rms = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2) + 1e-12) for i in range(0, len(x) - frame, frame)])
    return 20 * np.log10(rms)


def trim_silence(x: np.ndarray, pad: float = 0.1) -> np.ndarray:
    frame = int(0.02 * SR)
    db = frame_db(x, frame)
    active = np.where(db > db.max() - 35)[0]
    start = max(0, active[0] * frame - int(pad * SR))
    end = min(len(x), (active[-1] + 1) * frame + int(pad * SR))
    return x[start:end]


def split_at_pause(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut at the quietest 300 ms in the middle half of the clip (the pause between two sentences)."""
    frame = int(0.02 * SR)
    db = frame_db(x, frame)
    window = 15
    smooth = np.convolve(db, np.ones(window) / window, mode="same")
    lo, hi = len(db) // 4, 3 * len(db) // 4
    cut = (lo + int(np.argmin(smooth[lo:hi]))) * frame
    return trim_silence(x[:cut]), trim_silence(x[cut:])


def set_loudness(x: np.ndarray, lufs: float) -> np.ndarray:
    return pyloudnorm.normalize.loudness(x, meter.integrated_loudness(x), lufs)


def room_ir(rt60: float) -> np.ndarray:
    """Synthetic room response: exponentially decaying noise."""
    t = np.arange(int(rt60 * SR)) / SR
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    return tail / np.sqrt(np.sum(tail ** 2))


def phone_line(x: np.ndarray) -> np.ndarray:
    """300-3400 Hz band, 8 kHz G.711 mu-law, back to 16 kHz. Same chain as a landline call."""
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
           "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav", "-"]
    mulaw = subprocess.run(cmd, input=x.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-i", "-", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, input=mulaw, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)[:len(x)]


ROOMS = {"near": (room_ir(0.25), 0.15), "far": (room_ir(0.7), 0.9)}


def degrade(x: np.ndarray, condition: str) -> np.ndarray:
    if condition in ROOMS:
        ir, wet = ROOMS[condition]
        x = x + wet * fftconvolve(x, ir)[:len(x)]
    elif condition == "phone":
        x = phone_line(x)
    return set_loudness(x, LOUDNESS[condition])


def main() -> None:
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    SRC, out = args.sources, args.out

    clips, texts = [], []
    for _, src, part, _ in TURNS:
        audio = trim_silence(load(f"{src}.wav"))
        text = (SRC / f"{src}.txt").read_text().strip()
        if part is not None:
            audio = split_at_pause(audio)[part]
            sentences = text.replace(". ", ".\n").split("\n")
            text = sentences[part]
        clips.append(audio)
        texts.append(text)

    starts, t = [], 0.0
    for (_, _, _, pause), clip in zip(TURNS, clips):
        t += pause
        starts.append(t)
        t += len(clip) / SR
    total = int((max(s + len(c) / SR for s, c in zip(starts, clips)) + 0.5) * SR)

    clean = np.zeros(total, dtype=np.float32)
    mix = np.zeros(total, dtype=np.float32)
    truth = []
    for (speaker, src, _, _), clip, text, start in zip(TURNS, clips, texts, starts):
        condition = SPEAKERS[speaker]
        i = int(start * SR)
        clean[i:i + len(clip)] += set_loudness(clip, -20.0)
        mix[i:i + len(clip)] += degrade(clip, condition)
        truth.append({"speaker": speaker, "start": round(start, 2), "end": round(start + len(clip) / SR, 2),
                      "condition": condition, "lufs": LOUDNESS[condition], "source": f"fleurs bg_bg dev {src}.wav",
                      "text": text})

    noise = load("demand_pcafeter_ch01.wav")
    offset = rng.integers(0, len(noise) - total)
    mix += set_loudness(noise[offset:offset + total], NOISE_LUFS)

    peak = np.abs(mix).max()
    if peak > 0.98:
        mix *= 0.98 / peak

    sf.write(out / "conv01.wav", mix, SR)
    sf.write(out / "conv01_clean.wav", clean / max(1.0, np.abs(clean).max() / 0.98), SR)
    (out / "conv01.txt").write_text("\n".join(
        f"[{t['start']:05.2f} - {t['end']:05.2f}] Говорител {t['speaker']}: {t['text']}" for t in truth) + "\n")
    (out / "conv01.json").write_text(json.dumps(
        {"duration": round(total / SR, 2), "noise": {"source": "DEMAND PCAFETER ch01", "lufs": NOISE_LUFS},
         "turns": truth}, ensure_ascii=False, indent=1))
    print((out / "conv01.txt").read_text(), end="")
    print(f"duration {total / SR:.1f}s, mix {meter.integrated_loudness(mix):.1f} LUFS", file=sys.stderr)


if __name__ == "__main__":
    main()
