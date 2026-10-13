"""Build a synthetic multi-speaker conversation from a dialog script.

Spoken lines are rendered beforehand by bench/tts.py (OmniVoice, voices cloned from FLEURS speakers).
Each speaker gets their own recording condition (loud and close, quiet and far, phone line), sound effects
from ESC-50 are dropped in where the script says, and cafeteria noise from DEMAND runs under everything.

Timing: each item starts `gap` seconds after the latest-ending item so far (negative gap = overlap).

Writes into <out>/<name>/: <name>.wav (degraded), <name>_clean.wav (same timing, speech only, equal levels), <name>.txt and <name>.json.

Usage: uv run python bench/make_conv.py --dialog bench/dialogs/conv02.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

warnings.filterwarnings("ignore", "Possible clipped samples")

SR = 16000
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
rng = np.random.default_rng(7)
meter = pyloudnorm.Meter(SR)

LOUDNESS = {"near": -18.0, "far": -36.0, "phone": -25.0}  # LUFS each speaker ends up at
DEFAULT_GAP = 0.35


def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SR:
        g = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    return audio


def frame_db(x: np.ndarray, frame: int) -> np.ndarray:
    rms = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2) + 1e-12) for i in range(0, len(x) - frame, frame)])
    return 20 * np.log10(rms)


def trim_silence(x: np.ndarray, pad: float = 0.05) -> np.ndarray:
    frame = int(0.02 * SR)
    db = frame_db(x, frame)
    active = np.where(db > db.max() - 35)[0]
    start = max(0, active[0] * frame - int(pad * SR))
    end = min(len(x), (active[-1] + 1) * frame + int(pad * SR))
    return x[start:end]


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


def spoken_text(text: str) -> str:
    """Drop TTS tags like [laughter]; they are not words anyone should transcribe."""
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", "", text)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialog", type=Path, required=True)
    ap.add_argument("--sources", type=Path, default=Path("../samples/synthetic/sources"))
    ap.add_argument("--lines", type=Path, help="rendered lines (default: <out>/<name>/lines)")
    ap.add_argument("--out", type=Path, default=Path("../samples/synthetic"))
    args = ap.parse_args()
    name = args.dialog.stem
    out = args.out / name
    lines_dir = args.lines or out / "lines"
    spec = json.loads(args.dialog.read_text())
    sfx_files = sorted((args.sources / "sfx").glob("*.wav"))

    # place every item on the timeline
    placed, cursor = [], 0.0
    for i, item in enumerate(spec["script"]):
        if "speaker" in item:
            audio = trim_silence(load(lines_dir / f"{i:02d}_{item['speaker']}.wav"))
        else:
            options = [f for f in sfx_files if f.name.startswith(item["sfx"] + "_")]
            audio = trim_silence(load(options[rng.integers(len(options))]))
        start = max(0.0, cursor + item.get("gap", DEFAULT_GAP))
        end = start + len(audio) / SR
        placed.append((item, audio, start, end))
        cursor = max(cursor, end)
    total = int((cursor + 0.5) * SR)

    clean = np.zeros(total, dtype=np.float32)
    mix = np.zeros(total, dtype=np.float32)
    turns, events = [], []
    for item, audio, start, end in placed:
        i = int(start * SR)
        if "speaker" in item:
            condition = spec["speakers"][item["speaker"]]["condition"]
            clean[i:i + len(audio)] += set_loudness(audio, -20.0)
            mix[i:i + len(audio)] += degrade(audio, condition)
            turns.append({"speaker": item["speaker"], "start": round(start, 2), "end": round(end, 2),
                          "condition": condition, "text": spoken_text(item["text"])})
        else:
            mix[i:i + len(audio)] += set_loudness(audio, item.get("lufs", -30.0))
            events.append({"sound": item["sfx"], "start": round(start, 2), "end": round(end, 2)})

    noise_spec = spec["noise"]
    noise = load(args.sources / noise_spec["file"])
    offset = rng.integers(0, len(noise) - total)
    mix += set_loudness(noise[offset:offset + total], noise_spec["lufs"])

    peak = np.abs(mix).max()
    if peak > 0.98:
        mix *= 0.98 / peak

    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / f"{name}.wav", mix, SR)
    sf.write(out / f"{name}_clean.wav", clean / max(1.0, np.abs(clean).max() / 0.98), SR)
    txt = "\n".join(f"[{t['start']:05.2f} - {t['end']:05.2f}] Говорител {t['speaker']}: {t['text']}" for t in turns)
    (out / f"{name}.txt").write_text(txt + "\n")
    (out / f"{name}.json").write_text(json.dumps(
        {"duration": round(total / SR, 2), "speakers": spec["speakers"], "noise": noise_spec,
         "turns": turns, "events": events}, ensure_ascii=False, indent=1))
    print(txt)
    for e in events:
        print(f"[{e['start']:05.2f} - {e['end']:05.2f}] ({e['sound']})")
    print(f"duration {total / SR:.1f}s, mix {meter.integrated_loudness(mix):.1f} LUFS", file=sys.stderr)


if __name__ == "__main__":
    main()
