"""Cut one voice reference per speaker from real recordings, for cloning with OmniVoice.

For every file: diarize (exclusive, so no overlapping speech), take each speaker's longest uninterrupted
stretch (3-10 s), cut it from the original audio at 24 kHz and transcribe it with Whisper as the reference text.

Usage: FFMPEG=... uv run python bench/voice_refs.py ../samples/personal/*.flac --out ../samples/voices
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modal_app import Diarizer, Transcriber, app  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
MIN_LEN, MAX_LEN, JOIN_GAP = 3.0, 10.0, 0.5


def longest_stretches(turns: list[dict]) -> dict[str, tuple[float, float]]:
    """Join a speaker's back-to-back turns, then keep the longest stretch per speaker."""
    stretches = []
    for t in sorted(turns, key=lambda t: t["start"]):
        if stretches and stretches[-1][0] == t["speaker"] and t["start"] - stretches[-1][2] <= JOIN_GAP:
            stretches[-1][2] = t["end"]
        else:
            stretches.append([t["speaker"], t["start"], t["end"]])
    best = {}
    for speaker, start, end in stretches:
        if end - start >= MIN_LEN and end - start > best.get(speaker, (0, 0))[1] - best.get(speaker, (0, 0))[0]:
            best[speaker] = (start, end)
    return best


def cut(path: Path, start: float, end: float) -> bytes:
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(path),
           "-ac", "1", "-ar", "24000", "-f", "wav", "-"]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../samples/voices"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    index = []
    with modal.enable_output(), app.run():
        diarizer, transcriber = Diarizer(), Transcriber()
        for path in args.files:
            turns = diarizer.run.remote(path.read_bytes(), path.name, exclusive=True)
            print(f"\n{path.name}: {len({t['speaker'] for t in turns})} speakers")
            for t in turns:
                print(f"  {t['start']:6.2f} - {t['end']:6.2f}  {t['speaker']}")

            for speaker, (start, end) in longest_stretches(turns).items():
                segments = transcriber.run.remote(cut(path, start, end), "ref.wav")
                words = [w for s in segments for w in s["words"]]
                if end - start > MAX_LEN:
                    words = [w for w in words if w["end"] <= MAX_LEN]
                    end = start + words[-1]["end"] + 0.15 if words else start + MAX_LEN
                text = " ".join(w["text"] for w in words)
                name = f"{path.stem}__{speaker}"
                (args.out / f"{name}.wav").write_bytes(cut(path, start, end))
                (args.out / f"{name}.txt").write_text(text + "\n")
                index.append({"voice": name, "source": path.name, "start": start, "end": round(end, 2),
                              "text": text})
                print(f"  -> {name} ({end - start:.1f}s): {text}")

    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
