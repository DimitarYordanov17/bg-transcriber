"""Draft transcript of a real clip for hand correction: Whisper words, each given to the pyannote speaker it overlaps.

Usage: uv run python bench/draft.py ../data/real/unlabeled/<clip>.flac [more clips...]
Writes <clip>.draft.txt next to the audio, one line per turn: "[mm:ss] A: text". Correct it and save as <clip>.txt.
"""
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modal_app import Diarizer, Transcriber, app  # noqa: E402


def speaker_of(word: dict, turns: list[dict]) -> str:
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(word["end"], t["end"]) - max(word["start"], t["start"])
        if overlap > best_overlap:
            best, best_overlap = t, overlap
    if best is None:
        mid = (word["start"] + word["end"]) / 2
        best = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
    return best["speaker"]


def draft(segments: list[dict], turns: list[dict]) -> str:
    letters, lines = {}, []
    for word in (w for s in segments for w in s["words"]):
        letter = letters.setdefault(speaker_of(word, turns), chr(ord("A") + len(letters)))
        if lines and lines[-1][0] == letter:
            lines[-1][2].append(word["text"])
        else:
            lines.append([letter, word["start"], [word["text"]]])
    return "\n".join(f"[{int(t // 60):02d}:{t % 60:04.1f}] {s}: {' '.join(ws)}" for s, t, ws in lines) + "\n"


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    with modal.enable_output(), app.run():
        inputs = [(p.read_bytes(), p.name) for p in paths]
        turn_calls = [Diarizer().run.spawn(data, name, exclusive=True) for data, name in inputs]
        transcripts = list(Transcriber().run.starmap(inputs))
        all_turns = [call.get() for call in turn_calls]
    for path, segments, turns in zip(paths, transcripts, all_turns):
        out = path.with_suffix(".draft.txt")
        out.write_text(draft(segments, turns))
        print(f"{path.name}: {len({t['speaker'] for t in turns})} speakers -> {out.name}")


if __name__ == "__main__":
    main()
