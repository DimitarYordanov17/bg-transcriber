"""Real multi-speaker test data from the Bulgarian National Assembly: official stenograms + session audio.

For a session: the stenogram (parliament.bg API) is split into speaker turns, the turns are aligned to Whisper's words
on the session audio, and runs of consecutive well-aligned turns become clips with a reference transcript and speaker
names. A turn is kept when its first and last words are anchored in the audio and most of it aligned; turns are not
selected by how well Whisper did inside them (that would make every system look better than it is).

Usage: uv run python bench/parliament.py <stenogram id> <audio start seconds> <audio length seconds> <dev|test>
"""
import difflib
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.experiment import cached, sha, store  # noqa: E402
from bench.paths import REAL  # noqa: E402
from bench.score import normalize  # noqa: E402

API = "https://www.parliament.bg/api/v1"
OUT = REAL / "parliament"
ABBREVIATIONS = {"чл.": "член", "ал.": "алинея", "т.": "точка", "г.": "година", "№": "номер", "млн.": "милиона",
                 "млрд.": "милиарда", "хил.": "хиляди", "лв.": "лева", "ч.": "часа", "„": "", "“": ""}
CLIP_MAX = 180.0  # seconds


def fetch(sten_id: int, start: float, length: float) -> tuple[Path, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / f"sten_{sten_id}.json"
    for attempt in range(4):  # the API is slow at times
        if meta_path.exists():
            break
        try:
            meta = requests.get(f"{API}/pl-sten/{sten_id}", timeout=120, headers={"User-Agent": "Mozilla/5.0"}).json()
            meta_path.write_text(json.dumps(meta, ensure_ascii=False))
        except requests.exceptions.RequestException:
            if attempt == 3:
                raise
    meta = json.loads(meta_path.read_text())
    audio = OUT / f"parl_{meta['Pl_Sten_date']}_{int(start // 60)}m-{int((start + length) // 60)}m.flac"
    if not audio.exists():  # written under another name first, so an interrupted download never looks complete
        part = audio.with_suffix(".part.flac")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(start), "-i",
                        meta["video"]["default"], "-map", "0:a:0", "-t", str(length), "-ac", "1", "-ar", "16000",
                        "-c:a", "flac", "-fs", "200M", str(part)], check=True, timeout=3600)
        part.rename(audio)
    return audio, meta


def turns_of(meta: dict) -> list[dict]:
    """Speaker turns [{speaker, words}] from the stenogram; stage directions in brackets are removed."""
    text = html.unescape(re.sub(r"<br\s*/?>", "\n", meta["Pl_Sten_body"]))
    text = re.sub(r"<[^>]+>", " ", text)
    label = re.compile(r"\n\s*([А-ЯЁЍ][А-ЯЁЍ \-\.]{3,80})(?:\s*\(([^)\n]{1,40})\))?:\s")
    marks = list(label.finditer("\n" + text))
    out = []
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = ("\n" + text)[m.end():nxt.start() if nxt else None]
        body = re.sub(r"\([^)]*[а-я][^)]*\.\)", " ", body)  # (Шум.) (Ръкопляскания от ПБ.) (Реплика от ...)
        for short, full in ABBREVIATIONS.items():
            body = body.replace(short, full)
        name = re.sub(r"^(ПРЕДСЕДАТЕЛ|ЗАМЕСТНИК-ПРЕДСЕДАТЕЛ)\s+", "", m.group(1).strip())
        if body.split():
            out.append({"speaker": name.title(), "words": body.split()})
    return out


def align_turns(turns: list[dict], asr_words: list[dict], anchor: int = 4, min_aligned: float = 0.6) -> list[dict]:
    """Audio span of every turn that is anchored at both ends: [{speaker, text, start, end}]."""
    ref, owner = [], []  # normalized reference tokens, each with (turn, raw word index)
    for ti, t in enumerate(turns):
        for wi, w in enumerate(t["words"]):
            for tok in normalize(w, True):
                ref.append(tok)
                owner.append((ti, wi))
    hyp, hyp_word = [], []
    for k, w in enumerate(asr_words):
        for tok in normalize(w["text"], True):
            hyp.append(tok)
            hyp_word.append(k)
    matched = {}
    for a, b, n in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks():
        for i in range(n):
            matched[a + i] = b + i
    out = []
    for ti, t in enumerate(turns):
        idx = [i for i, (tt, _) in enumerate(owner) if tt == ti]
        if len(idx) < 3:
            continue
        hits = [i for i in idx if i in matched]
        head, tail = [i for i in idx[:anchor] if i in matched], [i for i in idx[-anchor:] if i in matched]
        if not head or not tail or len(hits) / len(idx) < min_aligned:
            continue
        first, last = head[0], tail[-1]
        a, b = asr_words[hyp_word[matched[first]]]["start"], asr_words[hyp_word[matched[last]]]["end"]
        if b <= a:
            continue
        words = t["words"][owner[first][1]:owner[last][1] + 1]
        out.append({"turn": ti, "speaker": t["speaker"], "text": " ".join(words), "start": a, "end": b})
    return out


def clips(segments: list[dict]) -> list[list[dict]]:
    """Runs of consecutive kept turns (no dropped turn in between), cut at CLIP_MAX seconds."""
    runs, current = [], []
    for s in segments:
        if current and (s["turn"] != current[-1]["turn"] + 1 or s["end"] - current[0]["start"] > CLIP_MAX):
            runs.append(current)
            current = []
        current.append(s)
    if current:
        runs.append(current)
    # a single speech can run 20+ minutes: too long for one clip (and the file size cap would cut it), so skip it
    return [r for r in runs if 20 <= r[-1]["end"] - r[0]["start"] <= 600]


def main() -> None:
    sten_id, start, length, split = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    audio, meta = fetch(sten_id, start, length)
    key = sha(audio.read_bytes(), 16) + "_long_whisper"
    if cached("asr", key) is None:
        from modal_app import Transcriber, app
        with app.run():
            store("asr", key, Transcriber(model="large-v3").run_opts.remote(audio.read_bytes(), audio.name,
                                                                            {"beam_size": 5})["segments"])
    asr_words = [w for s in cached("asr", key) for w in s["words"]]
    turns = turns_of(meta)
    kept = align_turns(turns, asr_words)
    covered = [t["turn"] for t in kept]
    print(f"{audio.name}: {len(turns)} stenogram turns, {len(kept)} kept "
          f"(turns {min(covered) if covered else '-'}-{max(covered) if covered else '-'} of the session)")
    target = REAL / f"parliament_{split}"
    target.mkdir(exist_ok=True)
    total = 0.0
    for run in clips(kept):
        a, b = max(0.0, run[0]["start"] - 0.5), run[-1]["end"] + 0.5
        name = f"parl_{meta['Pl_Sten_date']}_{int(start + a):05d}"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(a), "-t", str(b - a), "-i",
                        str(audio), "-c:a", "flac", "-fs", "20M", str(target / f"{name}.flac")], check=True)
        letters = {}
        lines = []
        for s in run:
            letter = letters.setdefault(s["speaker"], "ABCDEFGHIJ"[len(letters)])
            t = s["start"] - a
            lines.append(f"[{int(t // 60):02d}:{t % 60:04.1f}] {letter}: {s['text']}")
        (target / f"{name}.txt").write_text("\n".join(lines) + "\n")
        (target / f"{name}.json").write_text(json.dumps({"speakers": {v: k for k, v in letters.items()},
                                                         "source": meta["video"]["default"], "offset": start + a,
                                                         "stenogram": sten_id}, ensure_ascii=False))
        total += b - a
    print(f"wrote {len(clips(kept))} clips, {total / 60:.1f} min, to {target}")


if __name__ == "__main__":
    main()
