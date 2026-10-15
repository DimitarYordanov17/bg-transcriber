"""Real conversational test data from YouTube videos with human-made Bulgarian subtitles (podcasts, interviews).

For a video: a section of audio and its uploaded (not auto-generated) Bulgarian subtitles are downloaded, the cues are
aligned to Whisper's words, and runs of cues that are anchored in the audio AND about as long as the speech (subtitles
are often shortened) become clips with a reference transcript. Subtitles carry no speakers: these clips are for WER.
Cues are not selected by how well Whisper did inside them.

Usage: uv run python bench/youtube.py <video id> <start seconds> <length seconds> <dev|test>
"""
import difflib
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.experiment import cached, sha, store  # noqa: E402
from bench.paths import REAL  # noqa: E402
from bench.score import normalize  # noqa: E402

OUT = REAL / "youtube"
YTDLP = str(Path.home() / ".local/bin/yt-dlp")


def fetch(video: str, start: float, length: float) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    audio = OUT / f"yt_{video}_{int(start)}.flac"
    url = f"https://www.youtube.com/watch?v={video}"
    if not list(OUT.glob(f"yt_{video}.bg*.vtt")):  # the track can be "bg" or "bg-BG"
        subprocess.run([YTDLP, "--skip-download", "--write-subs", "--no-write-auto-subs", "--sub-langs", "bg.*",
                        "--sub-format", "vtt", "-o", str(OUT / f"yt_{video}"), url], check=True, timeout=300)
    subs = sorted(OUT.glob(f"yt_{video}.bg*.vtt"))[0]
    if not audio.exists():
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([YTDLP, "-f", "bestaudio", "--download-sections", f"*{start}-{start + length}",
                            "-o", f"{tmp}/a.%(ext)s", url], check=True, timeout=1800)
            src = next(Path(tmp).glob("a.*"))
            part = audio.with_suffix(".part.flac")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                            "-c:a", "flac", "-fs", "100M", str(part)], check=True)
            part.rename(audio)
    return audio, subs


def cues(vtt: Path, start: float, length: float) -> list[dict]:
    """Subtitle cues inside the audio section, times relative to the section start."""
    out = []
    for block in vtt.read_text().split("\n\n"):
        m = re.search(r"(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)", block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        a, b = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000, g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(block[m.end():].split())
        text = html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
        text = re.sub(r"^[-–]\s*", "", text)
        if start <= a and b <= start + length and text:
            out.append({"start": a - start, "end": b - start, "text": text})
    return out


def keep(cue_list: list[dict], asr_words: list[dict], anchor: int = 2) -> list[dict]:
    ref, owner = [], []
    for ci, c in enumerate(cue_list):
        for tok in normalize(c["text"], True):
            ref.append(tok)
            owner.append(ci)
    hyp = [(tok, k) for k, w in enumerate(asr_words) for tok in normalize(w["text"], True)]
    matched = {}
    for a, b, n in difflib.SequenceMatcher(None, ref, [t for t, _ in hyp], autojunk=False).get_matching_blocks():
        for i in range(n):
            matched[a + i] = b + i
    out = []
    for ci, c in enumerate(cue_list):
        idx = [i for i, o in enumerate(owner) if o == ci]
        if len(idx) < 2:
            continue
        head, tail = [i for i in idx[:anchor] if i in matched], [i for i in idx[-anchor:] if i in matched]
        if not head or not tail:
            continue
        first, last = hyp[matched[head[0]]][1], hyp[matched[tail[-1]]][1]
        spoken = last - first + 1  # Whisper words between the anchors
        if not 0.8 <= len(idx) / max(1, spoken) <= 1.25:  # shortened or padded subtitle
            continue
        out.append({**c, "cue": ci, "start": asr_words[first]["start"], "end": asr_words[last]["end"]})
    return out


def main() -> None:
    video, start, length, split = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    audio, subs = fetch(video, start, length)
    key = sha(audio.read_bytes(), 16) + "_long_whisper"
    if cached("asr", key) is None:
        from modal_app import Transcriber, app
        with app.run():
            store("asr", key, Transcriber(model="large-v3").run_opts.remote(audio.read_bytes(), audio.name,
                                                                            {"beam_size": 5})["segments"])
    words = [w for s in cached("asr", key) for w in s["words"]]
    all_cues = cues(subs, start, length)
    kept = keep(all_cues, words)
    runs, cur = [], []
    for c in kept:
        if cur and (c["cue"] != cur[-1]["cue"] + 1 or c["end"] - cur[0]["start"] > 180):
            runs.append(cur)
            cur = []
        cur.append(c)
    runs.append(cur)
    runs = [r for r in runs if r and r[-1]["end"] - r[0]["start"] >= 30]
    target = REAL / f"youtube_{split}"
    target.mkdir(exist_ok=True)
    total = 0.0
    for r in runs:
        a, b = max(0.0, r[0]["start"] - 0.3), r[-1]["end"] + 0.3
        name = f"yt_{video}_{int(start + a):05d}"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(a), "-t", str(b - a), "-i", str(audio),
                        "-c:a", "flac", str(target / f"{name}.flac")], check=True)
        lines = [f"[{int((c['start'] - a) // 60):02d}:{(c['start'] - a) % 60:04.1f}] A: {c['text']}" for c in r]
        (target / f"{name}.txt").write_text("\n".join(lines) + "\n")
        total += b - a
    print(f"{video}: {len(all_cues)} cues, {len(kept)} kept, {len(runs)} clips, {total / 60:.1f} min -> {target}")


if __name__ == "__main__":
    main()
