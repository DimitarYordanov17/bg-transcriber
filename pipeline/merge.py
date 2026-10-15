"""Turn ASR words + diarization turns into a speaker-labelled transcript."""
import re


def speaker_of(start: float, end: float, turns: list[dict]) -> str:
    """The speaker whose turn overlaps [start, end] most; the nearest turn if none overlaps."""
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(end, t["end"]) - max(start, t["start"])
        if overlap > best_overlap:
            best, best_overlap = t, overlap
    if best is None:
        mid = (start + end) / 2
        best = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
    return best["speaker"]


def words_with_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Every word gets a speaker. Segments without word timestamps are kept as one block."""
    out = []
    for seg in segments:
        units = seg["words"] or [{"text": seg["text"], "start": seg["start"], "end": seg["end"]}]
        for w in units:
            out.append({**w, "speaker": speaker_of(w["start"], w["end"], turns) if turns else "SPEAKER_00"})
    return out


UNCERTAIN_BELOW = 0.7  # on real speech, Whisper words under this confidence are wrong about half the time


def lines(words: list[dict]) -> list[dict]:
    """Consecutive words of one speaker become one line: {speaker, start, end, text, uncertain}. "uncertain" lists the
    line's low-confidence words (word, confidence) so a reader can be pointed at the spots worth checking."""
    out = []
    for w in words:
        if not (out and out[-1]["speaker"] == w["speaker"]):
            out.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": "", "uncertain": [],
                        "words": []})
        line = out[-1]
        line["end"] = w["end"]
        line["text"] = (line["text"] + " " + w["text"]).strip()
        line["words"].append([w["text"], w["start"], w["end"]])  # for subtitle timing
        if w.get("confidence", 1.0) < UNCERTAIN_BELOW:
            line["uncertain"].append([w["text"], round(w["confidence"], 2)])
    return out


def text_by_speaker(words: list[dict]) -> dict[str, str]:
    by = {}
    for w in words:
        by.setdefault(w["speaker"], []).append(w["text"])
    return {s: " ".join(ws) for s, ws in by.items()}


STOCK_HALLUCINATIONS = ["абонирайте се", "абонирай се", "благодаря за гледането", "благодаря ви за гледането",
                        "субтитри", "не забравяйте да се абонирате", "харесайте видеото", "нашия канал",
                        "благодаря ви че гледахте", "благодаря че гледахте"]


def drop_unvoiced(words: list[dict], turns: list[dict], tolerance: float = 1.0, min_turn: float = 0.2) -> list[dict]:
    """Remove words that lie more than `tolerance` seconds away from any diarized speech: there the diarizer heard no
    voice, so text is most likely an ASR hallucination on silence or noise. Blips shorter than `min_turn` seconds are
    not evidence of a voice. Whisper's stock hallucinations ("Абонирайте се!") are dropped unless they lie inside
    detected speech."""
    turns = [t for t in turns if t["end"] - t["start"] >= min_turn]
    if not turns:  # no diarizer heard any voice at all: nothing was said
        return []
    inside = lambda w: any(t["start"] <= (w["start"] + w["end"]) / 2 <= t["end"] for t in turns)  # noqa: E731
    keys = [re.sub(r"[^\w ]", "", w["text"].lower()).strip() for w in words]
    stock = set()
    for i in range(len(words)):
        for phrase in STOCK_HALLUCINATIONS:
            n = len(phrase.split())
            if " ".join(keys[i:i + n]) == phrase and not any(inside(words[k]) for k in range(i, i + n)):
                stock.update(range(i, i + n))
    words = [w for k, w in enumerate(words) if k not in stock]
    spans = sorted((t["start"] - tolerance, t["end"] + tolerance) for t in turns)
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        if any(a <= mid <= b for a, b in spans):
            out.append(w)
    return out


def smooth_flips(words: list[dict], max_words: int = 1, max_seconds: float = 0.6) -> list[dict]:
    """A single word attributed to someone else in the middle of one speaker's speech ("Къде са те? Къде" / "са" /
    "те?") is almost always a timestamp/boundary error in crosstalk: give it back to the surrounding speaker."""
    runs = []
    for w in words:
        if runs and runs[-1][0] == w["speaker"]:
            runs[-1][1].append(w)
        else:
            runs.append([w["speaker"], [w]])
    for i in range(1, len(runs) - 1):
        speaker, ws = runs[i]
        if (runs[i - 1][0] == runs[i + 1][0] != speaker and len(ws) <= max_words
                and ws[-1]["end"] - ws[0]["start"] <= max_seconds):
            runs[i][0] = runs[i - 1][0]
    return [{**w, "speaker": speaker} for speaker, ws in runs for w in ws]
