"""Speech that Whisper skipped. In long recordings Whisper's decoding sometimes drifts: a stretch of real speech (a
roll call in the council session) comes out as one long hallucinated segment ("Благодаря ви, че гледахте.") or as
nothing. The diarizer still hears voices there, so such stretches are found by their word rate and decoded again as
short clips, without the long-form context that caused the drift.
"""
import re

from pipeline.merge import STOCK_HALLUCINATIONS

WINDOW = 20.0  # seconds; long speech regions are checked in pieces of this size
MIN_RATE = 0.7  # words per second; normal speech is 2-3
PAD = 0.3


def _mid(w: dict) -> float:
    return (w["start"] + w["end"]) / 2


def _real_words(words: list[dict]) -> list[dict]:
    """Words that are not part of a stock hallucination phrase."""
    keys = [re.sub(r"[^\w ]", "", w["text"].lower()).strip() for w in words]
    stock = set()
    for i in range(len(words)):
        for phrase in STOCK_HALLUCINATIONS:
            n = len(phrase.split())
            if " ".join(keys[i:i + n]) == phrase:
                stock.update(range(i, i + n))
    return [w for k, w in enumerate(words) if k not in stock]


def find_windows(segments: list[dict], turns: list[dict], min_seconds: float = 2.0) -> list[dict]:
    """Pieces of detected speech where Whisper has (almost) no real words: [{start, end}] with padding."""
    words = _real_words([w for s in segments for w in s["words"]])
    regions = []
    for t in sorted(turns, key=lambda t: t["start"]):
        if regions and t["start"] - regions[-1][1] < 0.3:
            regions[-1][1] = max(regions[-1][1], t["end"])
        else:
            regions.append([t["start"], t["end"]])
    out = []
    for a, b in regions:
        n = max(1, round((b - a) / WINDOW))
        step = (b - a) / n
        for k in range(n):
            start, end = a + k * step, a + (k + 1) * step
            if end - start < min_seconds:
                continue
            inside = sum(1 for w in words if start <= _mid(w) < end)
            if inside / (end - start) < MIN_RATE:
                out.append({"start": round(max(0.0, start - PAD), 2), "end": round(end + PAD, 2)})
    return out


def apply(segments: list[dict], rescued: list[dict], min_confidence: float = 0.5) -> tuple[list[dict], int]:
    """Replace Whisper's words inside each rescued window by the re-decoded ones when the re-decode has more real
    words and is confident enough. Returns the new segments and how many windows were replaced."""
    accepted = []
    for r in rescued:
        a, b = r["start"] + PAD, r["end"] - PAD
        old = _real_words([w for s in segments for w in s["words"] if a <= _mid(w) < b])
        new = _real_words([w for w in r["words"] if a <= _mid(w) < b])
        if len(new) > len(old) and sum(w["confidence"] for w in new) / len(new) >= min_confidence:
            accepted.append((a, b, new))
    if not accepted:
        return segments, 0
    replaced = lambda w: any(a <= _mid(w) < b for a, b, _ in accepted)  # noqa: E731
    out = []
    for s in segments:
        kept = [w for w in s["words"] if not replaced(w)]
        if kept:
            out.append({**s, "words": kept, "text": " ".join(w["text"] for w in kept)})
    for _, _, new in accepted:
        out.append({"start": new[0]["start"], "end": new[-1]["end"], "text": " ".join(w["text"] for w in new),
                    "words": new})
    return sorted(out, key=lambda s: s["start"]), len(accepted)
