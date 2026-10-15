"""Subtitle files from transcript lines ({speaker, start, end, text}): SRT and WebVTT, speaker name in front."""


def _stamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _cues(lines: list[dict], max_seconds: float = 7.0, max_words: int = 16) -> list[dict]:
    """Long lines are split so a cue stays readable on screen."""
    cues = []
    for line in lines:
        if line.get("words"):  # real word timings: cut a cue every max_words words or max_seconds seconds
            cur = []
            for text, start, end in line["words"]:
                if cur and (len(cur) >= max_words or end - cur[0][1] > max_seconds):
                    cues.append({"speaker": line["speaker"], "start": cur[0][1], "end": cur[-1][2],
                                 "text": " ".join(t for t, _, _ in cur)})
                    cur = []
                cur.append((text, start, end))
            if cur:
                cues.append({"speaker": line["speaker"], "start": cur[0][1], "end": cur[-1][2],
                             "text": " ".join(t for t, _, _ in cur)})
            continue
        tokens = line["text"].split()
        if not tokens:
            continue
        n = max(1, -(-len(tokens) // max_words), int((line["end"] - line["start"]) // max_seconds) + 1)
        size = -(-len(tokens) // n)
        step = (line["end"] - line["start"]) / n
        for k in range(n):
            part = tokens[k * size:(k + 1) * size]
            if part:
                cues.append({"speaker": line["speaker"], "start": line["start"] + k * step,
                             "end": line["start"] + (k + 1) * step, "text": " ".join(part)})
    return cues


def srt(lines: list[dict]) -> str:
    return "\n".join(f"{i}\n{_stamp(c['start'], ',')} --> {_stamp(c['end'], ',')}\n{c['speaker']}: {c['text']}\n"
                     for i, c in enumerate(_cues(lines), 1))


def vtt(lines: list[dict]) -> str:
    return "WEBVTT\n\n" + "\n".join(f"{_stamp(c['start'], '.')} --> {_stamp(c['end'], '.')}\n<v {c['speaker']}>{c['text']}\n"
                                    for c in _cues(lines))
