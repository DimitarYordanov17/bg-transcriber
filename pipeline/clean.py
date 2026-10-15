"""Last-step cleanup of client-visible garbage that ASR produces on long real recordings.

- broken characters (U+FFFD) are removed;
- words containing letters of other scripts (Tamil, Arabic, CJK...) are dropped;
- single words mixing Latin and Cyrillic letters ("dominantазва") are dropped;
- pure-Latin words are kept only if at least `latin_support` other ASR systems also heard that word somewhere in the
  file: real names/terms ("Facebook") are recognized by several systems, Whisper's long-form hallucinations
  ("teaspoon", "Songs by the Angular") are not;
- a phrase of 4+ words repeated back-to-back by the same speaker is kept once (hallucination loops);
- "Р. България" is written out as "Република България" (what was said);
- one word repeated 10+ times in a row ("стой, стой, ..." x50) is cut to 3. Shorter runs stay: "Тук." x6 is a roll call.
"""
import re

REPLACEMENT_CHAR = "�"


def _letters_ok(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    if any(not ("а" <= c.lower() <= "я" or c.lower() == "ѝ" or c.isascii()) for c in letters):
        return False
    return not (any(c.isascii() for c in letters) and any(not c.isascii() for c in letters))


def _pure_latin(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    return bool(letters) and all(c.isascii() for c in letters)


def _key(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _expand_abbreviations(words: list[dict]) -> list[dict]:
    """Whisper writes "Р. България" where the speaker said "Република България": write what was said."""
    out = []
    for i, w in enumerate(words):
        nxt = words[i + 1]["text"] if i + 1 < len(words) else ""
        if w["text"] in ("Р.", "р.") and nxt.startswith("България"):
            w = {**w, "text": "Република"}
        out.append(w)
    return out


def _cap_word_runs(words: list[dict], max_run: int = 10, keep: int = 3) -> list[dict]:
    out, i = [], 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and _key(words[j + 1]["text"]) == _key(words[i]["text"]) and _key(words[i]["text"]):
            j += 1
        run = words[i:j + 1]
        out += run[:keep] if len(run) >= max_run else run
        i = j + 1
    return out


def clean(words: list[dict], others: dict[str, list[str]] | None = None, latin_support: int = 2,
          min_phrase: int = 4) -> list[dict]:
    heard = {}
    for toks in (others or {}).values():
        for k in {_key(t) for t in toks if _pure_latin(t)}:
            heard[k] = heard.get(k, 0) + 1
    out = []
    for w in words:
        text = w["text"].replace(REPLACEMENT_CHAR, "")
        if not text.strip() or not _letters_ok(text):
            continue
        if others is not None and _pure_latin(text) and heard.get(_key(text), 0) < latin_support:
            continue
        out.append({**w, "text": text})
    out = _expand_abbreviations(out)
    out = _cap_word_runs(out)
    i, result = 0, []
    while i < len(out):
        for n in range(12, min_phrase - 1, -1):  # longest loop first
            a, b = out[i:i + n], out[i + n:i + 2 * n]
            # same words back to back, even if the speaker label changed in between (a loop spread over speakers)
            if len(b) == n and all(_key(x["text"]) == _key(y["text"]) for x, y in zip(a, b)) \
                    and any(_key(x["text"]) for x in a):
                del out[i + n:i + 2 * n]
                break
        else:
            result.append(out[i])
            i += 1
    return result


SOUND_TAGS = {"аплодисменти", "аплодисмента", "смях", "музика", "шум", "кашлица", "смее", "ръкопляскане"}


def format_sound_tags(text: str) -> str:
    """Whisper writes sound events as capitalized words ("АПЛОДИСМЕНТИ"); show them as [аплодисменти]. Only a fixed
    list, so real acronyms (ЕГН, КАТ) are never touched. Repeated tags collapse into one."""
    out = []
    for token in text.split():
        core = re.sub(r"[^\w]", "", token)
        if core.isupper() and core.lower() in SOUND_TAGS:
            tag = f"[{core.lower()}]"
            if not out or out[-1] != tag:
                out.append(tag)
        else:
            out.append(token)
    return " ".join(out)
