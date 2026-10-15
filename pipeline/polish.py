"""Final text passes on transcript lines ({speaker, start, end, text}), after all word-level steps.

punctuate: the LLM may only add punctuation and fix capital letters; a line is replaced only if its words are exactly
the same. readable: a light editor pass for reading (sentence breaks, repeated phrases and false starts removed,
agreement fixed); a line is replaced only if it stays close to the original, so content cannot be rewritten.
"""
import difflib
import re

from pipeline.align import key

PUNCTUATE = """Това е автоматична транскрипция на разговор на български. Постави препинателните знаци и главните букви \
правилно: точки, запетаи, въпросителни, главна буква в началото на изречение и при имена. Не променяй, не добавяй и \
не махай нито една дума и не сменяй реда им.

Редове:
{lines}

Върни всички редове със същите номера, по един на ред, във формат "[N] текст". Нищо друго."""

READABLE = """Това е автоматична транскрипция на разговор на български. Направи всеки ред четим, като редактор на \
протокол:
- постави правилно препинателните знаци и главните букви;
- махни повторени фрази, запъвания и започнати и изоставени думи;
- поправи очевидни граматически грешки (съгласуване, окончания), които са грешки на разпознаването;
- не перифразирай, не съкращавай съдържанието, не добавяй нищо, което не е казано, не сменяй имена.

Редове:
{lines}

Върни всички редове със същите номера, по един на ред, във формат "[N] текст". Нищо друго."""


def _keys(text: str) -> list[str]:
    return [k for k in (key(t) for t in text.split()) if k]


def _chunks(lines: list[dict], max_words: int = 250) -> list[list[int]]:
    out, cur, size = [], [], 0
    for i, line in enumerate(lines):
        n = len(line["text"].split())
        if cur and size + n > max_words:
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    return out + [cur] if cur else out


def _parse(reply: str) -> dict[int, str]:
    out = {}
    for line in (reply or "").splitlines():
        m = re.match(r"\s*\[(\d+)\]\s*(.*)$", line)
        if m and m.group(2).strip():
            out[int(m.group(1))] = m.group(2).strip()
    return out


def _run(lines: list[dict], generate, prompt: str, accept) -> tuple[list[dict], int]:
    chunks = _chunks(lines)
    prompts = [prompt.format(lines="\n".join(f"[{n}] {lines[i]['text']}" for n, i in enumerate(c, 1))) for c in chunks]
    replies = generate(prompts) if prompts else []
    out, changed = [dict(line) for line in lines], 0
    for chunk, reply in zip(chunks, replies):
        fixed = _parse(reply)
        for n, i in enumerate(chunk, 1):
            new = fixed.get(n)
            if new and new != lines[i]["text"] and accept(lines[i]["text"], new):
                out[i]["text"] = new
                changed += 1
    return out, changed


def punctuate(lines: list[dict], generate) -> tuple[list[dict], int]:
    return _run(lines, generate, PUNCTUATE, lambda old, new: _keys(old) == _keys(new))


def readable(lines: list[dict], generate, min_similarity: float = 0.8) -> tuple[list[dict], int]:
    def accept(old: str, new: str) -> bool:
        a, b = _keys(old), _keys(new)
        return len(b) >= 0.6 * len(a) and difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= min_similarity
    return _run(lines, generate, READABLE, accept)


def polish_words(words: list[dict], generate, style: str = "readable") -> tuple[list[dict], dict]:
    """Both passes on word lists: lines are built per speaker turn, polished as text, and each line's new words are
    aligned back to the original words so they keep their real timings (subtitles stay in sync)."""
    from pipeline.correct import _retime, to_lines

    spans = to_lines(words)
    lines = [{"text": " ".join(w["text"] for w in words[a:b])} for a, b in spans]
    lines, n_punct = punctuate(lines, generate)
    n_read = 0
    if style == "readable":
        lines, n_read = readable(lines, generate)
    out = []
    for (a, b), line in zip(spans, lines):
        tokens = line["text"].split()
        same = [w["text"] for w in words[a:b]] == tokens
        out += words[a:b] if same or not tokens else _retime(words[a:b], tokens)
    return out, {"punctuated_lines": n_punct, "edited_lines": n_read}
