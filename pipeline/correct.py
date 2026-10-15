"""LLM correction: every line of the transcript is proofread by an LLM, plain text in -> text out.

Lines are numbered and sent in chunks; the LLM returns the same numbered lines corrected. Hints per line: words that
are not in the Bulgarian dictionary, and what the other ASR systems heard there. A corrected line is rejected (the
original kept) when its length changes a lot, it no longer resembles anything heard there, it drops speech another
system heard, or a negation appears/disappears that no ASR system heard. Corrected words get their timing from the
original words they align to; speakers never change.
"""
import difflib
import re

from pipeline.align import align, key

PROMPT = """Ти си коректор на автоматични транскрипции на български говор. Системата за разпознаване на реч често \
греши: пише несъществуващи думи ("къснение" вместо "закъснение"), руски форми ("дно" вместо "дъно"), сбъркано \
членуване и окончания, разкъсани или слепени думи, думи със сходно звучене, които нямат смисъл в контекста.

Поправи всеки ред така, че да е това, което човекът най-вероятно е казал:
- поправяй само грешки; ако редът е наред, върни го без промяна;
- поправяй само грешки на системата за разпознаване, не граматиката на говорещия: ако човекът говори разговорно или диалектно ("ама", "тука", "щото", "абе", "човека" вместо "човекът", "батко", "тегав"), остави го така;
- не перифразирай, не съкращавай, не добавяй мисли, които не са казани; никога не добавяй и не махай отрицание;
- махни паразитни звуци (ъъ, ммм) и случайни повторения на думи;
- поправи пунктуацията и главните букви, ако е нужно.

{hints_note}
Редове:
{lines}
{hints}
Върни всички редове в същия ред и със същите номера, по един на ред, във формат "[N] поправен текст". Нищо друго."""

HINTS_NOTE = """Под редовете има подсказки: думи, които ги няма в речника (вероятни грешки, но може и да са имена), и \
какво са чули други системи за разпознаване на същия звук (те също грешат, но помагат да се разбере какво е казано).
"""

NEGATIONS = {"не", "ни"}


def aligned_tokens(base_keys: list[str], words: list[str]) -> list[list[str]]:
    """For every base position, the other system's words aligned to it (insertions go to the next base word)."""
    keys = [key(w) for w in words]
    out = [[] for _ in base_keys]
    pending = []
    last = -1
    for i, j in align(base_keys, keys):
        if i is None:
            pending.append(words[j])
            continue
        out[i] = pending + ([words[j]] if j is not None else [])
        pending, last = [], i
    if pending and out:
        out[last] = out[last] + pending
    return out


def to_lines(words: list[dict], max_line: int = 60) -> list[tuple[int, int]]:
    """[start, end) word index ranges of consecutive words of one speaker. A long monologue is cut into pieces of
    about `max_line` words, at a sentence end where there is one, so every prompt fits a small LLM's context."""
    out = []
    for i, w in enumerate(words):
        if out and words[out[-1][0]]["speaker"] == w["speaker"]:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    pieces = []
    for a, b in out:
        while b - a > max_line * 1.5:
            ends = [k + 1 for k in range(a + max_line // 2, a + max_line) if words[k]["text"][-1:] in ".?!"]
            cut = ends[-1] if ends else a + max_line
            pieces.append((a, cut))
            a = cut
        pieces.append((a, b))
    return pieces


def _text(words: list[dict]) -> str:
    return " ".join(w["text"] for w in words)


def plan(words: list[dict], others: dict[str, list[str]] | None = None, known=None,
         max_words: int = 200) -> list[dict]:
    """Chunks of lines to send: [{lines: [(a, b)], prompt}]. `others`: other systems' words (for hints),
    `known(word) -> bool`: dictionary lookup (for non-word hints)."""
    lines = to_lines(words)
    base_keys = [key(w["text"]) for w in words]
    per_system = [aligned_tokens(base_keys, toks) for toks in (others or {}).values()]
    chunks, current, size = [], [], 0
    for a, b in lines:
        if current and size + (b - a) > max_words:
            chunks.append(current)
            current, size = [], 0
        current.append((a, b))
        size += b - a
    if current:
        chunks.append(current)
    out = []
    for chunk in chunks:
        numbered, hints = [], []
        for n, (a, b) in enumerate(chunk, 1):
            numbered.append(f"[{n}] {_text(words[a:b])}")
            if known is not None:
                bad = [k for w in words[a:b]
                       if (k := re.sub(r"[^\w-]", "", w["text"]).strip("-")) and not k.isdigit() and not known(k)]
                if bad:
                    hints.append(f"[{n}] няма ги в речника: {', '.join(dict.fromkeys(bad))}")
            # ordered (not a set): the same input must always give the same prompt
            heard = dict.fromkeys(" ".join(t for i in range(a, b) for t in toks[i]) for toks in per_system)
            heard = [f"«{h}»" for h in heard if h and key(h) != key(_text(words[a:b]))]
            if heard:
                hints.append(f"[{n}] други системи чуха: " + " | ".join(heard))
        prompt = PROMPT.format(hints_note=HINTS_NOTE if hints else "", lines="\n".join(numbered),
                               hints=("\nПодсказки:\n" + "\n".join(hints) + "\n") if hints else "")
        out.append({"lines": chunk, "prompt": prompt})
    return out


def parse(reply: str) -> dict[int, str]:
    out = {}
    for line in reply.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s*(?:[^:\[\]]{1,20}:\s*)?(.*)$", line)
        if m and m.group(2).strip():
            out[int(m.group(1))] = re.sub(r"\s+", " ", m.group(2)).strip()
    return out


def _similar(a: list[str], b: list[str]) -> float:
    return difflib.SequenceMatcher(None, [key(t) for t in a], [key(t) for t in b], autojunk=False).ratio()


def _drops_supported(old: list[str], new: list[str], heard: list[list[str]], min_run: int = 6) -> bool:
    """True if the correction drops a run of >= min_run words that Whisper had and another system also heard
    (real speech the LLM threw away; seen with a 27B model deleting a whole clause)."""
    ok, nk = [key(t) for t in old], [key(t) for t in new]
    heard_keys = [set(key(t) for t in h) for h in heard if h]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ok, nk, autojunk=False).get_opcodes():
        lost = ok[i1:i2] if op == "delete" else ok[i1:i2][: max(0, (i2 - i1) - (j2 - j1))] if op == "replace" else []
        if len(lost) >= min_run and any(sum(t in hk for t in lost) >= len(lost) / 2 for hk in heard_keys):
            return True
    return False


def _accept(old: list[str], new: list[str], heard: list[list[str]], min_similarity: float = 0.5) -> bool:
    n, m = len(old), len(new)
    if not (max(1, n // 2) - 1 <= m <= int(n * 1.5) + 2):
        return False
    # the LLM sometimes returns another line's text under this line's number: the new text must still resemble what
    # Whisper or one of the other systems heard here
    if max([_similar(old, new)] + [_similar(h, new) for h in heard if h]) < min_similarity:
        return False
    if _drops_supported(old, new, heard):
        return False
    neg = lambda toks: sum(key(t) in NEGATIONS for t in toks)  # noqa: E731
    return neg(new) == neg(old) or any(neg(h) == neg(new) for h in heard)


def replace_outliers(words: list[dict], others: dict[str, list[str]], max_similarity: float = 0.5,
                     min_agreement: float = 0.6) -> tuple[list[dict], int]:
    """Before the rewrite: a line where Whisper disagrees with every other system (it hallucinated or drifted) while
    the others agree among themselves gets the most central other version as its text. Timing and speaker stay."""
    base_keys = [key(w["text"]) for w in words]
    per_system = [aligned_tokens(base_keys, toks) for toks in others.values()]
    out, i, replaced = [], 0, 0
    for a, b in to_lines(words):
        old = [w["text"] for w in words[a:b]]
        heard = [[t for k in range(a, b) for t in toks[k]] for toks in per_system]
        heard = [h for h in heard if h]
        if len(heard) >= 2 and max(_similar(old, h) for h in heard) < max_similarity:
            center = max(heard, key=lambda h: sum(_similar(h, x) for x in heard))
            if sum(_similar(center, x) for x in heard if x is not center) / (len(heard) - 1) >= min_agreement:
                out += _retime(words[a:b], center)
                replaced += 1
                continue
        out += words[a:b]
    return out, replaced


def apply(words: list[dict], chunks: list[dict], replies: list[str],
          others: dict[str, list[str]] | None = None) -> tuple[list[dict], int]:
    """Corrected words, and how many lines were changed."""
    base_keys = [key(w["text"]) for w in words]
    per_system = [aligned_tokens(base_keys, toks) for toks in (others or {}).values()]
    replaced = {}
    for chunk, reply in zip(chunks, replies):
        fixed = parse(reply or "")
        for n, (a, b) in enumerate(chunk["lines"], 1):
            old = [w["text"] for w in words[a:b]]
            new = fixed.get(n, "").split()
            heard = [[t for i in range(a, b) for t in toks[i]] for toks in per_system]
            if new and new != old and _accept(old, new, heard):
                replaced[a] = (b, _retime(words[a:b], new))
    out, i = [], 0
    while i < len(words):
        if i in replaced:
            b, new = replaced[i]
            out += new
            i = b
        else:
            out.append(words[i])
            i += 1
    return out, len(replaced)


def _retime(old: list[dict], tokens: list[str]) -> list[dict]:
    """New tokens keep the timing/confidence of the old words they align to; others are placed between neighbours."""
    pairs = align([key(w["text"]) for w in old], [key(t) for t in tokens])
    match = {j: i for i, j in pairs if i is not None and j is not None}
    out, last_end = [], old[0]["start"]
    for j, t in enumerate(tokens):
        if j in match:
            w = old[match[j]]
            out.append({**w, "text": t})
            last_end = w["end"]
        else:
            nxt = next((old[match[k]]["start"] for k in range(j + 1, len(tokens)) if k in match), old[-1]["end"])
            out.append({**old[0], "text": t, "start": round(last_end, 2), "end": round(max(last_end, nxt), 2),
                        "confidence": 0.5, "corrected": True})
    return out


def rewrite(words: list[dict], others: dict[str, list[str]], known, generate, passes: int = 2,
            outlier_similarity: float = 0.65, repair: bool = True) -> tuple[list[dict], dict]:
    """The whole LLM step as the product runs it: replace Whisper's outlier lines (outlier_similarity 0: off), then
    `passes` rewrite passes, then the non-word repair. `generate(prompts) -> replies` is the LLM. Returns the words
    and a few counts for the log."""
    words, outliers = replace_outliers(words, others, outlier_similarity) if outlier_similarity else (words, 0)
    changed = []
    for _ in range(passes):
        chunks = plan(words, others, known)
        replies = generate([c["prompt"] for c in chunks]) if chunks else []
        words, n = apply(words, chunks, replies, others)
        changed.append(n)
    questions = repair_plan(words, others, known) if repair else []  # words the dictionary still rejects
    replies = generate([q["prompt"] for q in questions]) if questions else []
    words, fixed = repair_apply(words, questions, replies, known)
    return words, {"outlier_lines": outliers, "changed_lines": changed, "nonwords_asked": len(questions),
                   "nonwords_fixed": fixed}


REPAIR_PROMPT = """Това е изречение от автоматична транскрипция на български говор. Думата «{word}» не съществува в \
българския език: системата за разпознаване е сгрешила. Какво най-вероятно е казано на нейно място?

Изречение: {sentence}
{heard}
Отговори само с правилната дума или думи (без кавички и обяснения). Ако е име, което не познаваш, повтори думата."""


def repair_plan(words: list[dict], others: dict[str, list[str]], known, context: int = 12) -> list[dict]:
    """One focused question per word the dictionary rejects. Names (capitalized), numbers and abbreviations are
    skipped: without a list of the real names the LLM replaces right names with wrong ones."""
    base_keys = [key(w["text"]) for w in words]
    per_system = [aligned_tokens(base_keys, toks) for toks in others.values()]
    out = []
    for i, w in enumerate(words):
        core = re.sub(r"[^\w-]", "", w["text"]).strip("-")
        if (not core or any(c.isdigit() for c in core) or len(core) < 3 or core[:1].isupper()
                or known(core)):
            continue
        sentence = " ".join(x["text"] for x in words[max(0, i - context):i + context + 1])
        heard = dict.fromkeys(" ".join(toks[i]) for toks in per_system if toks[i])
        heard = [h for h in heard if key(h) != key(core)]
        out.append({"index": i, "word": core, "prompt": REPAIR_PROMPT.format(
            word=core, sentence=sentence, heard=("Други системи чуха на това място: " + " | ".join(f"«{h}»" for h in heard) + "\n")
            if heard else "")})
    return out


def repair_apply(words: list[dict], questions: list[dict], replies: list[str], known) -> tuple[list[dict], list]:
    """Replace each non-word by the LLM's answer if every answered word is a real word (at most 3 words)."""
    out, fixed = [dict(w) for w in words], []
    for q, reply in zip(questions, replies):
        answer = (reply or "").strip().splitlines()[0].strip(" .,«»\"'") if (reply or "").strip() else ""
        tokens = answer.split()
        if not tokens or len(tokens) > 3 or key(answer) == key(q["word"]):
            continue
        if not all(known(re.sub(r"[^\w-]", "", t)) for t in tokens):
            continue
        if difflib.SequenceMatcher(None, q["word"].lower(), answer.lower()).ratio() < 0.5:
            continue  # the fix must sound like the broken word, not be a free rewrite
        w = out[q["index"]]
        lead = re.match(r"^[^\w]*", w["text"]).group(0)
        tail = re.search(r"[^\w]*$", w["text"]).group(0)
        w["text"] = lead + " ".join(tokens) + tail
        fixed.append((q["word"], " ".join(tokens)))
    return out, fixed
