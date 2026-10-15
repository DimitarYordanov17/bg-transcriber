"""Word and character error rate for Bulgarian transcripts.

Both sides go through the same normalization: lowercase, no punctuation, every number written as single digits
("двеста и петдесет", "250" and "2 5 0" all become "2 5 0"), so formatting choices do not count as errors.

"verbatim" scores every word. Colloquial contractions are expanded (к'во -> какво, ш'взема -> ще взема) and Latin look-alike letters inside
Cyrillic words are fixed (a typo like "приемаjте").

"clean" also drops fillers and backchannels (ъъ, хмм, аха...), collapses immediate
repetitions ("чакай, чакай" -> "чакай") and drops cut-off words ("седе-"), which is closer to what a reader wants.
"""
import re

NUMBERS = {
    "нула": 0, "един": 1, "една": 1, "едно": 1, "два": 2, "две": 2, "три": 3, "четири": 4, "пет": 5, "шест": 6,
    "седем": 7, "осем": 8, "девет": 9, "десет": 10, "единадесет": 11, "единайсет": 11, "дванадесет": 12,
    "дванайсет": 12, "тринадесет": 13, "тринайсет": 13, "четиринадесет": 14, "четиринайсет": 14,
    "петнадесет": 15, "петнайсет": 15, "шестнадесет": 16, "шестнайсет": 16, "седемнадесет": 17,
    "седемнайсет": 17, "осемнадесет": 18, "осемнайсет": 18, "деветнадесет": 19, "деветнайсет": 19,
    "двадесет": 20, "двайсет": 20, "тридесет": 30, "трийсет": 30, "четиридесет": 40, "четирийсет": 40,
    "петдесет": 50, "шестдесет": 60, "шейсет": 60, "седемдесет": 70, "осемдесет": 80, "деветдесет": 90,
    "сто": 100, "двеста": 200, "триста": 300, "четиристотин": 400, "петстотин": 500, "шестстотин": 600,
    "седемстотин": 700, "осемстотин": 800, "деветстотин": 900, "хиляда": 1000,
}
CONTRACTIONS = {"к'во": "какво", "к'ъв": "какъв", "к'ва": "каква", "к'ви": "какви", "т'ва": "това", "т'ия": "тия"}
ABBREVIATIONS = {"г-жа": "госпожа", "г-н": "господин", "г-ца": "госпожица", "д-р": "доктор", "проф.": "професор"}
LATIN_LOOKALIKES = str.maketrans("aceopxyjiABCEHKMOPTXY", "асеорхуйіАВСЕНКМОРТХУ")
FILLERS = {"ъъ", "ъ", "ъм", "хм", "хмм", "ммм", "мм", "аха", "мхм", "ааа", "еее"}
# "двама" and "2-ма", "първи" and "1-ви" must match: personal and ordinal numerals become plain digits on both sides
PERSONAL = {"двама": 2, "трима": 3, "четирима": 4, "петима": 5, "шестима": 6, "седмина": 7, "осмина": 8,
            "деветима": 9, "десетима": 10}
ORDINAL_STEMS = {"първ": 1, "втор": 2, "трет": 3, "четвърт": 4, "пет": 5, "шест": 6, "седм": 7, "осм": 8,
                 "девет": 9, "десет": 10}
ORDINAL_ENDINGS = ("и", "а", "о", "ия", "ият", "ата", "ото", "ите")
ORDINALS = {stem + end: n for stem, n in ORDINAL_STEMS.items() for end in ORDINAL_ENDINGS}
ORDINALS.pop("пето", None)  # keep the ordinal set unambiguous with cardinals: "пети"/"пета" stay ordinals


def _can_extend(current: int, value: int) -> bool:
    """'двеста' + 'петдесет' -> 250, but 'две' + 'четири' stay separate (a dictated number)."""
    for limit in (10, 100, 1000):
        if value < limit:
            return current >= limit * (2 if limit == 10 else 1) and current % limit == 0
    return False


def _join_numbers(tokens: list[str]) -> list[str]:
    out, current = [], None
    for i, tok in enumerate(tokens):
        value = NUMBERS.get(tok)
        if tok == "и" and current is not None and i + 1 < len(tokens) and tokens[i + 1] in NUMBERS:
            continue
        if value is None:
            if current is not None:
                out.append(str(current))
                current = None
            out.append(tok)
        elif current is not None and _can_extend(current, value):
            current += value
        else:
            if current is not None:
                out.append(str(current))
            current = value
    if current is not None:
        out.append(str(current))
    return out


def fix_mixed_script(word: str) -> str:
    """'приемаjте' typed with a Latin j: swap Latin look-alikes when the word is otherwise Cyrillic."""
    if re.search(r"[а-яА-Я]", word) and re.search(r"[a-zA-Z]", word):
        return word.translate(LATIN_LOOKALIKES).replace("і", "и")
    return word


def normalize(text: str, clean: bool = False) -> list[str]:
    text = " ".join(fix_mixed_script(w) for w in text.split())
    text = text.lower().replace("ѝ", "и").replace("’", "'")
    text = re.sub(r"\b(г|д|г-?ж|г-?ц)\s+-\s*", lambda m: m.group(1) + "-", text)  # "Г -жа" -> "г-жа"
    for short, full in {**CONTRACTIONS, **ABBREVIATIONS}.items():
        text = text.replace(short, full)
    text = re.sub(r"\bш'(?=\w)", "ще ", text)
    text = text.replace("%", " процента ")
    text = re.sub(r"(\d+)\s?-\s?(ви|ри|ти|ми|ия|ият|ата|ото|та|то|и|ма)\b", r"\1", text)  # 1-ви, 18-и, 2-ма
    tokens = []
    for raw in text.split():
        if clean and re.search(r"\w-[^\w]*$", raw):  # cut-off word like "седе-"
            continue
        tokens += re.sub(r"[^\w]", " ", raw).split()
    tokens = [str(PERSONAL.get(t, ORDINALS.get(t, t))) for t in tokens]
    tokens = _join_numbers(tokens)
    tokens = [d for t in tokens for d in (list(t) if t.isdigit() else [t])]
    if clean:
        kept = []
        for t in tokens:
            if t in FILLERS or (kept and kept[-1] == t and not t.isdigit()):
                continue
            kept.append(t)
        tokens = kept
    return tokens


def edit_distance(ref: list, hyp: list) -> int:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
    return d[len(hyp)]


def error_types(ref: list, hyp: list) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of one optimal alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    sub = dele = ins = 0
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            sub += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            dele, i = dele + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return sub, dele, ins


def counts(ref_text: str, hyp_text: str, clean: bool) -> dict:
    """Raw error counts, so results can be summed over many files before dividing."""
    ref, hyp = normalize(ref_text, clean), normalize(hyp_text, clean)
    ref_chars, hyp_chars = list(" ".join(ref)), list(" ".join(hyp))
    sub, dele, ins = error_types(ref, hyp)
    return {"word_errors": sub + dele + ins, "words": len(ref), "sub": sub, "del": dele, "ins": ins,
            "char_errors": edit_distance(ref_chars, hyp_chars), "chars": len(ref_chars)}


def cp_counts(ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str], clean: bool = True) -> dict:
    """Speaker-attributed errors (cpWER): each hypothesis speaker is paired with at most one reference speaker so that
    total errors are minimal; words given to the wrong person or to an extra speaker count as errors."""
    from scipy.optimize import linear_sum_assignment

    refs = [normalize(t, clean) for t in ref_by_speaker.values()]
    hyps = [normalize(t, clean) for t in hyp_by_speaker.values()]
    size = max(len(refs), len(hyps))
    refs += [[]] * (size - len(refs))
    hyps += [[]] * (size - len(hyps))
    cost = [[edit_distance(r, h) for h in hyps] for r in refs]
    rows, cols = linear_sum_assignment(cost)
    return {"word_errors": int(sum(cost[r][c] for r, c in zip(rows, cols))), "words": sum(len(r) for r in refs)}
