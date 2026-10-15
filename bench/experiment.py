"""Run one pipeline configuration on a data split and log its scores.

Usage: uv run python bench/experiment.py bench/configs/local.json --split yt_dev

The pipeline is the product's own (pipeline/*.py, the same steps transcribe_file runs). Model outputs are cached per
file in runs/cache/, LLM answers per prompt, so a config that changes one step reuses everything else. With
COLLECT_PROMPTS=<file> the LLM prompts a run still needs are written to that file instead of being sent, so
bench/batch_llm.py can answer all of them with one model load; run again afterwards to continue.

Per-file results go to runs/<time>_<name>_<split>/per_file.json (used by compare.py), one summary row to
runs/results.md.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.paths import REAL, ROOT, RUNS, SYNTHETIC  # noqa: E402
from bench.providers import PROVIDERS, openai_chat  # noqa: E402
from bench.score import counts, cp_counts  # noqa: E402
from modal_app import (NEMO_MODELS, BgGPT, Diarizer, NemoTranscriber, SortformerDiarizer,  # noqa: E402
                       Transcriber, app)
from pipeline.clean import clean  # noqa: E402
from pipeline.correct import rewrite  # noqa: E402
from pipeline.merge import drop_unvoiced, smooth_flips, text_by_speaker, words_with_speakers  # noqa: E402
from pipeline.polish import polish_words  # noqa: E402
from pipeline.rescue import apply as apply_rescue, find_windows  # noqa: E402

CACHE = RUNS / "cache"
RESULTS = RUNS / "results.md"
CONFIGS = Path(__file__).parent / "configs"
TURN_PREFIX = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([A-ZА-Я])\s*(?:\([^)]*\))?\s*:\s*")
# split -> folders under data/real ("dev" also includes the synthetic dev dialogs). The two *_test splits were
# run once at the end, nothing was tuned on them.
SPLITS = {"dev": ("dev", "test"), "parliament": ("parliament_dev",), "yt_dev": ("youtube_dev",),
          "parl_test": ("parliament_test",), "yt_test": ("youtube_test",)}
HEADER = "| Date | Run | Split | Config | Norm | Synthetic WER | Synthetic cpWER | Real WER | Real cpWER | Notes |\n" \
         "|---|---|---|---|---|---|---|---|---|---|\n"


def sha(data: bytes | str, n: int = 8) -> str:
    return hashlib.sha1(data if isinstance(data, bytes) else data.encode()).hexdigest()[:n]


def cached(kind: str, key: str):
    path = CACHE / kind / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def store(kind: str, key: str, value) -> None:
    (CACHE / kind).mkdir(parents=True, exist_ok=True)
    (CACHE / kind / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False))


def dev_files(split: str) -> list[dict]:
    """Files of a split with audio, reference text by speaker and a group ("synthetic" or "real")."""
    files = []
    if split == "dev":
        for folder in sorted((SYNTHETIC / "dev").iterdir()):
            truth = json.loads((folder / f"{folder.name}.json").read_text())
            turns = sorted(truth["turns"], key=lambda t: t["start"])
            by = {}
            for t in turns:
                by.setdefault(t["speaker"], []).append(t["text"])
            files.append({"name": folder.name, "audio": folder / f"{folder.name}.wav", "group": "synthetic",
                          "ref": {s: " ".join(v) for s, v in by.items()}, "ref_text": " ".join(t["text"] for t in turns)})
    for folder in SPLITS[split]:
        for audio in sorted((REAL / folder).glob("*.flac")):
            ref_file = audio.with_suffix(".txt")
            if not ref_file.exists():
                continue
            by, ordered = {}, []
            for line in ref_file.read_text().splitlines():
                m = TURN_PREFIX.match(line)
                if m:
                    by.setdefault(m.group(1), []).append(line[m.end():])
                    ordered.append(line[m.end():])
            files.append({"name": audio.stem, "audio": audio, "group": "real",
                          "ref": {s: " ".join(v) for s, v in by.items()}, "ref_text": " ".join(ordered)})
    return files


def system_asr(name: str) -> dict:
    """ASR settings of another config by name (e.g. "canary_v2"), for the other models' versions."""
    return json.loads((CONFIGS / f"{name}.json").read_text())["asr"]


def ensure_asr(files: list[dict], asr: dict) -> dict[str, list]:
    """Segments of one ASR model for every file, computed only where the cache misses. Returns name -> segments."""
    keys = {f["name"]: sha(f["file_key"] + json.dumps(asr, sort_keys=True), 16) for f in files}
    todo = [f for f in files if cached("asr", keys[f["name"]]) is None]
    if todo:
        engine = NemoTranscriber if asr["model"] in NEMO_MODELS else Transcriber
        with app.run():
            outs = list(engine(model=asr["model"]).run_opts.starmap(
                [(f["bytes"], f["audio"].name, asr.get("opts", {})) for f in todo]))
        for f, out in zip(todo, outs):
            store("asr", keys[f["name"]], out["segments"])
    return {f["name"]: cached("asr", keys[f["name"]]) for f in files}


def ensure_turns(files: list[dict], route: bool) -> dict[str, list]:
    """Speaker turns: pyannote, or (route) the one of pyannote and Sortformer that found more speakers."""
    missing_pa = [f for f in files if cached("diar", f"{f['file_key']}_ex") is None]
    missing_sf = [f for f in files if route and cached("diar", f"{f['file_key']}_ex_sortformer-v1") is None]
    if missing_pa or missing_sf:
        with app.run():
            pa = [Diarizer().run.spawn(f["bytes"], f["audio"].name, exclusive=True) for f in missing_pa]
            sf = [SortformerDiarizer(model="sortformer-v1").run.spawn(f["bytes"], f["audio"].name) for f in missing_sf]
            for f, call in zip(missing_pa, pa):
                store("diar", f"{f['file_key']}_ex", call.get())
            for f, call in zip(missing_sf, sf):
                store("diar", f"{f['file_key']}_ex_sortformer-v1", call.get())
    out = {}
    for f in files:
        turns = cached("diar", f"{f['file_key']}_ex")
        if route:
            other = cached("diar", f"{f['file_key']}_ex_sortformer-v1")
            if len({t["speaker"] for t in other}) >= len({t["speaker"] for t in turns}):
                turns = other
        out[f["name"]] = turns
    return out


def ensure_rescue(files: list[dict], segments: dict, turns: dict) -> dict[str, list]:
    """Whisper segments with skipped speech re-decoded (pipeline/rescue.py), cached per file."""
    todo = []
    for f in files:
        windows = find_windows(segments[f["name"]], turns[f["name"]])
        if windows and cached("asr", f["rescue_key"]) is None:
            todo.append((f, windows))
    if todo:
        with app.run():
            calls = [Transcriber(model="large-v3").run_windows.spawn(
                f["bytes"], f["audio"].name, w, {"beam_size": 5, "condition_on_previous_text": False}) for f, w in todo]
            for (f, _), call in zip(todo, calls):
                store("asr", f["rescue_key"], call.get()["windows"])
    out = {}
    for f in files:
        rescued = cached("asr", f["rescue_key"])
        out[f["name"]] = apply_rescue(segments[f["name"]], rescued)[0] if rescued else segments[f["name"]]
    return out


class Missing(Exception):
    """An LLM answer is not cached yet (COLLECT_PROMPTS mode)."""


def llm(model: str, missing: list):
    """generate(prompts) -> replies with every answer cached by prompt. In COLLECT_PROMPTS mode a missing answer is
    recorded and Missing is raised, so the run collects the next batch of prompts instead of guessing."""
    def generate(prompts: list[str]) -> list[str]:
        keys = [sha(json.dumps([model, p]), 16) for p in prompts]
        todo = [(k, p) for k, p in zip(keys, prompts) if cached("llm", k) is None]
        if todo and os.environ.get("COLLECT_PROMPTS"):
            missing.extend((model, k, p) for k, p in todo)
            raise Missing
        if todo and model.startswith("gpt-"):
            for k, p in todo:
                store("llm", k, openai_chat(p, model, json_mode=False))
        elif todo:
            with app.run():
                for (k, _), reply in zip(todo, BgGPT(model=model).generate.remote([p for _, p in todo], 2048)):
                    store("llm", k, reply)
        return [cached("llm", k) for k in keys]
    return generate


def dictionary_lookup():
    """known(word) -> bool from the Bulgarian hunspell dictionary. Answers are kept in a file shared by all runs, so
    the (slow to load) dictionary is opened only when a new word appears."""
    path = CACHE / "dict_lookup.json"
    try:
        memo = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        memo = {}
    state = {"dict": None, "dirty": 0}

    def save():  # several runs share the file: write a copy and swap it in, never a half-written file
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(memo, ensure_ascii=False))
        tmp.replace(path)

    def known(w: str) -> bool:
        if w not in memo:
            if state["dict"] is None:
                from spylls.hunspell import Dictionary
                state["dict"] = Dictionary.from_files(str(ROOT / "data" / "dict" / "bg_BG"))
            memo[w] = bool(state["dict"].lookup(w) or state["dict"].lookup(w.lower()))
            state["dirty"] += 1
        return memo[w]

    import atexit
    atexit.register(lambda: state["dirty"] and save())
    return known


def run(config: dict, files: list[dict]) -> list[dict]:
    for f in files:
        f["bytes"] = f["audio"].read_bytes()
        f["file_key"] = sha(f["bytes"], 16)
        f["rescue_key"] = sha(f["file_key"] + json.dumps(config["asr"], sort_keys=True), 16) + "_rescue"

    provider = config.get("provider")  # a commercial service on its own, with its own speaker labels
    if provider:
        words_by_file = {}
        for f in files:
            k = sha(f["file_key"] + json.dumps({"model": provider}, sort_keys=True), 16)
            if cached("asr", k) is None:
                store("asr", k, PROVIDERS[provider](f["bytes"], f["audio"].name))
            words_by_file[f["name"]] = cached("asr", k)
        return score(files, words_by_file)

    turns = ensure_turns(files, config.get("route", True))
    if config.get("words_from"):  # mixed mode: a commercial service's words, our speakers
        segments = {}
        for f in files:
            k = sha(f["file_key"] + json.dumps({"model": config["words_from"]}, sort_keys=True), 16)
            if cached("asr", k) is None:
                store("asr", k, PROVIDERS[config["words_from"]](f["bytes"], f["audio"].name))
            segments[f["name"]] = [{"start": w["start"], "end": w["end"], "text": w["text"],
                                    "words": [{"text": w["text"], "start": w["start"], "end": w["end"],
                                               "confidence": 1.0}]} for w in cached("asr", k)]
    else:
        segments = ensure_asr(files, config["asr"])
        if config.get("rescue"):
            segments = ensure_rescue(files, segments, turns)

    others = {name: ensure_asr(files, system_asr(name)) for name in config.get("others", [])}
    known = dictionary_lookup()
    missing = []
    words_by_file = {}
    for f in files:
        words = words_with_speakers(segments[f["name"]], turns[f["name"]])
        if config.get("drop_unvoiced") is not None:
            words = drop_unvoiced(words, turns[f["name"]], config["drop_unvoiced"])
        if config.get("smooth_flips"):
            words = smooth_flips(words)
        hyps = {n: " ".join(s["text"] for s in o[f["name"]]).split() for n, o in others.items()}
        try:
            if config.get("rewrite"):
                cfg = config["rewrite"]
                words, _ = rewrite(words, hyps, known, llm(cfg["llm"], missing), cfg.get("passes", 2),
                                   cfg.get("outlier_sim", 0.65), cfg.get("repair", True))
            if config.get("polish"):
                words, _ = polish_words(words, llm(config["polish"]["llm"], missing), config["polish"]["style"])
        except Missing:
            continue
        if config.get("clean"):
            words = clean(words, hyps if hyps else None)
        words_by_file[f["name"]] = words
    if missing:
        with open(os.environ["COLLECT_PROMPTS"], "a") as fh:
            for model, k, prompt in {k: (m, k, p) for m, k, p in missing}.values():
                fh.write(json.dumps({"llm": model, "key": k, "prompt": prompt}, ensure_ascii=False) + "\n")
        sys.exit(f"collected {len({k for _, k, _ in missing})} prompts")
    return score(files, words_by_file)


def score(files: list[dict], words_by_file: dict) -> list[dict]:
    results = []
    for f in files:
        words = words_by_file[f["name"]]
        hyp = " ".join(w["text"] for w in words)
        results.append({"name": f["name"], "group": f["group"], "hyp": hyp,
                        "wer": counts(f["ref_text"], hyp, clean=True),
                        "cp": cp_counts(f["ref"], text_by_speaker(words), clean=True)})
    return results


def rate(rows: list[dict], key: str) -> float | None:
    words = sum(r[key]["words"] for r in rows)
    return sum(r[key]["word_errors"] for r in rows) / words if words else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--split", default="dev", choices=list(SPLITS))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    results = run(config, dev_files(args.split))

    run_dir = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{config['name']}_{args.split}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    (run_dir / "per_file.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    pct = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    syn = [r for r in results if r["group"] == "synthetic"]
    real = [r for r in results if r["group"] == "real"]
    norm = sha((Path(__file__).parent / "score.py").read_bytes())
    row = (f"| {time.strftime('%m-%d %H:%M')} | {config['name']} | {args.split} | {sha(json.dumps(config, sort_keys=True))} | "
           f"{norm} | {pct(rate(syn, 'wer'))} | {pct(rate(syn, 'cp'))} | {pct(rate(real, 'wer'))} | "
           f"{pct(rate(real, 'cp'))} | {args.notes} |\n")
    if not RESULTS.exists():
        RESULTS.write_text("# Results\n\nClean WER; cpWER also counts words given to the wrong speaker.\n\n" + HEADER)
    with RESULTS.open("a") as fh:
        fh.write(row)
    print(row, end="")
    print(f"-> {run_dir}")


if __name__ == "__main__":
    main()
