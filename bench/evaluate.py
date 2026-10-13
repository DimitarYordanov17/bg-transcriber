"""Transcribe every conversation of a split on Modal, score it and append one row to bench/results.md.

Usage: uv run python bench/evaluate.py --split test --name whisper-large-v3 --notes "no preprocessing"

Synthetic audio and truth come from ../samples/synthetic/<dialog>/. Real clips of the same split are scored too:
../samples/real/<split>/<name>.flac with the hand-corrected <name>.txt ("[mm:ss] A: text" per line).
Hypotheses and per-file scores are kept in ../samples/runs/<time>_<name>/ so a run can be inspected later.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.score import counts  # noqa: E402
from modal_app import Transcriber, app  # noqa: E402

PROFILES = ["clean", "quiet_speaker", "phone", "noisy", "mixed"]
RESULTS = Path(__file__).parent / "results.md"
HEADER = ("| Date | Commit | Split | Run | WER verbatim | WER clean | CER clean | "
          + " | ".join(f"{p} WER" for p in PROFILES) + " | Real WER | Notes |\n"
          + "|" + "---|" * (9 + len(PROFILES)) + "\n")
TURN_PREFIX = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?[A-ZА-Я]\s*(?:\([^)]*\))?\s*:\s*")


def real_clips(split: str) -> list[Path]:
    folder = Path("../samples/real") / split
    return sorted(p for p in folder.glob("*.flac") if p.with_suffix(".txt").exists())


def read_reference(path: Path) -> str:
    return " ".join(TURN_PREFIX.sub("", line) for line in path.read_text().splitlines() if line.strip())


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def rate(c: dict, kind: str = "word") -> float:
    return c[f"{kind}_errors"] / c[f"{kind}s"] if c[f"{kind}s"] else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--name", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--data", type=Path, default=Path("../samples/synthetic"))
    ap.add_argument("--clean-audio", action="store_true", help="use <name>_clean.wav (no degradation, equal levels)")
    args = ap.parse_args()

    names = sorted(p.stem for p in (Path(__file__).parent / "dialogs" / args.split).glob("*.json"))
    run_dir = Path("../samples/runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.name}"
    run_dir.mkdir(parents=True)

    with modal.enable_output(), app.run():
        suffix = "_clean" if args.clean_audio else ""
        inputs = [((args.data / n / f"{n}{suffix}.wav").read_bytes(), f"{n}.wav") for n in names]
        reals = [] if args.clean_audio else real_clips(args.split)
        inputs += [(p.read_bytes(), p.name) for p in reals]
        everything = list(Transcriber().run.starmap(inputs))
        outputs, real_outputs = everything[:len(names)], everything[len(names):]

    totals = {"verbatim": defaultdict(int), "clean": defaultdict(int)}
    by_profile = {p: defaultdict(int) for p in PROFILES}
    per_file = {}
    for name, segments in zip(names, outputs):
        truth = json.loads((args.data / name / f"{name}.json").read_text())
        ref = " ".join(t["text"] for t in sorted(truth["turns"], key=lambda t: t["start"]))
        hyp = " ".join(s["text"] for s in segments)
        v, c = counts(ref, hyp, clean=False), counts(ref, hyp, clean=True)
        for k in v:
            totals["verbatim"][k] += v[k]
            totals["clean"][k] += c[k]
            by_profile[truth["profile"]][k] += c[k]
        per_file[name] = {"profile": truth["profile"], "wer_verbatim": round(rate(v), 4),
                          "wer_clean": round(rate(c), 4), "hyp": hyp, "ref": ref}
        (run_dir / f"{name}.json").write_text(json.dumps(segments, ensure_ascii=False, indent=1))

    real_total = defaultdict(int)
    for clip, segments in zip(reals, real_outputs):
        hyp = " ".join(s["text"] for s in segments)
        c = counts(read_reference(clip.with_suffix(".txt")), hyp, clean=True)
        for k in c:
            real_total[k] += c[k]
        per_file[clip.stem] = {"profile": "real", "wer_clean": round(rate(c), 4), "hyp": hyp}
        (run_dir / f"{clip.stem}.json").write_text(json.dumps(segments, ensure_ascii=False, indent=1))

    summary = {"split": args.split, "name": args.name, "commit": commit(), "notes": args.notes,
               "wer_verbatim": rate(totals["verbatim"]), "wer_clean": rate(totals["clean"]),
               "cer_clean": rate(totals["clean"], "char"),
               "by_profile": {p: rate(c) for p, c in by_profile.items() if c["words"]},
               "real_wer": rate(real_total) if real_total["words"] else None, "real_words": real_total["words"]}
    (run_dir / "scores.json").write_text(json.dumps({"summary": summary, "files": per_file},
                                                    ensure_ascii=False, indent=1))

    cells = [f"{summary['by_profile'][p]:.1%}" if p in summary["by_profile"] else "-" for p in PROFILES]
    row = (f"| {time.strftime('%Y-%m-%d %H:%M')} | {summary['commit']} | {args.split} | {args.name} | "
           f"{summary['wer_verbatim']:.1%} | {summary['wer_clean']:.1%} | {summary['cer_clean']:.1%} | "
           + " | ".join(cells)
           + (f" | {summary['real_wer']:.1%} ({summary['real_words']} w)" if summary["real_wer"] is not None else " | -")
           + f" | {args.notes} |\n")
    if not RESULTS.exists():
        RESULTS.write_text("# Results\n\nOne row per evaluation run. WER = word error rate, lower is better.\n\n"
                           + HEADER)
    with RESULTS.open("a") as f:
        f.write(row)

    for name, r in per_file.items():
        print(f"{name}  {r['profile']:14s} clean {r['wer_clean']:6.1%}")
    print(HEADER + row, end="")
    print(f"details: {run_dir}")


if __name__ == "__main__":
    main()
