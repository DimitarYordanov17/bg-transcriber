"""Paired comparison of two runs on the same files: is the candidate really better than the base?

Usage: uv run python bench/compare.py runs/<base run> runs/<candidate run> [--metric cp|wer] [--mode gate|noninferior]

Synthetic dialogs are resampled with replacement (paired bootstrap, whole dialogs) to get a 95% interval for the
difference in error rate. Gate (adopt the candidate): gain >= 1 point AND interval above 0 AND better in >= 70% of the
dialogs that differ. Non-inferior (for robustness settings): the interval's worst case is not worse than 0.5 points.
The real panel is reported separately; a regression of more than 3 points there is flagged for human review.
The verdict is appended to runs/decisions.md.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.paths import RUNS  # noqa: E402


def load(run: Path) -> tuple[dict, dict]:
    config = json.loads((run / "config.json").read_text())
    return config, {r["name"]: r for r in json.loads((run / "per_file.json").read_text())}


def error_rate(rows: list[dict], key: str) -> float:
    return sum(r[key]["word_errors"] for r in rows) / max(1, sum(r[key]["words"] for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--metric", choices=["cp", "wer"], default="cp")
    ap.add_argument("--mode", choices=["gate", "noninferior"], default="gate")
    ap.add_argument("--group", choices=["synthetic", "real"], default="synthetic",
                    help="which files the bootstrap runs over (real: parliament and other real-speech splits)")
    args = ap.parse_args()
    base_cfg, base = load(args.base)
    cand_cfg, cand = load(args.candidate)
    if base_cfg["hashes"]["normalizer"] != cand_cfg["hashes"]["normalizer"]:
        sys.exit("refusing: runs were scored with different normalizers")
    key = args.metric

    names = sorted(n for n in base if base[n]["group"] == args.group and n in cand)
    b_rows, c_rows = [base[n] for n in names], [cand[n] for n in names]
    diff = error_rate(b_rows, key) - error_rate(c_rows, key)  # positive = candidate better

    rng = random.Random(0)
    boots = []
    for _ in range(5000):
        idx = [rng.randrange(len(names)) for _ in names]
        boots.append(error_rate([b_rows[i] for i in idx], key) - error_rate([c_rows[i] for i in idx], key))
    boots.sort()
    low, high = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

    per = [b[key]["word_errors"] / max(1, b[key]["words"]) - c[key]["word_errors"] / max(1, c[key]["words"])
           for b, c in zip(b_rows, c_rows)]
    better, worse = sum(d > 0 for d in per), sum(d < 0 for d in per)
    share = better / max(1, better + worse)

    real = sorted(n for n in base if base[n]["group"] == "real" and n in cand)
    real_diff = error_rate([base[n] for n in real], key) - error_rate([cand[n] for n in real], key) if real else 0.0

    if args.mode == "gate":
        adopted = diff >= 0.01 and low > 0 and share >= 0.70
        verdict = "ADOPT" if adopted else ("near-miss" if diff > 0 and low > -0.005 else "reject")
    else:
        adopted = low >= -0.005
        verdict = "NON-INFERIOR" if adopted else "worse"
    flag = " | FLAG: real panel regression" if real_diff < -0.03 else ""

    line = (f"{time.strftime('%m-%d %H:%M')} | {cand_cfg['name']} vs {base_cfg['name']} | {key} | "
            f"gain {diff * 100:+.2f} pt [95% {low * 100:+.2f}, {high * 100:+.2f}] | better in {better}/{better + worse} "
            f"dialogs | real {real_diff * 100:+.2f} pt | {verdict}{flag}")
    with (RUNS / "decisions.md").open("a") as fh:
        fh.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
