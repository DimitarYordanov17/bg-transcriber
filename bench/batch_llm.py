"""Answer prompts collected by experiment.py (COLLECT_PROMPTS=<file>) in one call to BgGPT, so the model loads
once for many runs. Answers go to the same cache the experiments read.

Usage: uv run python bench/batch_llm.py <prompts.jsonl>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.experiment import cached, store  # noqa: E402
from modal_app import BgGPT, app  # noqa: E402


def main() -> None:
    rows = {}
    for line in Path(sys.argv[1]).read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["key"], r)
    by_model = {}
    for r in rows.values():
        if cached("llm", r["key"]) is None:
            by_model.setdefault(r["llm"], []).append(r)
    with app.run():
        for model, todo in by_model.items():
            replies = BgGPT(model=model).generate.remote([r["prompt"] for r in todo], 2048)
            for r, reply in zip(todo, replies):
                store("llm", r["key"], reply)
            print(f"{model}: {len(todo)} answers stored", flush=True)


if __name__ == "__main__":
    main()
