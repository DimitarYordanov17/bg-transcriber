"""Where data and run outputs live: ../data and ../runs, next to the repo."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REAL = DATA / "real"
SYNTHETIC = DATA / "synthetic"
SOURCES = SYNTHETIC / "sources"
VOICES = DATA / "voices"
RUNS = ROOT / "runs"


def synthetic_dir(dialog: Path) -> Path:
    """bench/dialogs/test/t07.json -> data/synthetic/test/t07 (same split folder as the dialog)."""
    return SYNTHETIC / dialog.parent.name / dialog.stem
