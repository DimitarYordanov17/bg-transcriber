import subprocess

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to mono float32 samples at `sr` Hz."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)
