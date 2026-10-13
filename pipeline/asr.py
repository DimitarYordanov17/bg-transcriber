import numpy as np
from faster_whisper import WhisperModel


def load_model(name: str = "large-v3", cache_dir: str | None = None) -> WhisperModel:
    return WhisperModel(name, device="cuda", compute_type="float16", download_root=cache_dir)


def transcribe(model: WhisperModel, audio: np.ndarray, language: str = "bg") -> list[dict]:
    """Returns segments, each with its words: text, start, end and confidence (0-1)."""
    segments, _ = model.transcribe(audio, language=language, beam_size=5, word_timestamps=True)
    result = []
    for seg in segments:
        words = [{"text": w.word.strip(), "start": round(float(w.start), 2), "end": round(float(w.end), 2),
                  "confidence": round(float(w.probability), 3)} for w in seg.words]
        result.append({"start": round(float(seg.start), 2), "end": round(float(seg.end), 2),
                       "text": seg.text.strip(), "words": words})
    return result
