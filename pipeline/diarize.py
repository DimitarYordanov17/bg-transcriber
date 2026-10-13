import numpy as np
import torch
from pyannote.audio import Pipeline

MODEL = "pyannote/speaker-diarization-community-1"


def load_pipeline(token: str) -> Pipeline:
    pipeline = Pipeline.from_pretrained(MODEL, token=token)
    pipeline.to(torch.device("cuda"))
    return pipeline


def diarize(pipeline: Pipeline, audio: np.ndarray, sr: int = 16000, exclusive: bool = False) -> list[dict]:
    """Who speaks when: [{start, end, speaker}].

    exclusive=True gives at most one speaker at any moment (overlaps go to the most likely speaker).
    """
    output = pipeline({"waveform": torch.from_numpy(audio.copy()).unsqueeze(0), "sample_rate": sr})
    annotation = output.exclusive_speaker_diarization if exclusive else output.speaker_diarization
    return [{"start": round(float(turn.start), 2), "end": round(float(turn.end), 2), "speaker": speaker}
            for turn, speaker in annotation]
