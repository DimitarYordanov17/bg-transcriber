import os
import tempfile
from pathlib import Path

import modal

app = modal.App("bg-transcriber")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg")
    .pip_install("faster-whisper==1.2.0", "numpy", "requests")
    .add_local_python_source("pipeline")
)

models = modal.Volume.from_name("bgt-models", create_if_missing=True)


@app.cls(image=image, gpu="A10G", volumes={"/models": models}, scaledown_window=120)
class Transcriber:
    @modal.enter()
    def load(self):
        from pipeline.asr import load_model
        self.model = load_model("large-v3", cache_dir="/models/whisper")

    @modal.method()
    def run(self, data: bytes, filename: str) -> list[dict]:
        from pipeline.asr import transcribe
        from pipeline.decode import load_audio

        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1]) as f:
            f.write(data)
            f.flush()
            audio = load_audio(f.name)
        return transcribe(self.model, audio)


@app.local_entrypoint()
def main(file: str, threshold: float = 0.7):
    path = Path(file)
    segments = Transcriber().run.remote(path.read_bytes(), path.name)
    for seg in segments:
        words = [w["text"] if w["confidence"] >= threshold else f"[{w['text']} {w['confidence']:.2f}]"
                 for w in seg["words"]]
        print(f"[{seg['start']:6.2f} - {seg['end']:6.2f}] {' '.join(words)}")
