"""Render every spoken line of a dialog with OmniVoice, cloning each speaker's voice from a FLEURS clip.

Usage: uv run modal run bench/tts.py --dialog bench/dialogs/conv02.json [--voices ../samples/voices]
Writes <out>/<dialog name>/NN_<speaker>.wav (24 kHz).
"""
import json
from pathlib import Path

import modal

app = modal.App("bg-transcriber-tts")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("omnivoice", "soundfile", "torch", "huggingface_hub")
    .env({"HF_HOME": "/cache/huggingface"})
)

cache = modal.Volume.from_name("omnivoice-cache")


@app.function(image=image, gpu="T4", volumes={"/cache": cache}, timeout=900)
def render(lines: list[dict]) -> list[bytes]:
    """lines: [{"text", "ref_audio" (wav bytes), "ref_text"}] -> wav bytes per line."""
    import io
    import tempfile

    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    out = []
    for line in lines:
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
            ref.write(line["ref_audio"])
            ref.flush()
            audio = model.generate(text=line["text"], ref_audio=ref.name, ref_text=line["ref_text"])
        buf = io.BytesIO()
        sf.write(buf, audio[0], 24000, format="WAV")
        out.append(buf.getvalue())
    return out


@app.local_entrypoint()
def main(dialog: str, voices: str = "../samples/synthetic/sources", out: str = "../samples/synthetic/tts"):
    dialog_path, voices = Path(dialog), Path(voices)
    spec = json.loads(dialog_path.read_text())
    items = [(i, item) for i, item in enumerate(spec["script"]) if "speaker" in item]
    lines = []
    for _, item in items:
        voice = spec["speakers"][item["speaker"]]["voice"]
        lines.append({"text": item["text"], "ref_audio": (voices / f"{voice}.wav").read_bytes(),
                      "ref_text": (voices / f"{voice}.txt").read_text().strip()})

    wavs = render.remote(lines)

    out_dir = Path(out) / dialog_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for (i, item), wav in zip(items, wavs):
        (out_dir / f"{i:02d}_{item['speaker']}.wav").write_bytes(wav)
    print(f"{len(wavs)} lines -> {out_dir}")
