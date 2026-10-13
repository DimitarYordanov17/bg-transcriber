"""Render every spoken line of a dialog with OmniVoice, cloning each speaker's voice from a reference clip.

A line's "say" (if present) is what gets spoken, e.g. colloquial spelling like "к'во"; "text" is the transcript truth.
A speaker's "voice" can be a list of references (rotated line by line) and "speed" a [min, max] range (random per line).
The dialog's optional "tts" block overrides OmniVoice generation settings (e.g. class_temperature, denoise).

Usage: uv run modal run bench/tts.py --dialog bench/dialogs/conv02.json [--voices ../samples/voices]
Writes <out>/<dialog name>/lines/NN_<speaker>.wav (24 kHz).
"""
import json
import random
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
def render(lines: list[dict], settings: dict | None = None) -> list[bytes]:
    """lines: [{"text", "ref_audio" (wav bytes), "ref_text", "speed"}] -> wav bytes per line."""
    import io
    import tempfile

    import soundfile as sf
    import torch
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    config = OmniVoiceGenerationConfig.from_dict(settings) if settings else None
    out = []
    for line in lines:
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
            ref.write(line["ref_audio"])
            ref.flush()
            audio = model.generate(text=line["text"], ref_audio=ref.name, ref_text=line["ref_text"],
                                   speed=line["speed"], generation_config=config)
        buf = io.BytesIO()
        sf.write(buf, audio[0], 24000, format="WAV")
        out.append(buf.getvalue())
    return out


@app.local_entrypoint()
def main(dialog: str, voices: str = "../samples/synthetic/sources", out: str = "../samples/synthetic"):
    dialog_path, voices = Path(dialog), Path(voices)
    spec = json.loads(dialog_path.read_text())
    items = [(i, item) for i, item in enumerate(spec["script"]) if "speaker" in item]
    rng = random.Random(7)
    turn_count = {}
    lines = []
    for _, item in items:
        speaker = spec["speakers"][item["speaker"]]
        refs = speaker["voice"] if isinstance(speaker["voice"], list) else [speaker["voice"]]
        n = turn_count.get(item["speaker"], 0)
        turn_count[item["speaker"]] = n + 1
        voice = refs[n % len(refs)]
        speed = speaker.get("speed", 1.0)
        if isinstance(speed, list):
            speed = round(rng.uniform(*speed), 2)
        lines.append({"text": item.get("say", item["text"]), "speed": speed,
                      "ref_audio": (voices / f"{voice}.wav").read_bytes(),
                      "ref_text": (voices / f"{voice}.txt").read_text().strip()})
        print(f"{item['speaker']}  {voice}  speed {speed}")

    wavs = render.remote(lines, spec.get("tts"))

    out_dir = Path(out) / dialog_path.stem / "lines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for (i, item), wav in zip(items, wavs):
        (out_dir / f"{i:02d}_{item['speaker']}.wav").write_bytes(wav)
    print(f"{len(wavs)} lines -> {out_dir}")
