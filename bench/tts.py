"""Render every spoken line of a dialog with OmniVoice, cloning each speaker's voice from a reference clip.

A line's "say" (if present) is what gets spoken, e.g. colloquial spelling like "к'во"; "text" is the transcript truth.
A speaker's "voice" can be a list of references (rotated line by line) and "speed" a [min, max] range (random per line).
A speaker without "voice" is cast from <voices>/index.json: a person of the speaker's "gender" not used elsewhere
in the dialog, rotating through all of that person's references. The cast is saved as cast.json next to the lines.
The dialog's optional "tts" block overrides the OmniVoice generation settings below.

Usage: uv run modal run bench/tts.py --dialog bench/dialogs/conv05.json --voices ../samples/voices
       uv run modal run bench/tts.py --dialog bench/dialogs/dev --voices ../samples/voices   (every dialog in a folder)
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

DEFAULT_TTS = {"class_temperature": 0.7, "guidance_scale": 1.6, "denoise": False, "postprocess_output": False}
DEFAULT_SPEED = [0.95, 1.15]


@app.cls(image=image, gpu="T4", volumes={"/cache": cache}, timeout=1800, max_containers=8)
class TTS:
    @modal.enter()
    def load(self):
        import torch
        from omnivoice import OmniVoice
        self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

    @modal.method()
    def render(self, lines: list[dict], settings: dict | None = None) -> list[bytes]:
        """lines: [{"text", "ref_audio" (wav bytes), "ref_text", "speed"}] -> wav bytes per line."""
        import io
        import tempfile

        import soundfile as sf
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

        config = OmniVoiceGenerationConfig.from_dict(settings) if settings else None
        out = []
        for line in lines:
            with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
                ref.write(line["ref_audio"])
                ref.flush()
                audio = self.model.generate(text=line["text"], ref_audio=ref.name, ref_text=line["ref_text"],
                                            speed=line["speed"], generation_config=config)
            buf = io.BytesIO()
            sf.write(buf, audio[0], 24000, format="WAV")
            out.append(buf.getvalue())
        return out


def cast(spec: dict, index: list[dict], rng: random.Random) -> dict[str, list[str]]:
    """Speaker -> list of voice references. Explicit voices are kept, the rest are cast by gender."""
    taken = set()
    refs = {}
    for name, speaker in spec["speakers"].items():
        if "voice" in speaker:
            refs[name] = speaker["voice"] if isinstance(speaker["voice"], list) else [speaker["voice"]]
            taken |= {v["person"] for v in index if v["voice"] in refs[name]}
    for name, speaker in spec["speakers"].items():
        if name in refs:
            continue
        people = sorted({v["person"] for v in index if v["gender"] == speaker["gender"]} - taken)
        if not people:
            raise SystemExit(f"no unused '{speaker['gender']}' voice left for speaker {name}")
        person = rng.choice(people)
        taken.add(person)
        refs[name] = [v["voice"] for v in index if v["person"] == person]
    return refs


def prepare(dialog_path: Path, voices: Path) -> tuple[list[dict], dict, dict]:
    spec = json.loads(dialog_path.read_text())
    index = json.loads((voices / "index.json").read_text()) if (voices / "index.json").exists() else []
    rng = random.Random(dialog_path.stem)
    refs = cast(spec, index, rng)
    turn_count = {}
    lines = []
    for item in spec["script"]:
        if "speaker" not in item:
            continue
        speaker = spec["speakers"][item["speaker"]]
        n = turn_count.get(item["speaker"], 0)
        turn_count[item["speaker"]] = n + 1
        voice = refs[item["speaker"]][n % len(refs[item["speaker"]])]
        speed = speaker.get("speed", DEFAULT_SPEED)
        if isinstance(speed, list):
            speed = round(rng.uniform(*speed), 2)
        lines.append({"text": item.get("say", item["text"]), "speed": speed,
                      "ref_audio": (voices / f"{voice}.wav").read_bytes(),
                      "ref_text": (voices / f"{voice}.txt").read_text().strip()})
    return lines, spec.get("tts", DEFAULT_TTS), refs


@app.local_entrypoint()
def main(dialog: str, voices: str = "../samples/synthetic/sources", out: str = "../samples/synthetic"):
    path, voices = Path(dialog), Path(voices)
    dialogs = sorted(path.glob("*.json")) if path.is_dir() else [path]
    jobs = [prepare(d, voices) for d in dialogs]

    results = TTS().render.starmap([(lines, settings) for lines, settings, _ in jobs])

    for d, (_, _, refs), wavs in zip(dialogs, jobs, results):
        spec = json.loads(d.read_text())
        items = [(i, item) for i, item in enumerate(spec["script"]) if "speaker" in item]
        out_dir = Path(out) / d.stem / "lines"
        out_dir.mkdir(parents=True, exist_ok=True)
        for (i, item), wav in zip(items, wavs):
            (out_dir / f"{i:02d}_{item['speaker']}.wav").write_bytes(wav)
        (out_dir.parent / "cast.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1))
        print(f"{d.stem}: {len(wavs)} lines, cast {', '.join(f'{k}={v[0]}' for k, v in refs.items())}")
