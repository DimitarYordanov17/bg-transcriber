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

diarize_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("pyannote.audio>=4.0,<5", "numpy")
    .env({"HF_HOME": "/models/hf"})
    .add_local_python_source("pipeline")
)

nemo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "build-essential")
    .pip_install("nemo_toolkit[asr]", "soundfile", "numpy")
    .env({"HF_HOME": "/models/hf"})
    .add_local_python_source("pipeline")
)

llm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub", "hf_xet")
    .env({"HF_HOME": "/models/hf", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

models = modal.Volume.from_name("bgt-models", create_if_missing=True)


def offline_first(load):
    """Load a model from the cached weights in the bgt-models Volume without asking Hugging Face; go online only if
    the cache misses. Keeps containers starting during Hugging Face outages (seen 2026-10-14: API 500s)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return load()
    except Exception:
        import huggingface_hub.constants
        os.environ["HF_HUB_OFFLINE"] = "0"
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        return load()


def decode_bytes(data: bytes, filename: str):
    from pipeline.decode import load_audio

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1]) as f:
        f.write(data)
        f.flush()
        return load_audio(f.name)


WHISPER_MODELS = {"large-v3": "large-v3", "turbo": "large-v3-turbo", "buzzasr": "/models/ct2/buzzasr-bulgarian"}


@app.cls(image=image, gpu="A10G", volumes={"/models": models}, scaledown_window=60, timeout=3600, max_containers=8)
class Transcriber:
    model: str = modal.parameter(default="large-v3")

    @modal.enter()
    def load(self):
        from pipeline.asr import load_model
        self.whisper = offline_first(lambda: load_model(WHISPER_MODELS[self.model], cache_dir="/models/whisper"))

    @modal.method()
    def run(self, data: bytes, filename: str) -> list[dict]:
        from pipeline.asr import transcribe

        return transcribe(self.whisper, decode_bytes(data, filename))

    @modal.method()
    def run_opts(self, data: bytes, filename: str, options: dict) -> dict:
        import time

        from pipeline.asr import transcribe

        started = time.time()
        segments = transcribe(self.whisper, decode_bytes(data, filename), **options)
        return {"segments": segments, "seconds": round(time.time() - started, 1)}

    @modal.method()
    def run_windows(self, data: bytes, filename: str, windows: list[dict], options: dict) -> dict:
        """Transcribe each window [start, end] separately (speech Whisper skipped, see pipeline/rescue.py). Returns
        every word with absolute times."""
        import time

        from pipeline.asr import transcribe

        started = time.time()
        audio = decode_bytes(data, filename)
        sr = 16000
        out = []
        for w in windows:
            piece = audio[int(w["start"] * sr):int(w["end"] * sr)]
            words = []
            for seg in transcribe(self.whisper, piece, **options):
                for word in seg["words"]:
                    a, b = word["start"] + w["start"], word["end"] + w["start"]
                    words.append({**word, "start": round(a, 2), "end": round(b, 2)})
            out.append({**w, "words": words})
        return {"windows": out, "seconds": round(time.time() - started, 1)}


LLM_MODELS = {"bggpt-27b": "INSAIT-Institute/BgGPT-Gemma-3-27B-IT"}


@app.cls(image=llm_image, gpu="L40S", volumes={"/models": models}, scaledown_window=60, timeout=1800, max_containers=1,
         secrets=[modal.Secret.from_name("huggingface-secret")])
class BgGPT:
    """Open Bulgarian LLM (INSAIT BgGPT, Gemma-3 27B in FP8) served with vLLM; deterministic decoding."""
    model: str = modal.parameter(default="bggpt-27b")

    @modal.enter()
    def load(self):
        def load():
            from vllm import LLM
            return LLM(model=LLM_MODELS[self.model], dtype="bfloat16", max_model_len=8192,
                       gpu_memory_utilization=0.9, enforce_eager=True, quantization="fp8")  # bf16 does not fit 48 GB
        self.llm = offline_first(load)

    @modal.method()
    def generate(self, prompts: list[str], max_tokens: int = 1024) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outs = self.llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
        return [o.outputs[0].text for o in outs]


NEMO_MODELS = {"parakeet-v3": "nvidia/parakeet-tdt-0.6b-v3", "canary-v2": "nvidia/canary-1b-v2"}


@app.cls(image=nemo_image, gpu="A10G", volumes={"/models": models}, scaledown_window=60, timeout=3600, max_containers=6)
class NemoTranscriber:
    """NVIDIA NeMo ASR models; output in the same segment/word format as the Whisper Transcriber."""
    model: str = modal.parameter(default="parakeet-v3")

    @modal.enter()
    def load(self):
        def load():
            import nemo.collections.asr as nemo_asr
            return nemo_asr.models.ASRModel.from_pretrained(model_name=NEMO_MODELS[self.model])
        self.asr = offline_first(load)
        self.asr.eval()

    @modal.method()
    def run_opts(self, data: bytes, filename: str, options: dict) -> dict:
        import time

        import soundfile as sf

        started = time.time()
        audio = decode_bytes(data, filename)
        sr = 16000
        piece = 120 * sr if len(audio) > 600 * sr else len(audio)  # long files: 2-minute pieces
        kwargs = {"timestamps": True}
        if self.model.startswith("canary"):
            kwargs.update(source_lang="bg", target_lang="bg")
        segments = []
        for offset in range(0, len(audio), max(1, piece)):
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, audio[offset:offset + piece], sr)
                hyp = self.asr.transcribe([f.name], **kwargs)[0]
            shift = offset / sr
            stamps = getattr(hyp, "timestamp", None) or {}
            words = [{"text": w.get("word", w.get("char", "")), "start": round(float(w["start"]) + shift, 2),
                      "end": round(float(w["end"]) + shift, 2), "confidence": 1.0} for w in stamps.get("word", [])]
            found = []
            for seg in stamps.get("segment", []):
                a, b = float(seg["start"]) + shift, float(seg["end"]) + shift
                found.append({"start": round(a, 2), "end": round(b, 2), "text": seg.get("segment", ""),
                              "words": [w for w in words if a - 0.01 <= w["start"] < b + 0.01]})
            segments += found or [{"start": shift, "end": shift + min(piece, len(audio) - offset) / sr,
                                   "text": hyp.text, "words": words}]
        return {"segments": segments, "seconds": round(time.time() - started, 1)}


SORTFORMER_MODELS = {"sortformer-v1": "nvidia/diar_sortformer_4spk-v1"}


@app.cls(image=nemo_image, gpu="A10G", volumes={"/models": models}, scaledown_window=60, timeout=1200, max_containers=4,
         secrets=[modal.Secret.from_name("huggingface-secret")])
class SortformerDiarizer:
    """NVIDIA Sortformer end-to-end diarization (up to 4 speakers); same turn format as Diarizer."""
    model: str = modal.parameter(default="sortformer-v1")

    @modal.enter()
    def load(self):
        def load():
            from nemo.collections.asr.models import SortformerEncLabelModel
            return SortformerEncLabelModel.from_pretrained(SORTFORMER_MODELS[self.model])
        self.diar = offline_first(load)
        self.diar.eval()

    @modal.method()
    def run(self, data: bytes, filename: str) -> list[dict]:
        import soundfile as sf

        audio = decode_bytes(data, filename)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, 16000)
            segments = self.diar.diarize(audio=[f.name], batch_size=1)[0]
        turns = []
        for seg in segments:
            start, end, speaker = (seg.split() if isinstance(seg, str) else seg)[:3]
            turns.append({"start": round(float(start), 2), "end": round(float(end), 2), "speaker": str(speaker)})
        return sorted(turns, key=lambda t: t["start"])


@app.cls(image=diarize_image, gpu="T4", volumes={"/models": models},
         secrets=[modal.Secret.from_name("huggingface-secret")], scaledown_window=60, timeout=1800, max_containers=6)
class Diarizer:
    @modal.enter()
    def load(self):
        from pipeline.diarize import load_pipeline
        self.pipeline = offline_first(lambda: load_pipeline(os.environ["HF_TOKEN"]))

    @modal.method()
    def run(self, data: bytes, filename: str, exclusive: bool = False) -> list[dict]:
        from pipeline.diarize import diarize

        return diarize(self.pipeline, decode_bytes(data, filename), exclusive=exclusive)


DICT_DIR = Path(__file__).resolve().parent.parent / "data" / "dict"  # Bulgarian hunspell dictionary (LibreOffice)
orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("spylls", "requests")
    .add_local_dir(str(DICT_DIR), "/dict")
    .add_local_python_source("pipeline")
)

# Second opinions: what these models heard is shown to the LLM next to Whisper's text
OTHER_SYSTEMS = [("buzzasr", Transcriber), ("canary-v2", NemoTranscriber), ("parakeet-v3", NemoTranscriber),
                   ("turbo", Transcriber)]
# Whisper keeps its temperature fallback: at temperature 0 long recordings loop ("стой," x50) and lose speech. The
# fallback samples, so the same file can come out slightly different each time.
WHISPER_OPTIONS = {"beam_size": 5}
SORTFORMER_MAX_SECONDS = 15 * 60  # Sortformer is not tested beyond this; longer files use pyannote only


def _scribe_words(data: bytes, filename: str) -> list[dict]:
    """ElevenLabs Scribe's words with timings, as Whisper-style segments (mixed mode only)."""
    import requests

    r = requests.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                      data={"model_id": "scribe_v1", "language_code": "bg", "diarize": "false", "tag_audio_events": "false"},
                      files={"file": (filename, data)}, timeout=1800)
    r.raise_for_status()
    return [{"start": w["start"], "end": w["end"], "text": w["text"],
             "words": [{"text": w["text"], "start": w["start"], "end": w["end"], "confidence": 1.0}]}
            for w in r.json()["words"] if w.get("type") == "word"]


@app.function(image=orchestrator_image, timeout=5400,
              secrets=[modal.Secret.from_name("elevenlabs-secret")])
def transcribe_file(data: bytes, filename: str, mode: str = "local", style: str = "readable") -> dict:
    """Speaker-labelled Bulgarian transcript of one recording, with SRT and WebVTT subtitles.

    mode "local": only open models (Whisper large-v3 words, BgGPT-27B rewrite). mode "mixed": ElevenLabs Scribe words
    and one BgGPT-27B rewrite pass. Both use our speaker detection and the open models as hints. Only the words
    and pyannote are essential: if another part fails, the transcript is still produced and the failure is listed in
    "fallbacks". style "readable": a final light edit (sentences, repeats, agreement); "verbatim": punctuation only."""
    import subprocess
    import time
    from functools import lru_cache

    from pipeline.clean import clean, format_sound_tags
    from pipeline.correct import rewrite
    from pipeline.merge import drop_unvoiced, lines, smooth_flips, words_with_speakers
    from pipeline.polish import polish_words
    from pipeline.rescue import apply as apply_rescue, find_windows
    from pipeline.subtitles import srt, vtt

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1]) as f:
        f.write(data)
        f.flush()
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f.name],
                               capture_output=True, text=True)
    duration = float(probe.stdout.strip() or 0)
    started, timings, fallbacks = time.time(), {}, []

    def optional(name: str, call):
        """Result of an optional component, or None (noted in fallbacks) if it failed."""
        try:
            return call()
        except Exception as e:
            fallbacks.append(f"{name}: {type(e).__name__}")
            return None

    pyannote = Diarizer().run.spawn(data, filename, exclusive=True)
    sortformer = (SortformerDiarizer(model="sortformer-v1").run.spawn(data, filename)
                  if duration <= SORTFORMER_MAX_SECONDS else None)
    whisper = Transcriber(model="large-v3").run_opts.spawn(data, filename, WHISPER_OPTIONS)
    others = {name: engine(model=name).run_opts.spawn(data, filename, WHISPER_OPTIONS if engine is Transcriber else {})
              for name, engine in OTHER_SYSTEMS}

    turns = pyannote.get()
    turns_sf = optional("sortformer", sortformer.get) if sortformer is not None else None
    if turns_sf is not None and len({t["speaker"] for t in turns_sf}) >= len({t["speaker"] for t in turns}):
        turns = turns_sf
    timings["speakers"] = round(time.time() - started)

    if mode == "mixed":
        segments = _scribe_words(data, filename)
    else:
        segments = whisper.get()["segments"]
        windows = find_windows(segments, turns)  # detected speech where Whisper drifted and wrote (almost) nothing
        if windows:
            rescued = optional("rescue", lambda: Transcriber(model="large-v3").run_windows.remote(
                data, filename, windows, {**WHISPER_OPTIONS, "condition_on_previous_text": False})["windows"])
            segments = apply_rescue(segments, rescued)[0] if rescued is not None else segments
    words = smooth_flips(drop_unvoiced(words_with_speakers(segments, turns), turns, tolerance=30.0))
    timings["words"] = round(time.time() - started)

    hyps = {}
    for name, call in others.items():
        out = optional(name, call.get)
        if out is not None:
            hyps[name] = " ".join(s["text"] for s in out["segments"]).split()
    timings["hints"] = round(time.time() - started)

    rewrite_log = {}
    if hyps:
        from spylls.hunspell import Dictionary

        d = Dictionary.from_files("/dict/bg_BG")
        known = lru_cache(None)(lambda w: bool(d.lookup(w) or d.lookup(w.lower())))
        if mode == "mixed":  # Scribe's words are already good: one pass, Whisper as one more hint, no line swaps
            whisper_words = optional("whisper", whisper.get)
            if whisper_words is not None:
                hyps = {"whisper": " ".join(s["text"] for s in whisper_words["segments"]).split(), **hyps}
            out = optional("rewrite", lambda: rewrite(words, hyps, known, BgGPT(model="bggpt-27b").generate.remote,
                                                      passes=1, outlier_similarity=0, repair=False))
        else:
            out = optional("rewrite", lambda: rewrite(words, hyps, known, BgGPT(model="bggpt-27b").generate.remote))
        if out is not None:
            words, rewrite_log = out
    timings["rewrite"] = round(time.time() - started)
    polished = optional("polish", lambda: polish_words(words, BgGPT(model="bggpt-27b").generate.remote, style))
    polish_log = {}
    if polished is not None:
        words, polish_log = polished
    timings["polish"] = round(time.time() - started)
    words = clean(words, hyps)  # broken characters, other scripts, Latin no other system heard, loops

    result = lines(words)
    names = {}
    for line in result:
        line["speaker"] = names.setdefault(line["speaker"], f"Говорител {len(names) + 1}")
        line["text"] = format_sound_tags(line["text"])
    return {"mode": mode, "style": style, "duration": duration, "speakers": len(names), "lines": result,
            "srt": srt(result), "vtt": vtt(result), "timings": timings, "fallbacks": fallbacks, "rewrite": rewrite_log,
            "polish": polish_log}


uploads = modal.Volume.from_name("bgt-uploads", create_if_missing=True)
web_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]", "python-multipart")
PAGE = Path(__file__).resolve().parent / "web" / "index.html"


@app.function(image=orchestrator_image, timeout=5400, volumes={"/uploads": uploads},
              secrets=[modal.Secret.from_name("elevenlabs-secret")])
def transcribe_upload(name: str, mode: str, style: str = "readable") -> dict:
    """A file the web page uploaded to the Volume (large files never travel as function arguments). The upload is
    deleted as soon as it is read: recordings are not kept, only the log entry of the job."""
    import json
    import time

    uploads.reload()
    with open(f"/uploads/{name}", "rb") as f:
        data = f.read()
    os.remove(f"/uploads/{name}")
    uploads.commit()
    entry, started = {"file": name}, time.time()
    try:
        result = transcribe_file.local(data, name, mode, style)
        entry.update(status="done", audio_seconds=round(result["duration"]), speakers=result["speakers"],
                     words=sum(len(line["text"].split()) for line in result["lines"]), fallbacks=result["fallbacks"])
        return result
    except Exception as e:
        entry.update(status="error", error=type(e).__name__)
        raise
    finally:
        entry["took_seconds"] = round(time.time() - started)
        with open(f"/uploads/log/{name}.done.json", "w") as f:  # own file: the web container owns <name>.json
            json.dump(entry, f, ensure_ascii=False)
        uploads.commit()


@app.function(image=web_image.add_local_file(str(PAGE), "/page/index.html"), volumes={"/uploads": uploads})
@modal.asgi_app()
def web():
    import re
    import uuid

    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse

    api = FastAPI()

    @api.get("/", response_class=HTMLResponse)
    def index():
        return open("/page/index.html").read()

    @api.post("/jobs")
    def start(request: Request, file: UploadFile = File(...), mode: str = Form("local"), verbatim: str = Form("")):
        import json
        from datetime import datetime
        from zoneinfo import ZoneInfo

        safe = re.sub(r"[^\w.-]", "_", file.filename or "audio")  # no backslash inside the f-string (Python < 3.12)
        name = f"{uuid.uuid4().hex[:8]}_{safe}"
        with open(f"/uploads/{name}", "wb") as out:
            while chunk := file.file.read(1 << 20):
                out.write(chunk)
        mode, style = ("mixed" if mode == "mixed" else "local"), ("verbatim" if verbatim else "readable")
        os.makedirs("/uploads/log", exist_ok=True)  # one small file per job: who uploaded what (the audio is not kept)
        with open(f"/uploads/log/{name}.json", "w") as f:
            json.dump({"time": datetime.now(ZoneInfo("Europe/Sofia")).strftime("%Y-%m-%d %H:%M:%S"), "file": file.filename,
                       "bytes": os.path.getsize(f"/uploads/{name}"), "mode": mode, "style": style,
                       "ip": request.headers.get("x-forwarded-for", request.client.host if request.client else ""),
                       "browser": request.headers.get("user-agent", "")}, f, ensure_ascii=False)
        uploads.commit()
        call = transcribe_upload.spawn(name, mode, style)
        return {"id": call.object_id}

    @api.get("/jobs/{job_id}")
    def status(job_id: str):
        call = modal.FunctionCall.from_id(job_id)
        try:
            return {"status": "done", "result": call.get(timeout=0)}
        except TimeoutError:
            return {"status": "running"}
        except Exception as e:
            return {"status": "error", "error": type(e).__name__}

    return api


@app.local_entrypoint()
def main(file: str, mode: str = "local", style: str = "readable"):
    path = Path(file)
    out = transcribe_file.remote(path.read_bytes(), path.name, mode, style)
    print(f"# mode {out['mode']}, duration {out['duration']:.0f}s, speakers {out['speakers']}, timings (s since start) "
          f"{out['timings']}, fallbacks {out['fallbacks'] or 'none'}, rewrite {out['rewrite']}")
    path.with_suffix(f".{mode}.srt").write_text(out["srt"])
    for line in out["lines"]:
        m, s = divmod(int(line["start"]), 60)
        print(f"[{m:02d}:{s:02d}] {line['speaker']}: {line['text']}")
