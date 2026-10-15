"""Commercial speech-to-text services used as benchmarks (never part of the product).

Each returns speaker-labelled words [{text, start, end, speaker}] using the service's own diarization.
Keys come from ../.env (OPENAI_API_KEY, DEEPGRAM_API_KEY). OpenAI spend is tracked in runs/openai_spend.json and
calls are refused past the hard stop.
"""
import json
import mimetypes
import os

import threading

import requests

from bench.paths import ROOT, RUNS

OPENAI_SPEND = RUNS / "openai_spend.json"
OPENAI_HARD_STOP = 4.42  # tracked spend; = 2.47 before 2026-10-15 + 1.95 the account has left
# USD per 1M tokens, conservative: audio input, text input, output
OPENAI_PRICES = {"audio": 6.0, "text": 2.5, "output": 10.0}


def load_keys() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def deepgram(data: bytes, filename: str) -> list[dict]:
    load_keys()
    mime = mimetypes.guess_type(filename)[0] or "audio/wav"
    for attempt in range(4):  # network errors happen (a write timeout lost a whole test run once)
        try:
            r = requests.post("https://api.deepgram.com/v1/listen",
                              params={"model": "nova-3", "language": "bg", "diarize": "true", "smart_format": "true",
                                      "punctuate": "true"},
                              headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": mime},
                              data=data, timeout=(30, 600))
            r.raise_for_status()
            break
        except requests.exceptions.ConnectionError:
            if attempt == 3:
                raise
            import time
            time.sleep(5 * (attempt + 1))
    words = r.json()["results"]["channels"][0]["alternatives"][0]["words"]
    return [{"text": w.get("punctuated_word", w["word"]), "start": w["start"], "end": w["end"],
             "speaker": f"DG_{w.get('speaker', 0)}"} for w in words]


def _openai_spend(usage: dict, what: str) -> float:
    spend = json.loads(OPENAI_SPEND.read_text()) if OPENAI_SPEND.exists() else {"usd": 0.0, "calls": []}
    details = usage.get("input_token_details", {})
    usd = (details.get("audio_tokens", 0) * OPENAI_PRICES["audio"] + details.get("text_tokens", 0)
           * OPENAI_PRICES["text"] + usage.get("output_tokens", 0) * OPENAI_PRICES["output"]) / 1e6
    spend["usd"] = round(spend["usd"] + usd, 4)
    spend["calls"].append({"what": what, "usd": round(usd, 4)})
    OPENAI_SPEND.write_text(json.dumps(spend, indent=1))
    return spend["usd"]


def openai_diarize(data: bytes, filename: str) -> list[dict]:
    """gpt-4o-transcribe-diarize: segments with speakers; words get times spread evenly inside their segment."""
    load_keys()
    spent = json.loads(OPENAI_SPEND.read_text())["usd"] if OPENAI_SPEND.exists() else 0.0
    if spent >= OPENAI_HARD_STOP:
        raise RuntimeError(f"OpenAI hard stop reached (${spent:.2f})")
    r = requests.post("https://api.openai.com/v1/audio/transcriptions",
                      headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                      data={"model": "gpt-4o-transcribe-diarize", "language": "bg",
                            "response_format": "diarized_json", "chunking_strategy": "auto"},
                      files={"file": (filename, data)}, timeout=600)
    r.raise_for_status()
    d = r.json()
    _openai_spend(d.get("usage", {}), f"diarize {filename}")
    words = []
    for seg in d["segments"]:
        tokens = seg["text"].split()
        step = (seg["end"] - seg["start"]) / max(1, len(tokens))
        for i, t in enumerate(tokens):
            words.append({"text": t, "start": round(seg["start"] + i * step, 2),
                          "end": round(seg["start"] + (i + 1) * step, 2), "speaker": f"OA_{seg['speaker']}"})
    return words


def elevenlabs_credits() -> int:
    load_keys()
    r = requests.get("https://api.elevenlabs.io/v1/user/subscription",
                     headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}, timeout=30)
    r.raise_for_status()
    return r.json()["character_count"]


ELEVENLABS_STOP = 9950  # credits used; the free tier has 10,000


def elevenlabs(data: bytes, filename: str) -> list[dict]:
    """ElevenLabs Scribe with its own diarization."""
    used = elevenlabs_credits()
    if used >= ELEVENLABS_STOP:
        raise RuntimeError(f"ElevenLabs credit stop reached ({used} used)")
    r = requests.post("https://api.elevenlabs.io/v1/speech-to-text",
                      headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                      data={"model_id": "scribe_v1", "language_code": "bg", "diarize": "true",
                            "tag_audio_events": "false"},
                      files={"file": (filename, data)}, timeout=600)
    r.raise_for_status()
    return [{"text": w["text"], "start": w["start"], "end": w["end"], "speaker": f"EL_{w.get('speaker_id', 0)}"}
            for w in r.json()["words"] if w.get("type") == "word"]


PROVIDERS = {"deepgram": deepgram, "openai-diarize": openai_diarize, "elevenlabs": elevenlabs}


SPEND_LOCK = threading.Lock()  # openai_chat is called from several threads
LLM_PRICES = {"gpt-5.4-mini": (1.0, 5.0), "gpt-5.4": (3.0, 15.0), "gpt-5.5": (5.0, 20.0)}  # USD per 1M in/out, conservative


def openai_chat(prompt: str, model: str = "gpt-5.4-mini", json_mode: bool = True) -> str:
    """One chat completion, spend tracked against the same OpenAI hard stop."""
    load_keys()
    with SPEND_LOCK:
        spent = json.loads(OPENAI_SPEND.read_text())["usd"] if OPENAI_SPEND.exists() else 0.0
    if spent >= OPENAI_HARD_STOP:
        raise RuntimeError(f"OpenAI hard stop reached (${spent:.2f})")
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                      json={"model": model, "messages": [{"role": "user", "content": prompt}],
                            **({"response_format": {"type": "json_object"}} if json_mode else {})}, timeout=300)
    r.raise_for_status()
    d = r.json()
    usage = d.get("usage", {})
    price_in, price_out = LLM_PRICES.get(model, (5.0, 20.0))
    usd = (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6
    with SPEND_LOCK:
        spend = json.loads(OPENAI_SPEND.read_text()) if OPENAI_SPEND.exists() else {"usd": 0.0, "calls": []}
        spend["usd"] = round(spend["usd"] + usd, 4)
        spend["calls"].append({"what": f"chat {model}", "usd": round(usd, 4)})
        OPENAI_SPEND.write_text(json.dumps(spend, indent=1))
    return d["choices"][0]["message"]["content"]
