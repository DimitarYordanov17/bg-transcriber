# bg-transcriber

Speaker-labelled transcription of Bulgarian recordings (meetings, interviews, lectures), with SRT/VTT subtitles and a
small web page. Runs on [Modal](https://modal.com).

Two modes:

- **Fully local** — open models only, nothing leaves our GPUs: Whisper large-v3 for the words, pyannote and NVIDIA
  Sortformer for the speakers, Canary, Parakeet, BuzzASR and Whisper turbo as second opinions, BgGPT-27B (INSAIT) to
  correct the text.
- **Mixed** — ElevenLabs Scribe for the words, then one BgGPT correction pass; everything else the same.

## How it works

1. **Speakers.** pyannote and Sortformer both run; the one that finds more speakers is used (Sortformer only up to
   15 minutes). Each word gets the speaker talking at its timestamp.
2. **Words.** Whisper large-v3. Stretches where speech was detected but Whisper wrote almost nothing are decoded
   again; Whisper text that lies far from any detected speech is dropped.
3. **Second opinions.** Canary, Parakeet, BuzzASR and Whisper turbo transcribe the same audio. A Whisper line that none
   of them heard (a hallucination loop, for example) is replaced by their most common version.
4. **Correction.** BgGPT-27B rewrites every line in two passes. The prompt shows what the other models heard and which
   words are not in the Bulgarian dictionary. A correction is rejected if it lands on another line, deletes speech
   another model heard, or adds a negation.
5. **Broken words.** Every word still missing from the dictionary gets one focused question to the LLM; the answer is
   used only if it is a real word and sounds like the broken one. Names are never changed.
6. **Punctuation and cleanup.** Punctuation and capital letters, loops and broken characters removed. By default a
   light reading edit follows (false starts and repeats removed); "verbatim" skips it. Words are realigned to their
   timestamps so subtitles stay in sync.

## Results

Word error rate on test sets that were not used for tuning (about 10,000 words of real Bulgarian speech), verbatim
output. The reading edit adds about 1 point:

| | Whisper large-v3 alone | Fully local | Mixed | ElevenLabs Scribe alone |
|---|---|---|---|---|
| Parliament sessions (official stenograms) | 18.5% | 7.5% | 4.9% | 5.1% |
| Podcasts and interviews (human subtitles) | 15.2% | 12.7% | 9.3% | 8.8% |

The fully local pipeline makes 2.5x fewer errors than the best open model on formal speech. On casual speech the
limit is the speech model itself: even picking the best of the five models for every stretch would not reach Scribe.

Names are the weak spot of every model here. Unknown surnames are guessed by sound.

## Running

```
uv sync
modal deploy modal_app.py                          # web page + transcription
modal run modal_app.py --file meeting.m4a          # one file from the command line (--mode mixed, --style verbatim)
```

Modal secrets: `huggingface-secret` (HF_TOKEN with access to pyannote/speaker-diarization-community-1) and
`elevenlabs-secret` (mixed mode only). BuzzASR has to be converted to CTranslate2 once: `bench/convert_model.py`.

## Benchmarks

`bench/` holds the evaluation: scoring (`score.py`), the experiment runner with cached model outputs
(`experiment.py`), paired comparison with bootstrap intervals (`compare.py`), and the builders for the test sets
(`parliament.py`: National Assembly stenograms aligned to the session audio; `youtube.py`: videos with human-made
subtitles).
