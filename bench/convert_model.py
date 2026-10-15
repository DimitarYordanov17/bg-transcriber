"""One-time conversion of a Hugging Face Whisper fine-tune to CTranslate2 (for faster-whisper), stored in bgt-models.

Usage: uv run modal run bench/convert_model.py --repo BuzzASR/bulgarian --name buzzasr-bulgarian
"""
import modal

app = modal.App("bg-transcriber-convert")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "ctranslate2==4.6.0", "transformers[torch]==4.56.2", "huggingface_hub")
models = modal.Volume.from_name("bgt-models")


@app.function(image=image, volumes={"/models": models}, timeout=1800, memory=16384, cpu=4)
def convert(repo: str, name: str) -> list[str]:
    import os
    import subprocess

    out = f"/models/ct2/{name}"
    if os.path.exists(f"{out}/model.bin"):
        return os.listdir(out)
    from huggingface_hub import list_repo_files

    extra = [f for f in ("tokenizer.json", "preprocessor_config.json") if f in list_repo_files(repo)]  # older repos lack some
    subprocess.run(["ct2-transformers-converter", "--model", repo, "--output_dir", out, "--quantization", "float16"]
                   + (["--copy_files", *extra] if extra else []), check=True)
    models.commit()
    return os.listdir(out)


@app.local_entrypoint()
def main(repo: str, name: str):
    print(convert.remote(repo, name))
