"""Replicate VLM (image captioning) client for asset analysis.

Mirrors the text contract of vlm_dashscope: takes a prompt + image and returns
a description string. The model ref carries an optional ``replicate:`` prefix,
e.g. ``replicate:lucataco/qwen2-vl-7b-instruct``.

ponytail: assumes the common {image, prompt} input shape that most Replicate
VLMs accept (qwen2-vl, llava, blip, moondream). A model with exotic param
names needs an elif here. Video is not supported — Replicate VLMs are image-only.
"""

import os


def _strip_prefix(model: str) -> str:
    return model.split("replicate:", 1)[-1].strip()


def query_vlm(api_token: str, model_ref: str, prompt: str,
              image_paths: list[str], local_proxy: str = None) -> str:
    if not api_token:
        raise RuntimeError("Replicate API token is not configured.")
    if not image_paths:
        raise RuntimeError("Replicate VLM analysis requires an image.")

    import replicate  # lazy: only import when a replicate model is used

    ref = _strip_prefix(model_ref)
    first = image_paths[0]

    inputs = {"prompt": prompt}
    opened = None
    if first.startswith("http"):
        inputs["image"] = first
    elif os.path.exists(first):
        opened = open(first, "rb")
        inputs["image"] = opened
    else:
        raise FileNotFoundError(f"Image not found for Replicate VLM: {first}")

    try:
        client = replicate.Client(api_token=api_token)
        output = client.run(ref, input=inputs)
    finally:
        if opened:
            opened.close()

    # Output is typically a list of streamed string chunks, or a single string.
    if isinstance(output, list):
        return "".join(str(chunk) for chunk in output).strip()
    return str(output).strip()
