"""Modal app for CogVideoX1.5-5B-I2V (Zhipu/THUDM), image-to-video --
backs `vitals/adapters/cogvideox.py`. Chosen for architectural diversity
at the CHEAPEST tier of any real model here (bf16 9GB with offload, per
the model card -> A10G), and for being the open model most often ranked
best on prompt adherence.

**License restriction, stated plainly (confirmed directly against the
repo's own LICENSE file, 2026-09):** the "CogVideoX LICENSE" is NOT a
permissive license. Academic research use is free; commercial use
requires registering for a basic commercial license and is capped at
1M visits/month beyond which a further license is needed; governing law
is the PRC (Haidian District People's Court, Beijing); prohibited uses
include military use and activities against PRC national security/
unity. By this project's own standard -- Hunyuan's territorial clause
made IT `restricted=True` -- this is registered `restricted=True` and
needs `VITALS_ENABLE_RESTRICTED_MODELS=cogvideox15` per run. Do not
remove that gate to make it "just work".

Facts confirmed against the `THUDM/CogVideoX1.5-5B-I2V` model card:
- Pipeline `CogVideoXImageToVideoPipeline`, `torch_dtype=bfloat16`.
- Native: 81 frames @ 16 fps (the card's export example writes at 8 fps
  -- a display choice; 16 is the stated native rate). "Works best with
  81 and 161 frames" -> frame counts follow (n-1) % 16 == 0.
- Resolution: min(W,H)=768, max(W,H) in [768,1360], %16==0. This
  project's 4:3 renders map to 1024x768.
- Memory: card's own recipe is `enable_sequential_cpu_offload()` +
  `vae.enable_tiling()` + `vae.enable_slicing()` -> "BF16: 9GB minimum".
  Used as-is on A10G (24GB); a first real call is the debugging pass.
- Not documented as gated; no HF token wired. If the download 401s,
  that is the first thing to revisit.

Usage:
    modal deploy remote/modal_app_cogvideox.py
    python3 remote/call_cogvideox.py download_checkpoints
    python3 remote/call_cogvideox.py image2video --input-path frame.png --prompt "..." --save-path out.mp4
"""
import modal

GPU_TYPE = "A10G"
MODEL_ID = "THUDM/CogVideoX1.5-5B-I2V"

app = modal.App("vitals-cogvideox")

cogvideox_cache = modal.Volume.from_name("vitals-cogvideox-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "diffusers>=0.32.0",   # CogVideoX1.5 I2V support is in released diffusers (v0.30.3+ added I2V)
        "transformers>=4.45",
        "accelerate",
        "sentencepiece",
        "pillow",
        "numpy",
    )
    .env({"HF_HOME": "/cache/hf"})
)


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": cogvideox_cache}, timeout=1800)
def download_checkpoints():
    """One-time setup: pulls the checkpoint into the persistent volume."""
    import torch
    from diffusers import CogVideoXImageToVideoPipeline

    CogVideoXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    cogvideox_cache.commit()
    return "CHECKPOINTS_DOWNLOADED"


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": cogvideox_cache}, timeout=3600)
def image2video(image, prompt: str, seed: int = 0, num_frames: int = 81,
                height: int = 768, width: int = 1024,
                num_inference_steps: int = 50, guidance_scale: float = 6.0):
    """Runs ONE CogVideoX1.5 image-to-video generation, returns frames as
    a raw (T,H,W,3) uint8 numpy array.

    Memory recipe, revised after a REAL measurement (2026-09): the model
    card's own `enable_sequential_cpu_offload()` (its "9GB minimum")
    streams every layer over PCIe on every denoising step -- the first
    real call on A10G hit the 1800s timeout without finishing (50 steps x
    81 frames x 1024x768). Switched to model-level `enable_model_cpu_
    offload()` (one component resident at a time, far fewer transfers;
    expected ~16-20GB, inside A10G's 24GB), VAE tiling/slicing kept, and
    the timeout raised to 3600s so the next call can actually be TIMED
    rather than cut off. If it still doesn't fit or finish, escalate
    straight to A100-80GB with no offload (a known-working tier) rather
    than a second intermediate guess."""
    import numpy as np
    import torch
    from diffusers import CogVideoXImageToVideoPipeline
    from PIL import Image

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    pil_image = Image.fromarray(np.asarray(image))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        prompt=prompt, image=pil_image, num_videos_per_prompt=1,
        num_frames=num_frames, height=height, width=width,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        generator=generator, output_type="np",
    ).frames[0]
    cogvideox_cache.commit()

    frames = np.asarray(out)
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    return frames
