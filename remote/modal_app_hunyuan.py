"""Modal app for HunyuanVideo-1.5 (Tencent), Image2Video-480p inference --
backs `vitals/adapters/hunyuan.py`'s own `generate_fn`. The THIRD model's
own GPU infrastructure, its own separate Modal app from `vitals-phi`
(SAM2/DINOv2), `vitals-cosmos`, and `vitals-wan` -- same "one small file
per model" discipline as everywhere else in `vitals/adapters/`/`remote/`.

**License: `tencent-hunyuan-community` -- restricted in `MODEL_REGISTRY`
(vitals/adapters/__init__.py), NOT enabled by default.** Confirmed
directly against the actual license text before building this, not
assumed: it does not apply in the European Union, United Kingdom, or
South Korea (a blanket territorial exclusion, not scale-dependent), and
separately requires a Tencent license above 100M MAU for a deployed
product (almost certainly irrelevant to using this as a benchmark
adapter, not a deployed product, but stated here for completeness). See
ADDING_A_MODEL.md's own "Restricted-license models" section for how the
opt-in (`VITALS_ENABLE_RESTRICTED_MODELS=hunyuan`) works.

**Chosen over the original 13B HunyuanVideo-I2V specifically for VRAM,**
confirmed directly against both model cards, not assumed: the original
needs 60-80GB (would force the same A100-80GB tier as Wan/Cosmos-14B);
HunyuanVideo-1.5 is a newer, smaller 8.3B rebuild documented at ~14GB
with model-offloading enabled -- Tencent's own stated design goal is
"efficient inference on consumer-grade GPUs," a genuinely different cost
profile from every other real model in this project so far, worth
actually testing.

**GPU tier: A10G (24GB), a reasoned starting choice, not a confirmed
number -- treat the first real call here as a real debugging pass,
same as every other model integration in this project.** 14GB stated +
model offloading leaves meaningful headroom on a 24GB card (a much
better margin than Wan's own first attempt had on a 24GB L4 against ITS
card's wrong "~16GB" claim). The real risk this stated figure may not
account for: HunyuanVideo-1.5's own text encoder is Qwen2.5-VL-7B-
Instruct, a 7B-parameter vision-language model, not a small CLIP encoder
like Wan's -- co-resident with the 8.3B transformer and VAE, actual peak
VRAM could plausibly exceed the stated 14GB. If the first real call OOMs,
escalate straight to A100-80GB (Wan's own already-known-working tier)
rather than guessing a second intermediate size.

STATUS (2026-09): first real `download_checkpoints` call failed with
huggingface_hub's offline-mode error. Traced to a WRONG `MODEL_ID` below,
not gating: `hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v` 404s (HF's
API returns 401 "Invalid username or password" for a nonexistent repo
path too, not just for actually-gated ones -- confirmed by hitting both
paths directly). The real repo is
`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v` (note
`-Diffusers-`) -- confirmed 200 on a plain anonymous GET against its API
endpoint, so the module's original "Not gated" claim above was correct
all along; no HF_TOKEN/Modal Secret needed here.

Usage:
    modal deploy remote/modal_app_hunyuan.py
    python3 remote/call_hunyuan.py download_checkpoints
    python3 remote/call_hunyuan.py image2video --input-path prefix_last_frame.png --prompt "..." --save-path out.mp4
"""
import modal

GPU_TYPE = "A10G"
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"

app = modal.App("vitals-hunyuan")

# SEPARATE from vitals-cosmos's/vitals-wan's own cache volumes -- different
# model, different weights, no reason to share a cache namespace.
hunyuan_cache = modal.Volume.from_name("vitals-hunyuan-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "diffusers>=0.40.0",   # HunyuanVideo15ImageToVideoPipeline needs this; confirmed released on PyPI, not main-only
        "transformers>=4.45",
        "accelerate",
        "sentencepiece",       # Qwen2.5-VL / T5 tokenizers both need this
        "ftfy",
        "pillow",
    )
    .env({"HF_HOME": "/cache/hf"})
)


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": hunyuan_cache}, timeout=1800)
def download_checkpoints():
    """One-time setup -- `from_pretrained` auto-downloads from the Hugging
    Face Hub into HF_HOME (the persistent volume) on first use; this just
    forces that download to happen once, up front, with a generous
    timeout, rather than eating a cold-download penalty (and a shorter
    default timeout) on the first real `image2video` call. Not gated:
    confirmed 200 on an anonymous GET against the model's own HF API
    endpoint (see module docstring's STATUS note)."""
    import torch
    from diffusers import HunyuanVideo15ImageToVideoPipeline

    HunyuanVideo15ImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

    hunyuan_cache.commit()
    return "CHECKPOINTS_DOWNLOADED"


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": hunyuan_cache}, timeout=1200)
def image2video(image, prompt: str, seed: int = 0, num_frames: int = 121,
                num_inference_steps: int = 50):
    """Runs ONE HunyuanVideo-1.5 image-to-video generation, returns the
    generated frames as a raw (T,H,W,3) uint8 numpy array.

    image: a plain (H,W,3) uint8 numpy array -- converted to PIL here, not
    by the caller, so the caller (running locally, no torch/diffusers
    installed) never needs those dependencies.

    No `guidance_scale` argument here, unlike Wan/Cosmos -- confirmed
    directly from the pipeline's own `__call__` signature: HunyuanVideo1.5
    controls guidance through a `pipe.guider` object instead of a runtime
    kwarg (a newer diffusers "Guider" pattern), and left at its own
    default (ClassifierFreeGuidance, guidance_scale=6.0) here rather than
    reached into, since nothing in this project's own protocol requires
    tuning it away from the model's own default."""
    import numpy as np
    import torch
    from diffusers import HunyuanVideo15ImageToVideoPipeline
    from PIL import Image

    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    # Memory-saving path (offloading + VAE tiling), not straight `.to("cuda")`
    # -- see module docstring on why 14GB-stated is not fully trusted yet.
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    pil_image = Image.fromarray(np.asarray(image))

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(
        image=pil_image, prompt=prompt, num_frames=num_frames,
        num_inference_steps=num_inference_steps, generator=generator,
    ).frames[0]

    hunyuan_cache.commit()
    # output_type defaults to "np" -- (T,H,W,3) float in [0,1], same
    # convention Wan's own pipeline returns; convert to uint8 [0,255]
    # (Frames' own docstring: "(T,H,W,3) uint8").
    frames = (np.stack(output) * 255).clip(0, 255).astype(np.uint8)
    return frames
