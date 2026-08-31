"""Modal app for Wan2.1 (Alibaba), Image2Video-14B-480P inference
(AGENT.md M6) -- backs `vitals/adapters/wan.py`'s own `generate_fn`. The
SECOND model's own GPU infrastructure, deliberately its own separate
Modal app from both `vitals-phi` (SAM2/DINOv2) and `vitals-cosmos`
(Cosmos-Predict2) -- each model's own dependency stack kept isolated, the
same "one small file per model" discipline as everywhere else in
`vitals/adapters/` and `remote/`.

Real, concrete advantages over the Cosmos integration, confirmed directly
before building this, not assumed:
- NOT gated. No HF license-acceptance step, no `huggingface-token` Modal
  secret needed -- confirmed on the model's own Hugging Face page.
- ~16GB VRAM (vs. Cosmos's 32.54GB), confirmed on the model card.
- Native `diffusers` integration -- a real Python class
  (`WanImageToVideoPipeline`), not a CLI wrapped in `subprocess`. No CUDA
  "devel" base image needed either: unlike Cosmos (whose `transformer_
  engine` dependency needed the real CUDA toolkit's `libnvrtc` at import
  time), plain `diffusers`+`torch` pip wheels are self-contained --
  `debian_slim` should be sufficient here. Treat the FIRST deploy as a
  real debugging pass regardless (same honest caveat every image in this
  project has needed at least once so far).

GPU tier: **A100-80GB**, NOT L4 -- L4 (24GB) was the original choice
(cheaper than A10G at equal VRAM, checked Modal's real pricing before
picking), based on the model card's own "~16GB" figure. That figure was
WRONG for this actual pipeline, confirmed by real OOM, not guessed: just
loading the full pipeline onto the GPU (CLIP image encoder + VAE in
fp32, the 14B transformer in bf16, nothing offloaded) already consumed
~22GB before a single inference step ran, on a 24GB card with ~22GB
usable. Rather than guess a second, narrower tier and risk a THIRD
debug cycle, moved straight to A100-80GB -- already a known-working tier
from the Cosmos integration, so this removes tier uncertainty entirely
rather than trading a small amount of headroom for another guess.

Usage:
    modal deploy remote/modal_app_wan.py
    python3 remote/call_wan.py image2video --input-path prefix_last_frame.png --prompt "..." --save-path out.mp4
"""
import modal

GPU_TYPE = "A100-80GB"
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

app = modal.App("vitals-wan")

# SEPARATE from vitals-cosmos's own cache volume -- different model,
# different weights, no reason to share a cache namespace.
wan_cache = modal.Volume.from_name("vitals-wan-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "diffusers>=0.31",
        "transformers>=4.45",
        "accelerate",
        "ftfy",
        "imageio[ffmpeg]",   # only for _resample_to_target's own PIL/array handling on the caller side, not needed here directly, but harmless to have available for any local debugging inside the container
        "pillow",
    )
    .env({"HF_HOME": "/cache/hf"})
)


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": wan_cache}, timeout=1800)
def download_checkpoints():
    """One-time setup -- `from_pretrained` auto-downloads from the Hugging
    Face Hub into HF_HOME (the persistent volume) on first use; this just
    forces that download to happen once, up front, with a generous
    timeout, rather than eating a cold-download penalty (and a shorter
    default timeout) on the first real `image2video` call. No token, no
    gating -- confirmed directly on the model's own HF page (module
    docstring)."""
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)

    wan_cache.commit()
    return "CHECKPOINTS_DOWNLOADED"


def _aspect_ratio_resize(image, pipe, max_area=480 * 832):
    """Resizes to Wan2.1's own required (mod_value-aligned) dimensions
    while PRESERVING the input image's own aspect ratio -- this project's
    own frames are 320x240 (4:3), not Wan2.1's own 480x832 (16:9) default,
    and forcing the default would distort every frame. Copied from
    Wan2.1's own documented usage pattern (diffusers docs' own image-to-
    video example), not invented ad hoc."""
    import numpy as np
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": wan_cache}, timeout=1200)
def image2video(image, prompt: str, seed: int = 0, guidance_scale: float = 5.0,
                num_inference_steps: int = 50):
    """Runs ONE Wan2.1 image-to-video generation, returns the generated
    frames as a raw (T,H,W,3) uint8 numpy array -- NOT a video file
    (Wan2.1's own `diffusers` pipeline returns frames directly,
    `output_type="np"`, a real structural difference from Cosmos's
    CLI/file-based interface, not an inconsistency to paper over).

    image: a plain (H,W,3) uint8 numpy array -- converted to PIL here,
    not by the caller, so the caller (running locally, no torch/diffusers
    installed) never needs those dependencies."""
    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel
    from PIL import Image

    image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    pil_image = Image.fromarray(np.asarray(image))
    pil_image, height, width = _aspect_ratio_resize(pil_image, pipe)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(
        image=pil_image, prompt=prompt, height=height, width=width,
        guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]

    wan_cache.commit()
    # output is a list of (H,W,3) float arrays in [0,1] (diffusers'
    # own "np" output_type convention) -- convert to the uint8 [0,255]
    # convention every other frame array in this project already uses
    # (Frames' own docstring: "(T,H,W,3) uint8").
    frames = (np.stack(output) * 255).clip(0, 255).astype(np.uint8)
    return frames
