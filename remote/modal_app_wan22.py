"""Modal app for Wan2.2 I2V-A14B (Alibaba), image-to-video -- backs
`vitals/adapters/wan22.py`. Its OWN app, separate from `vitals-wan`
(Wan2.1): 2.2 needs diffusers from git main (model card: "features
currently available only in the main branch"), while the working 2.1
app pins a released diffusers -- upgrading in place would risk the one
wan population path that finally works (see AGENT.md's wan timeout
entry). Separate cache volume too.

Facts confirmed against the `Wan-AI/Wan2.2-I2V-A14B-Diffusers` model
card (2026-09), not assumed. Note "Wan 2.7" (seen in blog roundups) does
NOT exist as open weights on the Hub -- 2.2 is the current open I2V
line; a "wan3" exists only behind Runway's API gateway.
- Pipeline `WanImageToVideoPipeline`, a SINGLE `from_pretrained(...,
  torch_dtype=bfloat16)` -- unlike 2.1's manual image_encoder/vae
  loading. A14B = a two-expert MoE (high/low-noise), 14B active.
- Card example: num_frames=81, guidance_scale=3.5, num_inference_steps
  =40, export fps=16 -> native 16 fps, same as 2.1.
- License Apache 2.0, not gated -> `restricted=False`.
- "80GB recommended for single-GPU inference" -> A100-80GB, the tier
  Wan2.1 already runs on here.
- Latency UNMEASURED. Wan2.1 measured 27:46/call on this tier; a two-
  expert 14B is plausibly similar or slower. Server timeout 2400s
  (matching the wan fix), client default above it. Replace with a real
  timed call, same discipline.

Usage:
    modal deploy remote/modal_app_wan22.py
    python3 remote/call_wan22.py download_checkpoints
    python3 remote/call_wan22.py image2video --input-path frame.png --prompt "..." --save-path out.mp4
"""
import modal

GPU_TYPE = "A100-80GB"
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

app = modal.App("vitals-wan22")

wan22_cache = modal.Volume.from_name("vitals-wan22-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.4", "transformers>=4.45", "accelerate", "ftfy", "sentencepiece", "pillow", "numpy")
    .pip_install("git+https://github.com/huggingface/diffusers")   # Wan2.2 I2V-A14B needs main
    .env({"HF_HOME": "/cache/hf"})
)


def _aspect_size(image, pipe, max_area=480 * 832):
    """Same aspect-preserving, mod-aligned sizing `modal_app_wan.py` uses
    for 2.1 (duplicated: separate Modal app files can't import each
    other) -- this project's 4:3 renders come out ~544x720 there."""
    import numpy as np
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return height, width


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": wan22_cache}, timeout=3600)
def download_checkpoints():
    import torch
    from diffusers import WanImageToVideoPipeline
    WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    wan22_cache.commit()
    return "CHECKPOINTS_DOWNLOADED"


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": wan22_cache}, timeout=2400)
def image2video(image, prompt: str, seed: int = 0, num_frames: int = 81,
                guidance_scale: float = 3.5, num_inference_steps: int = 40):
    """Runs ONE Wan2.2 image-to-video generation, returns (T,H,W,3) uint8."""
    import numpy as np
    import torch
    from diffusers import WanImageToVideoPipeline
    from PIL import Image

    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    pil_image = Image.fromarray(np.asarray(image))
    height, width = _aspect_size(pil_image, pipe)
    pil_image = pil_image.resize((width, height))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(image=pil_image, prompt=prompt, height=height, width=width, num_frames=num_frames,
               guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
               generator=generator, output_type="np").frames[0]
    wan22_cache.commit()

    frames = np.asarray(out)
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    return frames
