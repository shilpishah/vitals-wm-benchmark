"""Modal app for NVIDIA Cosmos 3 (June 2026), **Nano** (16B) checkpoint,
image-to-video -- backs `vitals/adapters/cosmos3.py`. Its own separate
app from `vitals-cosmos` (Cosmos-Predict2): different architecture (a
Mixture-of-Transformers omni-model, not the Predict2 diffusion stack),
different dependency set, different cache namespace.

Every fact below was confirmed directly against the diffusers Cosmos 3
documentation and the `nvidia/Cosmos3-Super-Image2Video` model card
(2026-09), not assumed -- the Hunyuan integration's own wrong-repo-name
incident is why this discipline is non-negotiable here:

- **Nano, not Super.** Super's ~120GB of bf16 weights do not fit any
  single GPU ("needs TP" -- tensor parallelism across >=2 GPUs, docs);
  Nano "fits on a single GPU" (16B -> ~32GB bf16). Super is a future
  multi-GPU integration, not a flag here.
- Pipeline: `Cosmos3OmniPipeline` -- the SAME checkpoint serves t2v/
  t2i/i2v; i2v is selected by passing `image=`. Load signature uses
  `dtype=` (not `torch_dtype=`) and `device_map="cuda"`.
- Native output fps is **24, confirmed** (the pipeline takes `fps=24.0`
  and every example exports at 24) -- unlike Hunyuan's own documented
  guess.
- `num_frames` 5..400 (default 189 = ~7.9s @ 24fps). Resolution tiers
  256p/480p/720p at 16:9, 4:3, 1:1, 3:4, 9:16 (model card); this
  project's 320x240 (4:3) renders map to 480p 4:3 = 640x480.
- Not gated; license OpenMDW-1.1 ("ready for commercial and non-
  commercial use", model card) -> registered `restricted=False`.
- **Safety checker stays ON.** The docs state it is mandatory under the
  NVIDIA Open Model License Agreement (the card says OpenMDW-1.1 -- a
  discrepancy noted, not resolved here; keeping the checker on is
  correct under either). `cosmos_guardrail` is installed for it.
- Diffusers: installed from git main -- `Cosmos3OmniPipeline` is
  documented under /main/ and the distilled modular variant is
  explicitly "until a release contains it". An unpinned main means an
  image rebuild can shift behavior; noted, accepted for a first
  integration.

**Prompt format -- a protocol decision, stated plainly.** Cosmos 3 was
trained on long JSON-structured captions and NVIDIA's recommended path
LLM-upsamples a short prompt via an Anthropic API call. A language model
in the measurement path is permanently out of scope for this project
(AGENT.md §11) and would also break the "identical frozen prompt across
models" rule. So `image2video` wraps the SAME frozen `scenario_prompts`
text in the model-native container deterministically -- `json.dumps(
{"scene": prompt})`, the exact form the docs' own modular example uses --
with no LLM anywhere. Cosmos 3 may underperform its own upsampled-prompt
setting as a result; that is a real, documented handicap, not a hidden
one. No negative prompt is passed (the docs' recommended one ships as an
asset file, not text; omitting it is the honest default over inventing
one).

**GPU tier: H100 (80GB).** A reasoned starting choice, not a measured
one -- 32GB weights + activations for ~49 frames at 480p should fit with
margin; the first real call is a debugging pass, same as every other
model here. Latency is unmeasured; the client timeout in
`run_model_population.py` is a placeholder until then.

**The safety checker's own weights are GATED (found directly, second
download attempt, after the libGL fix got the import working):**
`nvidia/Cosmos-1.0-Guardrail` returns `GatedRepoError: 401` -- the Nano
model is not gated, but the license-mandated checker pulls a repo that
is. Two prerequisites, same shape as Cosmos-Predict2's own in
`modal_app_cosmos.py`: (1) the HF account behind the token must accept
the gate at huggingface.co/nvidia/Cosmos-1.0-Guardrail (a one-time human
action on HF's site, not scriptable); (2) that token must reach the
container -- both functions attach the existing `vitals-cosmos` Modal
secret (`HF_TOKEN`, the same account Cosmos-Predict2 uses; reused rather
than a new per-model secret because its VALUE isn't available here to
duplicate). Disabling the checker to route around the gate was
deliberately NOT done: the docs call it mandatory under the license.

Usage:
    modal deploy remote/modal_app_cosmos3.py
    python3 remote/call_cosmos3.py download_checkpoints
    python3 remote/call_cosmos3.py image2video --input-path frame.png --prompt "..." --save-path out.mp4
"""
import modal

GPU_TYPE = "H100"
MODEL_ID = "nvidia/Cosmos3-Nano"

app = modal.App("vitals-cosmos3")

cosmos3_cache = modal.Volume.from_name("vitals-cosmos3-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # libgl1/libglib2.0-0: FOUND DIRECTLY on the first download_checkpoints
    # call -- importing `pipeline_cosmos3_omni` failed with "libGL.so.1:
    # cannot open shared object file" (the mandatory `cosmos_guardrail`
    # pulls in OpenCV, which links against system GL). debian_slim ships
    # neither; the standard headless-OpenCV fix.
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "accelerate",
        "sentencepiece",
        "ftfy",
        "pillow",
        "numpy",
        "cosmos_guardrail",   # mandatory safety checker -- see module docstring
    )
    .pip_install("git+https://github.com/huggingface/diffusers")   # Cosmos3OmniPipeline lives on main
    .env({"HF_HOME": "/cache/hf"})
)


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": cosmos3_cache}, timeout=3600,
              secrets=[modal.Secret.from_name("vitals-cosmos")])   # HF_TOKEN for the gated guardrail repo
def download_checkpoints():
    """One-time setup: pulls the Nano checkpoint (and the guardrail models
    the default-on safety checker needs) into the persistent volume, so
    the first real `image2video` call isn't a ~32GB cold download racing
    its own timeout. CPU-side load only (no device_map) -- this is a
    download, not an inference."""
    import torch
    from diffusers import Cosmos3OmniPipeline

    Cosmos3OmniPipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    cosmos3_cache.commit()
    return "CHECKPOINTS_DOWNLOADED"


def _stage_guardrail_nltk_data(hf_home="/cache/hf", dst="/root/nltk_data"):
    """FOUND DIRECTLY on the first two real image2video calls (2026-09-11):
    the guardrail's text safety check calls `nltk.word_tokenize`, and the
    installed nltk's hardened opener (`nltk.pathsec`, a CWE-59 TOCTOU
    guard) refuses to open its punkt tokenizer data:

        PermissionError: Security Violation [pathsec.open]: refusing to
        follow a symlink at open time for '/__modal/volumes/vo-.../hf/hub/
        models--nvidia--Cosmos-1.0-Guardrail/snapshots/<sha>/blocklist/
        nltk_data/tokenizers/punkt_tab/english/collocations.tab'

    The first fix attempt dereferenced per-file symlinks on the volume and
    found ZERO -- the files there are real. The path in the error is the
    tell: HF_HOME is /cache/hf, but the error names /__modal/volumes/...,
    i.e. the Modal MOUNT POINT itself is the symlink, and the opener
    refuses any path that resolves through one. Nothing on the volume can
    fix that. Repair, not bypass: copy the guardrail snapshot's whole
    `nltk_data` tree (punkt_tab, wordnet, ...) to a real local directory
    in the container and export NLTK_DATA to it. cosmos_guardrail 0.3.1
    only ever APPENDS its own paths to nltk.data.path, and NLTK_DATA
    entries are searched first, so the checker runs exactly as shipped on
    identical data. Must run BEFORE anything imports nltk (nltk reads
    NLTK_DATA at import); stdlib only for that reason. Cheap (a few MB)
    and repeated per container start."""
    import glob
    import os
    import shutil
    srcs = glob.glob(f"{hf_home}/hub/models--nvidia--Cosmos-1.0-Guardrail/snapshots/*/blocklist/nltk_data")
    if not srcs:
        raise RuntimeError("guardrail nltk_data not found in the HF cache -- run download_checkpoints first")
    shutil.copytree(srcs[0], dst, symlinks=False, dirs_exist_ok=True)
    os.environ["NLTK_DATA"] = dst
    return dst


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": cosmos3_cache}, timeout=1500,
              secrets=[modal.Secret.from_name("vitals-cosmos")])
def image2video(image, prompt: str, seed: int = 0, num_frames: int = 49,
                height: int = 480, width: int = 640,
                num_inference_steps: int = 35, guidance_scale: float = 6.0):
    """Runs ONE Cosmos 3 Nano image-to-video generation, returns frames as
    a raw (T,H,W,3) uint8 numpy array. `image`: (H,W,3) uint8, converted
    to PIL here so the caller never needs torch/diffusers. Defaults
    (35 steps, guidance 6.0) are the docs' own t2v example values; 480p
    4:3 matches this project's render aspect. See module docstring for
    the prompt-format decision."""
    print("guardrail nltk_data staged at", _stage_guardrail_nltk_data())   # BEFORE any nltk import (see helper)
    import json
    import numpy as np
    import torch
    from diffusers import Cosmos3OmniPipeline
    from PIL import Image

    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")

    pil_image = Image.fromarray(np.asarray(image))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=json.dumps({"scene": prompt}),   # frozen text, model-native container, no LLM
        image=pil_image,
        num_frames=num_frames, height=height, width=width, fps=24.0,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        generator=generator, output_type="np",
    )
    cosmos3_cache.commit()

    frames = np.asarray(result.video)
    if frames.ndim == 5:              # some pipelines return [B,T,H,W,C]
        frames = frames[0]
    if frames.dtype != np.uint8:      # float [0,1] -> uint8, same convention as wan/hunyuan
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    return frames
