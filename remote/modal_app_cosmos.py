"""Modal app for NVIDIA Cosmos-Predict2's Video2World inference (AGENT.md
M6) -- backs `vitals/adapters/cosmos.py`'s own `generate_fn` once it's
pointed at this instead of a local subprocess (this machine has no GPU;
Cosmos needs one).

A SEPARATE Modal app from `remote/modal_app.py` (`vitals-phi`,
SAM2+DINOv2), deliberately not merged into it: Cosmos needs its own CUDA
12.6 / torch stack, almost certainly incompatible with `modal_app.py`'s
own pinned `torch==2.6.0`/`torchvision==0.21.0` (chosen there specifically
for SAM2's compatibility, not Cosmos's). Keeping model-specific GPU
infrastructure isolated per model -- not shared, not tangled -- matches
the "one small file per model" discipline `vitals/adapters/` already
follows (2026-08: "the adapter and this evaluation suite should be
model-agnostic").

STATUS: real code, written directly against Cosmos's own published repo
and Hugging Face model card -- NOT YET DEPLOYED OR RUN. Two things only
the account owner can do stand between this and a real first call,
confirmed directly, not assumed:

1. **The model is GATED.** Confirmed on its own Hugging Face page
   (huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World): "agree to
   share your contact information to access this model." Someone with a
   Hugging Face account has to visit that page and accept before ANY
   download attempt succeeds, regardless of token validity otherwise --
   a one-time human action on HF's own site, not something scriptable
   from here.
2. **That account's HF access token needs to exist as a Modal Secret**:
        modal secret create vitals-cosmos HF_TOKEN=hf_...
   `download_checkpoints` below reads it from the environment
   (`os.environ["HF_TOKEN"]`) -- Cosmos's own `download_checkpoints.py`
   authenticates through `huggingface_hub`'s standard token resolution,
   which checks exactly this.

GPU tier: **H100**, not modal_app.py's own A10G (24GB, ruled out
immediately -- Cosmos's own model card: 32.54GB VRAM for THIS model
alone) and not A100-80GB either, on a real cost calculation, not a
default: Modal's own published per-second pricing (checked directly,
2026-08) is A100-80GB $0.000694/s (~$2.50/hr) vs. H100 $0.001097/s
(~$3.95/hr) -- H100 costs 58% more per second. But Cosmos's own published
benchmark shows H100 SXM finishing a 720p/16fps generation in ~229s;
Cosmos publishes no A100 number, but A100 (Ampere, 2020) typically runs
2-3x slower than H100 (Hopper, 2022) on transformer/diffusion-heavy
workloads like this one. At even a conservative 2x slowdown, A100 costs
MORE per call than H100 (~$0.32 vs. ~$0.25, back-of-envelope) despite the
lower hourly rate, while also being slower wall-clock -- worse on both
axes actually being compared, not a tradeoff. H100 also has at least as
much VRAM (80GB+) as A100-80GB, so this costs no safety margin either.
The A100 choice in an earlier version of this file was reasoned from VRAM
headroom alone, without ever actually running this comparison -- corrected
directly (2026-08: "wait why aren't we using a h100 gpu for this?").

Usage, once the two blockers above are cleared:
    modal deploy remote/modal_app_cosmos.py
    python3 remote/call_cosmos.py download_checkpoints
    python3 remote/call_cosmos.py video2world --input-path prefix.mp4 --prompt "..." --save-path out.mp4

For the 14B Video2World variant (2026-08, added to test whether the 2B
result reflects a real physics-consistency gap or just the deliberately-
smaller checkpoint chosen for the first pipeline-validation run): accept
14B's OWN gated access terms at huggingface.co/nvidia/Cosmos-Predict2-
14B-Video2World (separate from 2B's own acceptance), then
    python3 remote/call_cosmos.py download_checkpoints --model-sizes 14B
    python3 remote/call_cosmos.py video2world --model-size 14B ...
No new GPU tier needed -- 14B's own documented 56.38GB requirement fits
inside this app's existing H100 (80GB) with room to spare.
"""
import pathlib
import modal

GPU_TYPE = "H100"
COSMOS_REPO_DIR = "/root/cosmos-predict2"

app = modal.App("vitals-cosmos")

# SEPARATE from vitals-phi's own model_cache volume -- different model,
# different image, no reason to share a cache namespace with SAM2/DINOv2
# weights; keeps a Cosmos-specific cache cleanly inspectable/clearable on
# its own.
cosmos_cache = modal.Volume.from_name("vitals-cosmos-cache", create_if_missing=True)

# NOT modal.Image.debian_slim -- the FIRST real deploy attempt (2026-08)
# failed at import time: transformer_engine (pulled in transitively via
# Cosmos's own text encoder -> megatron.core) needs libnvrtc and other
# CUDA TOOLKIT shared libraries physically present on the system, which
# debian_slim never has -- torch's own pip-bundled CUDA runtime is not the
# same thing and doesn't satisfy this. Fixed per Modal's own documented
# pattern for exactly this case (modal.com/docs/guide/cuda): build from an
# actual NVIDIA CUDA "devel" base image, which ships the full toolkit as
# system packages, not debian_slim + pip-only torch.
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu24.04", add_python="3.10")   # Cosmos's own setup.md: Python 3.10
    .entrypoint([])   # Modal's own recommendation -- suppresses the CUDA base image's verbose startup logging
    .apt_install("git", "ffmpeg")
    .run_commands(f"git clone https://github.com/nvidia-cosmos/cosmos-predict2.git {COSMOS_REPO_DIR}")
    # Cosmos's own documented dependency install is `uv sync --extra cu126`
    # (setup.md).
    .pip_install("uv")
    .run_commands(f"cd {COSMOS_REPO_DIR} && uv pip install --system -e '.[cu126]'")
    .pip_install("huggingface_hub", "imageio[ffmpeg]")
    .env({"HF_HOME": "/cache/hf"})
)


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": cosmos_cache}, timeout=3600,
              secrets=[modal.Secret.from_name("vitals-cosmos")])
def download_checkpoints(model_sizes=("2B",)):
    """One-time setup -- downloads the Video2World checkpoint(s) named in
    `model_sizes`, at BOTH resolutions cosmos.py's own code can request
    (`--resolution 480 720` -- 480 is `COSMOS_RESOLUTION`'s own default
    there, chosen for cost/turnaround; 720 is examples/video2world.py's
    own script default, in case a caller overrides resolution upward) --
    not Cosmos's full ~250GB checkpoint set across every model size/
    variant/task (confirmed from download_checkpoints.py's own CLI:
    `--model_sizes`/`--model_types` exist specifically to select one
    variant). Requires the HF_TOKEN secret above AND that account having
    already accepted EACH requested size's OWN gated access terms on
    Hugging Face (2B and 14B are gated SEPARATELY -- confirmed directly
    against each model's own HF page, accepting one does NOT cover the
    other) -- fails loudly, not silently, if either is missing.

    `model_sizes` defaults to `("2B",)` (this app's own original,
    already-downloaded checkpoint) so a plain re-run of this function
    never re-downloads or breaks anything already working -- call with
    `model_sizes=("14B",)` explicitly to add the larger variant (2026-08,
    added specifically to test whether the 2B result reflects a real
    physics-consistency gap or just the deliberately-smaller checkpoint
    chosen for the first pipeline-validation run).

    `--checkpoint_dir` IS a real flag on THIS script (download_checkpoints.
    py's own argparse) -- do not confuse with `video2world()` below, where
    the SAME-sounding flag does not exist on examples/video2world.py at
    all (see that function's own comment for the real mechanism there,
    COSMOS_PREDICT2_ARGS) -- confirmed by reading both scripts' actual
    source directly, not assumed from the flag name alone."""
    import os
    import subprocess

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not set -- create the Modal secret first, see module docstring.")

    result = subprocess.run(
        ["python", "scripts/download_checkpoints.py",
         "--model_sizes", *model_sizes, "--model_types", "video2world",
         "--resolution", "480", "720",
         "--checkpoint_dir", "/cache/checkpoints"],
        cwd=COSMOS_REPO_DIR, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        raise RuntimeError(f"checkpoint download failed for model_sizes={model_sizes} "
                            f"(exit {result.returncode}):\n{result.stderr[-4000:]}\n\n"
                            f"Most likely cause if this is the first attempt for a given size: the HF "
                            f"account behind HF_TOKEN has not yet accepted THAT size's own gated access "
                            f"terms -- visit huggingface.co/nvidia/Cosmos-Predict2-{model_sizes[0]}-Video2World "
                            f"while logged in and accept, THEN retry (each size needs its own acceptance).")
    cosmos_cache.commit()
    return f"CHECKPOINTS_DOWNLOADED: {model_sizes}"


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": cosmos_cache}, timeout=1200,
              secrets=[modal.Secret.from_name("vitals-cosmos")])
def video2world(prefix_video_bytes: bytes, prompt: str, num_conditional_frames: int = 5,
                fps: int = 16, resolution: str = "480", seed: int = 0, model_size: str = "2B") -> bytes:
    """Runs ONE Video2World generation call remotely, returns the output
    video as raw mp4 bytes -- the actual remote counterpart to
    `vitals/adapters/cosmos.py`'s subprocess call, now correctly running
    on a machine that has both Cosmos AND a GPU, neither of which this
    project's own local dev machine has. Bytes in, bytes out (not file
    paths) so the caller (running locally, no shared filesystem with this
    container) doesn't need Modal's own file-mount machinery for a single
    short clip.

    `--num_conditional_frames 5`: Cosmos's own documented max for
    multi-frame conditioning (AGENT.md M6's own cosmos.py docstring) --
    matches that file's own convention exactly, so results are comparable
    regardless of whether generate_fn calls this function or a local
    Cosmos install directly.

    model_size: "2B" (default) or "14B" -- see download_checkpoints's own
    docstring for why this needs its OWN checkpoint download and gated-
    access acceptance first; a call with a size that was never downloaded
    fails with whatever error `examples/video2world.py` itself raises for
    a missing checkpoint, not a special-cased message here (same "let the
    real error surface" discipline as everywhere else in this file). No
    CPU-offload flags added for 14B -- its own documented 56.38GB
    requirement (64GB recommended) sits comfortably inside this app's
    already-provisioned H100 (80GB), same GPU_TYPE as 2B."""
    import pathlib
    import subprocess
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        in_path = tmp / "prefix.mp4"
        out_path = tmp / "continuation.mp4"
        in_path.write_bytes(prefix_video_bytes)

        cmd = [
            "python", "-m", "examples.video2world",
            "--model_size", model_size,
            "--input_path", str(in_path),
            "--num_conditional_frames", str(num_conditional_frames),
            "--prompt", prompt,
            "--save_path", str(out_path),
            "--fps", str(fps),
            "--resolution", resolution,
            "--seed", str(seed),
            # --disable_guardrail: the SECOND real deploy attempt (2026-08)
            # failed here -- Cosmos-Guardrail1's own downloaded checkpoint is
            # missing its "whitelist" blocklist subdirectory (confirmed via
            # `modal volume ls`: blocklist/{exact_match,custom,nltk_data}
            # exist, blocklist/whitelist does not -- a gap in NVIDIA's own
            # checkpoint packaging, not something this project controls).
            # Disabling is the right call regardless, not just a bug
            # workaround: this is a content-safety filter for public
            # generation, irrelevant to a synthetic physics benchmark
            # generating a red ball and gray walls.
            "--disable_guardrail",
            # --disable_prompt_refiner: NOT a bug workaround -- a real
            # protocol decision. The refiner rewrites the input prompt via
            # an LLM before generation; this project's own SCENARIO_PROMPTS
            # are frozen, pre-registered text (input-normalization protocol,
            # AGENT.md M6/3.12) -- letting another model silently rewrite
            # them per call would undermine that discipline, not just add
            # an extra failure surface.
            "--disable_prompt_refiner",
        ]
        # NOT a --checkpoint_dir CLI flag -- examples/video2world.py has no
        # such flag (confirmed by reading imaginaire/constants.py directly,
        # not assumed from the download script's own, unrelated flag of the
        # same-sounding name). CHECKPOINTS_DIR is read from the
        # COSMOS_PREDICT2_ARGS environment variable instead (shlex-split,
        # parsed as this package's own --checkpoints flag), defaulting to
        # the relative path "checkpoints" if unset -- which would resolve
        # to an EMPTY dir inside this container, not the persistent volume
        # download_checkpoints wrote to.
        env = dict(os.environ, COSMOS_PREDICT2_ARGS="--checkpoints /cache/checkpoints")
        result = subprocess.run(cmd, cwd=COSMOS_REPO_DIR, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Cosmos video2world failed (exit {result.returncode}):\n{result.stderr[-4000:]}")

        return out_path.read_bytes()
