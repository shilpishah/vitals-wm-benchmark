"""generate_fn for NVIDIA Cosmos-Predict2's Video2World model (AGENT.md
M6) -- plugs into VideoWorldModel's own `generate_fn` slot
(vitals/adapters/video_model.py).

https://github.com/nvidia-cosmos/cosmos-predict2

STATUS: real, documented-against-the-actual-repo code, NOT yet executed
against a live install -- this machine has no GPU, and running this needs
(a) `cosmos-predict2` installed with its own CUDA 12.6 environment
(Ampere-or-newer GPU), (b) the Video2World-2B checkpoint downloaded via
Cosmos's own `scripts/download_checkpoints.py` (Hugging Face-hosted;
Cosmos's own docs report ~250GB for the FULL checkpoint set across every
model variant, not this one model alone -- exact 2B Video2World size not
found in Cosmos's own docs), and (c) real GPU time to run inference. None
of that has been provisioned -- do not read "written" as "verified."

Facts this design is built on, confirmed directly from Cosmos's own repo
docs (not assumed):
- Called via a documented CLI (`python -m examples.video2world`), not a
  clearly-documented direct Python class -- a `Video2WorldPipeline` class
  exists internally but its exact call signature isn't in Cosmos's own
  published docs, so this wraps the CLI (subprocess), the more stable
  documented surface, not an internal API that could change underneath.
- Takes ONE video file as `--input_path` plus a required text `--prompt`
  -- Video2World is not purely "frames in, frames out" the way this
  project's own `generate_fn` contract implies; the text prompt is a real,
  new per-scenario input this adapter has to supply (see
  SCENARIO_PROMPTS below), frozen once per scenario like everything else
  in the input-normalization protocol, never varied per episode.
- "Multi-frame conditioning" uses only the LAST 5 consecutive frames of
  whatever video file it's given -- our own 30-frame (1.0s) prefix
  (AGENT.md M6 protocol item 4) still works unmodified here: hand Cosmos
  the whole prefix clip, it uses the tail of it. No protocol change
  needed for this model specifically.
- Native output: 16fps (or 10fps), 480p or 720p -- NOT this project's own
  30fps/320x240 convention. Resampled both directions per protocol items
  2-3 (resize/resample going in, resize/resample coming back out, applied
  identically to every model, never tuned per-model).
- UNRESOLVED, stated honestly rather than guessed: Cosmos's own docs do
  not specify how many frames one `video2world` call actually returns, or
  whether reaching a full `horizon_s` needs one call or several chained
  autoregressive calls (feeding each output back in as the next call's
  conditioning). This wrapper requests one call and trims/pads the result
  to the requested frame count (see `_resample_to_target` below) as an
  honest stand-in until this is confirmed against a real run -- do not
  trust the padding behavior as a real generative capability, it exists
  only so `VideoWorldModel`'s own frame-count contract doesn't break.
"""
import pathlib
import subprocess
import tempfile
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import _write_video, _read_video, resample_to_target as _resample_to_target

COSMOS_NATIVE_FPS = 16
COSMOS_RESOLUTION = "480"   # "480" or "720" -- 480p chosen for cost/turnaround, not quality


def make_cosmos_generate_fn_local(scenario_name, model_size="2B", cosmos_repo_dir=None,
                                  target_fps=30, target_hw=(240, 320), seed=0):
    """Builds a `generate_fn(prefix_frames, n_frames) -> continuation_frames`
    closure that shells out to a LOCAL `cosmos-predict2` checkout with GPU
    access. Only useful on a machine that actually has both -- this
    project's own dev machine has neither, so `make_cosmos_generate_fn_modal`
    below (calling the deployed `vitals-cosmos` Modal app,
    `remote/modal_app_cosmos.py`) is the path that actually runs anywhere
    real work happens in this project, matching how every other GPU call
    here already works (SAM2/DINOv2 via `remote/modal_app.py`, never a
    local install). Kept for a hypothetical future machine with local
    Cosmos+GPU access -- not the recommended path today.

    Bound to one scenario's own frozen prompt (SCENARIO_PROMPTS) and this
    project's own frame-rate/resolution convention.

    model_size: "2B" (default) or "14B". 2B chosen as the default over
    14B for the FIRST real run -- Cosmos's own docs flag the 14B model as
    needing CPU-offload flags to fit in memory at all ("requires
    significant GPU memory"), and this project's own established
    philosophy (GATE 3, the baseline ladder) is start with what's cheapest
    to validate the PIPELINE with, not the biggest model available.

    cosmos_repo_dir: path to a cloned+installed cosmos-predict2 checkout
    (its own `examples/video2world.py` is invoked via `python -m` from
    there) -- REQUIRED, no default, since this repo does not vendor or
    install Cosmos itself.

    seed: passed straight through to Cosmos's own `--seed` -- fixed, not
    re-randomized per call, matching the input-normalization protocol's
    own sampling policy (item 6: a rerun of the identical episode must
    reach the identical verdict)."""
    if cosmos_repo_dir is None:
        raise ValueError("make_cosmos_generate_fn needs cosmos_repo_dir -- path to a real, "
                          "installed cosmos-predict2 checkout. This project does not vendor Cosmos.")
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen Cosmos prompt for scenario {scenario_name!r} -- add one to "
                        f"SCENARIO_PROMPTS above, do not invent one at call time (pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]
    cosmos_repo_dir = pathlib.Path(cosmos_repo_dir)

    def generate_fn(prefix_frames, n_frames):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            in_path = tmp / "prefix.mp4"
            out_path = tmp / "continuation.mp4"
            _write_video(prefix_frames, in_path, fps=target_fps)

            cmd = [
                "python", "-m", "examples.video2world",
                "--model_size", model_size,
                "--input_path", str(in_path),
                "--num_conditional_frames", "5",
                "--prompt", prompt,
                "--save_path", str(out_path),
                "--fps", str(COSMOS_NATIVE_FPS),
                "--resolution", COSMOS_RESOLUTION,
                "--seed", str(seed),
            ]
            result = subprocess.run(cmd, cwd=str(cosmos_repo_dir), capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Cosmos video2world failed (exit {result.returncode}):\n"
                                    f"{result.stderr[-4000:]}")

            raw_frames, raw_fps = _read_video(out_path)

        return _resample_to_target(raw_frames, raw_fps, target_fps, n_frames, target_hw)

    return generate_fn


def make_cosmos_generate_fn_modal(scenario_name, app_name="vitals-cosmos", function_name="video2world",
                                  target_fps=30, target_hw=(240, 320), seed=0, model_size="2B",
                                  resolution=COSMOS_RESOLUTION):
    """The RECOMMENDED `generate_fn` builder -- calls the deployed
    `vitals-cosmos` Modal app (`remote/modal_app_cosmos.py`'s own
    `video2world` function) instead of a local subprocess, matching how
    every other real GPU call in this project already works (SAM2/DINOv2
    via `remote/modal_app.py`/`remote/call.py`, never a local install).
    Needs `modal_app_cosmos.py` deployed (`modal deploy remote/
    modal_app_cosmos.py`) and its own two prerequisites cleared first --
    see that module's own docstring: the Hugging Face account behind the
    `vitals-cosmos` Modal secret must have already accepted the SPECIFIC
    model size's own gated access terms (2B and 14B are gated
    SEPARATELY -- accepting one does not cover the other, confirmed
    directly against each model's own Hugging Face page), and that size's
    checkpoint must already be downloaded (`download_checkpoints(model_
    sizes=(model_size,))`, a one-time setup call). Neither is checked
    here -- a failure surfaces as whatever error the remote function
    itself raises, same as any other Modal call in this project.

    model_size: "2B" (default, matches every existing published result)
    or "14B" -- Cosmos-Predict2's own larger Video2World variant (2026-08,
    requested directly, to test whether the 2B result reflects a genuine
    physics-consistency gap or just this project's own deliberate choice
    of the SMALLER checkpoint for the first pipeline-validation run, see
    this module's own docstring). Real VRAM figures, fetched directly
    from Cosmos-Predict2's own `documentations/performance.md`, not
    guessed: 2B needs 32.54GB, 14B needs 56.38GB (64GB recommended) --
    both comfortably inside a single H100's 80GB, so no CPU-offload flags
    are needed for 14B either, same GPU tier as 2B.

    resolution: `COSMOS_RESOLUTION` ("480") by default, matching every
    existing published result -- pass "720" to test the OTHER honest
    caveat named alongside the 2B/14B one (2026-08): `COSMOS_RESOLUTION`
    was chosen "for cost/turnaround, not quality" (this module's own
    comment where it's defined), never tested against. Real published
    timing, fetched directly, not guessed: 480p/16fps takes 79.87-87.32s
    on H100 (no NATTEN, matching this project's own call -- no `--natten`
    flag is ever passed); 720p/16fps takes 228.8-378.5s, ~3-4x slower,
    depending on the specific H100 variant benchmarked. Isolated as its
    own parameter, independent of model_size, so a caller can vary
    EXACTLY one axis at a time (this project's own established
    calibration discipline, AGENT.md's "vary one thing, hold the rest")
    rather than conflating "is it resolution or is it model size" in one
    combined experiment."""
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen Cosmos prompt for scenario {scenario_name!r} -- add one to "
                        f"SCENARIO_PROMPTS above, do not invent one at call time (pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]

    def generate_fn(prefix_frames, n_frames):
        import modal

        with tempfile.TemporaryDirectory() as tmp:
            in_path = pathlib.Path(tmp) / "prefix.mp4"
            _write_video(prefix_frames, in_path, fps=target_fps)
            prefix_bytes = in_path.read_bytes()

        f = modal.Function.from_name(app_name, function_name)
        out_bytes = f.remote(prefix_video_bytes=prefix_bytes, prompt=prompt,
                             num_conditional_frames=5, fps=COSMOS_NATIVE_FPS,
                             resolution=resolution, seed=seed, model_size=model_size)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "continuation.mp4"
            out_path.write_bytes(out_bytes)
            raw_frames, raw_fps = _read_video(out_path)

        return _resample_to_target(raw_frames, raw_fps, target_fps, n_frames, target_hw)

    return generate_fn
