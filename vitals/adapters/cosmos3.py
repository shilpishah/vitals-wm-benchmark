"""generate_fn for NVIDIA Cosmos 3 Nano, image-to-video (2026-09) -- plugs
into `VideoWorldModel`'s own `generate_fn` slot. The successor to
Cosmos-Predict2 (`cosmos.py`), integrated as its OWN backend name
(`cosmos3nano`) rather than a flag on `cosmos`, so its results never
overwrite the Predict2 populations under the same file name -- the same
separate-name rule `cosmos14b`/`cosmos720p` already follow.

Facts this adapter is built on (confirmed against the diffusers Cosmos 3
docs, see `remote/modal_app_cosmos3.py`'s own docstring for sources):
- **Single-image conditioning** (`image=`, frame 0 anchored to it) --
  only the prefix's LAST frame reaches the model, same structural
  limitation as Wan/Hunyuan. Cosmos 3 also has a video-to-video mode
  that conditions on leading frames; deliberately NOT used here so
  Cosmos 3 is compared within the SAME single-image setting as every
  other model (input-normalization protocol: matched settings only).
  A multi-frame-conditioned variant would be a separate backend name.
- Native fps 24, confirmed (not a guess). `num_frames` 5..400; this
  adapter requests just enough native frames to cover n_frames at
  target_fps (+1), capped at the docs' default 189 -- asking for 7.9s
  of video to keep 2s wastes real H100 time.
- The SAME frozen `scenario_prompts.SCENARIO_PROMPTS` text, wrapped in
  the model-native JSON container inside the Modal app -- no LLM prompt
  upsampling, see the app's docstring for why.
"""
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target

COSMOS3_NATIVE_FPS = 24        # confirmed (pipeline `fps=24.0`, docs export at 24)
COSMOS3_MAX_FRAMES = 189       # docs' default; the hard ceiling is 400


def make_cosmos3_generate_fn_modal(scenario_name, app_name="vitals-cosmos3", function_name="image2video",
                                   target_fps=30, target_hw=(240, 320), seed=0,
                                   num_inference_steps=35, guidance_scale=6.0):
    """Builds `generate_fn(prefix_frames, n_frames) -> continuation_frames`
    calling the deployed `vitals-cosmos3` app. Needs `modal deploy remote/
    modal_app_cosmos3.py` + `download_checkpoints` first."""
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
                        f"(pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]

    def generate_fn(prefix_frames, n_frames):
        import modal

        last_frame = np.asarray(prefix_frames[-1])   # single-image conditioning -- see module docstring
        requested = min(COSMOS3_MAX_FRAMES,
                        max(5, int(np.ceil(n_frames * COSMOS3_NATIVE_FPS / target_fps)) + 1))

        f = modal.Function.from_name(app_name, function_name)
        raw = np.asarray(f.remote(image=last_frame, prompt=prompt, seed=seed, num_frames=requested,
                                  num_inference_steps=num_inference_steps, guidance_scale=guidance_scale))
        return resample_to_target(raw, COSMOS3_NATIVE_FPS, target_fps, n_frames, target_hw)

    return generate_fn
