"""generate_fn for Wan2.1 (Alibaba), Image2Video-14B-480P (AGENT.md M6) --
plugs into `VideoWorldModel`'s own `generate_fn` slot. The SECOND real
model integrated into this project -- the first, `cosmos.py`'s own
Cosmos-Predict2, already proved the architecture is genuinely
model-agnostic before this file was written (a structurally unrelated
`generate_fn`, `pixel_constant_velocity.py`, ran through the identical,
unmodified `VideoWorldModel` with zero changes elsewhere -- see AGENT.md's
own writeup). This file is that same proof point applied to an actual
second real model, not just a synthetic stand-in.

https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers

Chosen over other open-weight candidates specifically for (all confirmed
directly against the model's own docs/card before building this, not
assumed):
- **NOT gated** -- no HF license-acceptance step, no Modal secret needed.
  A real, concrete difference from Cosmos, which needed both.
- **Apache 2.0** -- more permissive than Cosmos's NVIDIA Open Model
  License.
- **~16GB VRAM** -- roughly half Cosmos's 32.54GB.
- **Native `diffusers` integration** -- a real, documented Python class
  (`WanImageToVideoPipeline`), not a CLI wrapped in `subprocess` the way
  Cosmos's own interface required.

Facts this design is built on, confirmed directly, not assumed:
- **Single-IMAGE conditioning, not multi-frame video conditioning** like
  Cosmos's own 5-frame option -- `WanImageToVideoPipeline.__call__`'s own
  signature takes one `image` (plus an optional `last_image` for a
  DIFFERENT generation mode, first-last-frame interpolation, not used
  here). This project's own 30-frame prefix is still rendered in full
  (protocol item 4 unchanged, and reused identically for Cosmos too), but
  only ITS LAST FRAME actually reaches Wan2.1 -- a real, structural
  difference in how much of the prefix each model actually gets to use,
  stated plainly here, not hidden.
- Requires a text prompt, same as Cosmos -- uses the SAME frozen
  `scenario_prompts.SCENARIO_PROMPTS`, not a Wan-specific rewrite, so the
  two models are compared under identical wording (input-normalization
  protocol: "compared only within matched settings").
- Native output: 480p, `num_frames=81` DEFAULT at 16fps (~5.06s) --
  resampled to this project's own 30fps/320x240 convention via the SAME
  `resample_to_target` Cosmos uses.
- Output comes back as a raw numpy array directly (`output_type="np"`,
  handled inside `remote/modal_app_wan.py`'s own `image2video` function),
  never a video file -- no `_write_video`/`_read_video` round trip needed
  for this model specifically, a real, structural difference from
  Cosmos's file-based interface.
"""
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target

WAN_NATIVE_FPS = 16


def make_wan_generate_fn_modal(scenario_name, app_name="vitals-wan", function_name="image2video",
                               target_fps=30, target_hw=(240, 320), seed=0,
                               guidance_scale=5.0, num_inference_steps=50):
    """Builds a `generate_fn(prefix_frames, n_frames) -> continuation_frames`
    closure calling the deployed `vitals-wan` Modal app (`remote/
    modal_app_wan.py`'s own `image2video` function) -- the only real path
    today, matching how every other real GPU call in this project already
    works (never a local install; this dev machine has no GPU regardless).
    Needs `modal_app_wan.py` deployed (`modal deploy remote/
    modal_app_wan.py`) -- unlike Cosmos, no gated-access or Modal-secret
    prerequisite to clear first (see module docstring)."""
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
                        f"(pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]

    def generate_fn(prefix_frames, n_frames):
        import modal

        last_frame = np.asarray(prefix_frames[-1])   # single-image conditioning -- see module docstring

        f = modal.Function.from_name(app_name, function_name)
        raw_frames = f.remote(image=last_frame, prompt=prompt, seed=seed,
                              guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
        raw_frames = np.asarray(raw_frames)

        return resample_to_target(raw_frames, WAN_NATIVE_FPS, target_fps, n_frames, target_hw)

    return generate_fn
