"""generate_fn for LTX-2.3 (Lightricks), image-to-video (2026-09).
Registered `ltx23`, PROVISIONALLY `restricted=True` until the LTX-2
Community License text has been read (see `remote/modal_app_ltx.py`).
Single-image conditioning (last prefix frame), native 24 fps, frame
counts on the VAE's 8k+1 grid: this adapter requests the smallest valid
count covering n_frames, floored at 25 and capped at the docs' 121.
"""
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target

LTX_NATIVE_FPS = 24
LTX_MIN_FRAMES, LTX_MAX_FRAMES = 25, 121


def requested_native_frames(n_frames, target_fps=30):
    """Smallest (8k+1) >= native frames needed, clamped to [25, 121]."""
    need = int(np.ceil(n_frames * LTX_NATIVE_FPS / target_fps)) + 1
    valid = ((max(need, 1) - 1 + 7) // 8) * 8 + 1
    return max(LTX_MIN_FRAMES, min(LTX_MAX_FRAMES, valid))


def make_ltx_generate_fn_modal(scenario_name, app_name="vitals-ltx", function_name="image2video",
                               target_fps=30, target_hw=(240, 320), seed=0,
                               guidance_scale=3.0, num_inference_steps=30):
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
                        f"(pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]

    def generate_fn(prefix_frames, n_frames):
        import modal
        last_frame = np.asarray(prefix_frames[-1])
        f = modal.Function.from_name(app_name, function_name)
        raw = np.asarray(f.remote(image=last_frame, prompt=prompt, seed=seed,
                                  num_frames=requested_native_frames(n_frames, target_fps),
                                  guidance_scale=guidance_scale, num_inference_steps=num_inference_steps))
        return resample_to_target(raw, LTX_NATIVE_FPS, target_fps, n_frames, target_hw)

    return generate_fn
