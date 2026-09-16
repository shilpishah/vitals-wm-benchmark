"""generate_fn for Wan2.2 I2V-A14B (2026-09) -- the open successor to the
Wan2.1 adapter (`wan.py`), integrated as its OWN backend name (`wan22`)
so results never overwrite the 2.1 populations. Same single-image
conditioning (last prefix frame), same native 16 fps, same shared frozen
prompt table and resampler. Model-specific defaults come from the 2.2
model card (guidance 3.5, 40 steps, 81 frames) -- each model runs at its
own documented settings, exactly as hunyuan/cosmos already do. See
`remote/modal_app_wan22.py` for sources and the "Wan 2.7 doesn't exist as
open weights" note.
"""
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target

WAN22_NATIVE_FPS = 16   # card exports at 16, same as 2.1


def make_wan22_generate_fn_modal(scenario_name, app_name="vitals-wan22", function_name="image2video",
                                 target_fps=30, target_hw=(240, 320), seed=0,
                                 guidance_scale=3.5, num_inference_steps=40, num_frames=81):
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
                        f"(pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]

    def generate_fn(prefix_frames, n_frames):
        import modal
        last_frame = np.asarray(prefix_frames[-1])
        f = modal.Function.from_name(app_name, function_name)
        raw = np.asarray(f.remote(image=last_frame, prompt=prompt, seed=seed, num_frames=num_frames,
                                  guidance_scale=guidance_scale, num_inference_steps=num_inference_steps))
        return resample_to_target(raw, WAN22_NATIVE_FPS, target_fps, n_frames, target_hw)

    return generate_fn
