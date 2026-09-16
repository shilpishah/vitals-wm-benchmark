"""generate_fn for CogVideoX1.5-5B-I2V (Zhipu/THUDM), image-to-video
(2026-09) -- plugs into `VideoWorldModel`'s own `generate_fn` slot.

**Restricted license** -- registered `restricted=True` (`cogvideox15`),
opt in per run with `VITALS_ENABLE_RESTRICTED_MODELS=cogvideox15`. See
`remote/modal_app_cogvideox.py`'s own docstring for the exact clauses
(commercial registration, 1M visits/month cap, PRC governing law).

Facts this adapter is built on (model card, confirmed):
- Single-image conditioning (`image=`) -- only the prefix's LAST frame
  reaches the model, same limitation as Wan/Hunyuan/Cosmos 3.
- Native 16 fps; valid frame counts satisfy (n-1) % 16 == 0 and the
  card says the model "works best with 81 and 161 frames". This adapter
  requests the smallest valid count >= what n_frames needs, but never
  below 81 (the card's own quality floor) and never above 161.
- The SAME frozen `scenario_prompts.SCENARIO_PROMPTS` text.
"""
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target

COGVIDEOX_NATIVE_FPS = 16
COGVIDEOX_MIN_FRAMES, COGVIDEOX_MAX_FRAMES = 81, 161


def requested_native_frames(n_frames, target_fps=30):
    """Smallest (16k+1) >= native frames needed, clamped to [81, 161]."""
    need = int(np.ceil(n_frames * COGVIDEOX_NATIVE_FPS / target_fps)) + 1
    valid = ((max(need, 1) - 1 + 15) // 16) * 16 + 1
    return max(COGVIDEOX_MIN_FRAMES, min(COGVIDEOX_MAX_FRAMES, valid))


def make_cogvideox_generate_fn_modal(scenario_name, app_name="vitals-cogvideox", function_name="image2video",
                                     target_fps=30, target_hw=(240, 320), seed=0,
                                     num_inference_steps=50, guidance_scale=6.0):
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
                                  num_inference_steps=num_inference_steps, guidance_scale=guidance_scale))
        return resample_to_target(raw, COGVIDEOX_NATIVE_FPS, target_fps, n_frames, target_hw)

    return generate_fn
