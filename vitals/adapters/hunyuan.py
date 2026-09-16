"""generate_fn for HunyuanVideo-1.5 (Tencent), Image2Video-480p (2026-09) --
plugs into `VideoWorldModel`'s own `generate_fn` slot. The THIRD real
model integrated into this project, after Cosmos-Predict2 and Wan2.1 --
chosen specifically for architectural diversity (a different lab,
different text-encoder stack, different training pipeline from both) so
the "real models lose to baselines" finding this project keeps producing
isn't just a Cosmos/Wan-specific artifact.

https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v

**License restriction, stated plainly, not buried:** `tencent-hunyuan-
community`. Confirmed directly against the actual license text -- it does
NOT apply in the European Union, United Kingdom, or South Korea (a
blanket territorial exclusion), and separately gates deployment above
100M MAU behind a separate Tencent license (almost certainly irrelevant
to benchmark use, not a deployed product). This is why `hunyuan` is
registered `restricted=True` in `vitals/adapters/__init__.py::
MODEL_REGISTRY` and requires `VITALS_ENABLE_RESTRICTED_MODELS=hunyuan` to
activate -- see ADDING_A_MODEL.md. Do not remove that gate to make this
"just work" by default; that is the one thing it must never do.

Chosen over the original 13B HunyuanVideo-I2V for VRAM (confirmed
directly against both model cards): 60-80GB there vs. ~14GB (with
offloading) here -- a genuinely different, cheaper-to-run tier than every
other real model this project has integrated so far, if the real number
holds up under an actual run (see `remote/modal_app_hunyuan.py`'s own
docstring for the honest caveat on why 14GB-stated is not yet 14GB-
confirmed: co-resident with the 8.3B transformer is a 7B-parameter
Qwen2.5-VL text encoder, not a small CLIP encoder like Wan's).

Facts this design is built on, confirmed directly against the pipeline's
own diffusers documentation, not assumed:
- **Single-IMAGE conditioning, not multi-frame video conditioning** like
  Cosmos's own 5-frame option -- same structural limitation as Wan.
  `HunyuanVideo15ImageToVideoPipeline.__call__`'s own signature takes one
  `image`, nothing else from the prefix. Only the prefix's LAST frame
  actually reaches this model.
- Requires a text prompt, same as Cosmos/Wan -- uses the SAME frozen
  `scenario_prompts.SCENARIO_PROMPTS`, not a Hunyuan-specific rewrite.
- **No `guidance_scale` call-time argument** -- confirmed directly from
  the `__call__` signature, unlike Wan/Cosmos. Guidance is controlled via
  a `pipe.guider` object instead (a newer diffusers "Guider" pattern);
  `remote/modal_app_hunyuan.py` leaves it at the pipeline's own default
  rather than reach into it.
- **Native output frame rate is NOT explicitly documented for the I2V
  pipeline** -- the closest confirmed signal is the pipeline's own
  official usage example, which exports at `fps=24`. Treated as the best
  available estimate, not a confirmed spec value, and flagged here rather
  than silently assumed -- if a real run's motion looks wrong-speed
  relative to other models, this is the first thing to re-check against
  an actual timed generation, not `resample_to_target`'s own logic.
- `num_frames` default 121 (vs. Wan's 81 @ 16fps, Cosmos's own convention)
  -- passed through as-is unless the caller needs fewer; `resample_to_
  target`'s own trim/pad logic (its own docstring: "the HONEST stand-in
  for 'how many frames does one call return'") absorbs any mismatch
  between what's requested and what actually comes back, the same as
  every other model here.
- Output comes back as a raw numpy array (`output_type="np"` default),
  never a video file -- same structural convenience as Wan, handled
  inside `remote/modal_app_hunyuan.py`'s own `image2video` function.
"""
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target

# Not a confirmed spec value -- see module docstring's own caveat.
HUNYUAN_NATIVE_FPS = 24


def make_hunyuan_generate_fn_modal(scenario_name, app_name="vitals-hunyuan", function_name="image2video",
                                   target_fps=30, target_hw=(240, 320), seed=0,
                                   num_inference_steps=50):
    """Builds a `generate_fn(prefix_frames, n_frames) -> continuation_frames`
    closure calling the deployed `vitals-hunyuan` Modal app (`remote/
    modal_app_hunyuan.py`'s own `image2video` function) -- the only real
    path today, matching how every other real GPU call in this project
    already works (never a local install; this dev machine has no GPU
    regardless). Needs `modal deploy remote/modal_app_hunyuan.py` first,
    same invocation discipline as every other Modal app here (deploy
    once, never `modal run` directly)."""
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
                        f"(pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]

    def generate_fn(prefix_frames, n_frames):
        import modal

        last_frame = np.asarray(prefix_frames[-1])   # single-image conditioning -- see module docstring

        # Request roughly enough native-fps frames to cover n_frames at
        # target_fps, capped by the pipeline's own documented default --
        # asking for far more than needed wastes real GPU time for no
        # benefit once resample_to_target trims the rest.
        requested_native_frames = min(121, max(1, int(np.ceil(n_frames * HUNYUAN_NATIVE_FPS / target_fps)) + 1))

        f = modal.Function.from_name(app_name, function_name)
        raw_frames = f.remote(image=last_frame, prompt=prompt, seed=seed,
                              num_frames=requested_native_frames,
                              num_inference_steps=num_inference_steps)
        raw_frames = np.asarray(raw_frames)

        return resample_to_target(raw_frames, HUNYUAN_NATIVE_FPS, target_fps, n_frames, target_hw)

    return generate_fn
