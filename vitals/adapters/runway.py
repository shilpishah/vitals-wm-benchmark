"""API-access video models via the Runway developer API (2026-09) -- the
first adapter here for a model whose weights are NOT open, answering a
direct requirement: "it should not be a hard limit that any model that's
open access via an API cannot be used."

Same adapter contract as every open-weights model (`generate_fn(prefix_
frames, n_frames) -> continuation_frames`, exactly n_frames of (H,W,3)
uint8), same conditioning discipline (input-normalization protocol item
1, pixels only: the API receives the LAST rendered prefix frame as a
base64 data URI plus the scenario's own frozen prompt from
`scenario_prompts.SCENARIO_PROMPTS` -- byte-identical to what wan.py/
hunyuan.py's own image2video calls receive, so an API model is compared
within matched settings, never a different protocol), same resample-in/
resample-out via `video_utils.resample_to_target`. What differs is only
WHERE inference runs: nothing is deployed to Modal, the call goes to
Runway's hosted endpoint, so there is no `remote/modal_app_*.py` for
this adapter at all.

Runway's API is a GATEWAY, not one model -- confirmed directly from the
SDK's own typed surface (runwayml 5.20.0, `image_to_video.create`), not
from marketing: the same endpoint accepts `gen4.5`, `gen4_turbo`,
`veo3.1`, `veo3.1_fast`, `seedance2`, `hailuo3`, `wan3`, and more, each
with its own allowed `ratio`/`duration` literals. So this ONE adapter is
parameterized by `model`; a new closed model reachable through Runway is
a registry entry + a dispatch line, not a new adapter (the same "two
one-line additions" rule `run_model_population.make_generate_fn`'s own
docstring already states for open models).

Honest differences from the open-weights adapters, stated plainly:

- **Cost per call, and retries re-bill.** Every generation is charged in
  credits (the terminal task reports `cost.credits`, logged below per
  call). `run_model_population.py`'s own `with_one_retry` re-submits on
  a transient failure -- for an API model that is a second charge, not
  a free retry of an already-paid GPU container.
- **Reproducibility is weaker.** `seed` is accepted and passed (fixed at
  0, mirroring wan.py -- episode-to-episode variation comes from the
  physics-perturbed prefix, never from the model seed), but a hosted
  endpoint gives no guarantee that the same seed yields identical
  pixels across days or model updates. `seed_test_retest` may not be
  exactly 0 here the way it is for a pinned local checkpoint.
- **The model can change under you.** A model id like `gen4.5` is a
  moving target the provider can update silently; an open checkpoint
  is a fixed artifact. Results should record the run date.
- **Output duration is coarse.** Runway takes an integer number of
  seconds (2..10 for gen4.5); this adapter requests the smallest
  duration that covers n_frames, then trims to exactly n_frames -- the
  same trim/pad discipline every other adapter already uses.
- **Aspect ratio is fixed to Runway's allowed set.** Renders here are
  320x240 (4:3); the closest allowed ratio is `1104:832` (1.327 vs
  1.333), then `resample_to_target` resizes back to 320x240. Same class
  of aspect handling wan.py already documents for its own 544x720.
- **Output fps is read from the returned mp4's own metadata**, not
  assumed -- the SDK's typed surface doesn't state fps, so this adapter
  refuses to hardcode it; `_read_video`'s 24 fps fallback applies only
  if the file carries no fps tag at all.

Setup:
    pip install -e ".[runway]"
    export RUNWAYML_API_SECRET=...          # the SDK's own env var (confirmed from runwayml/_client.py)
    # or, for the Modal orchestrator: modal secret create vitals-runway RUNWAYML_API_SECRET=...
    python3 scripts/run_model_population.py --model runway_gen4.5 --scenario collision --seeds 1,2,3
"""
from __future__ import annotations
import base64
import io
import math
import os
import tempfile
import urllib.request

import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target, _read_video

# 4:3 is the render aspect here (320x240); this is Runway's nearest allowed
# ratio for gen4.5/gen4_turbo (1104/832 = 1.327). Other gateway models
# have different allowed sets -- pass `ratio=` explicitly for those.
DEFAULT_RATIO = "1104:832"
RUNWAY_DURATION_MIN_S, RUNWAY_DURATION_MAX_S = 2, 10


def frame_to_data_uri(frame):
    """(H,W,3) uint8 -> `data:image/png;base64,...` -- Runway's own
    documented prompt_image form (up to 5MB); a 320x240 PNG is a few KB,
    so no hosting/upload step is needed to satisfy pixels-only
    conditioning. PNG, not JPEG: lossless, so the model sees exactly the
    rendered prefix frame, not a re-encoded approximation of it."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def duration_for(n_frames, target_fps, allowed_durations=None):
    """Smallest integer second count covering n_frames at target_fps,
    clamped to Runway's own [2, 10] -- generate at least as much as
    scoring needs, never less, then let resample_to_target trim.
    `allowed_durations` (2026-09-12, the other gateway models): the
    model's own allowed clip lengths, e.g. (5, 10) for gen4_turbo/kling
    or (4, 6, 8) for veo3.1 -- the smallest allowed value covering the
    need is used (and billed), the largest if none covers it."""
    need = math.ceil(n_frames / target_fps)
    if allowed_durations:
        covering = [d for d in sorted(allowed_durations) if d >= need]
        return covering[0] if covering else max(allowed_durations)
    return max(RUNWAY_DURATION_MIN_S, min(RUNWAY_DURATION_MAX_S, need))


def _download(url):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    urllib.request.urlretrieve(url, path)
    return path


def _default_client_factory():
    from runwayml import RunwayML   # lazy: only a real call needs the SDK/key
    if not os.environ.get("RUNWAYML_API_SECRET"):
        raise RuntimeError("RUNWAYML_API_SECRET not set -- see vitals/adapters/runway.py's own module "
                           "docstring for setup (the SDK reads exactly this variable).")
    return RunwayML()


PROMPT_IMAGE_MIN_SIDE = 256   # gateway rule, surfaced by h3_max (2026-09-14): "Height and width must be at least 256px"


def _upscale_for_prompt(frame, min_side=PROMPT_IMAGE_MIN_SIDE):
    """Integer-factor upscale (nearest: every source pixel becomes an
    exact block, nothing invented) of a frame whose shorter side is below
    the gateway's minimum -- 320x240 -> 640x480. The model is conditioned
    on the same picture at twice the size; the output is resampled back
    to target_hw as before, so scoring is untouched."""
    h, w = frame.shape[:2]
    if min(h, w) >= min_side:
        return frame
    k = math.ceil(min_side / min(h, w))
    return np.repeat(np.repeat(frame, k, axis=0), k, axis=1)


def make_runway_generate_fn(scenario_name, model="gen4.5", ratio=DEFAULT_RATIO, seed=0,
                            target_fps=30, target_hw=(240, 320), wait_timeout_s=600,
                            client_factory=None, log=print, allowed_durations=None,
                            supports_seed=True):
    """Builds `generate_fn(prefix_frames, n_frames) -> continuation_frames`
    for one Runway-gateway model. `model`: any id the SDK's
    `image_to_video.create` accepts (`gen4.5`, `gen4_turbo`, `veo3.1`,
    ...). `client_factory`: injection point (same pattern as
    VideoWorldModel's own generate_fn/phi_fn) -- tests pass a fake
    client so nothing here needs a key or network; production uses the
    real SDK client via RUNWAYML_API_SECRET.

    Per-model request shape (2026-09-14 smokes, the gateway's own 400s):
    `ratio=None` omits the key (h3_max: "Unrecognized key: ratio");
    `supports_seed=False` omits `seed` (gemini_omni_flash_1.1:
    "Unrecognized key: seed" -- that model is then NOT seed-reproducible
    on the provider side, recorded per population). A task that FAILS on
    the provider side is re-raised as a plain RuntimeError carrying the
    provider's failure text (the SDK's TaskFailedError holds a lock and
    cannot cross a Modal boundary, which hid seedance2.5's first failure
    entirely)."""
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
                        f"(pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]
    make_client = client_factory or _default_client_factory

    def generate_fn(prefix_frames, n_frames):
        client = make_client()
        last_frame = np.asarray(prefix_frames[-1])   # single-image conditioning, same as wan/hunyuan
        duration = duration_for(n_frames, target_fps, allowed_durations)

        request = dict(model=model, prompt_image=frame_to_data_uri(_upscale_for_prompt(last_frame)),
                       prompt_text=prompt, duration=duration)
        if ratio is not None:
            request["ratio"] = ratio
        if supports_seed:
            request["seed"] = seed
        task = client.image_to_video.create(**request)
        try:
            done = task.wait_for_task_output(timeout=wait_timeout_s)   # raises on FAILED (SDK's own polling)
        except Exception as e:
            details = getattr(e, "task_details", None)
            failure = None
            if details is not None:
                failure = {k: getattr(details, k, None) for k in ("id", "status", "failure", "failure_code", "failureCode")}
            log(f"[runway:{model}] task FAILED: {type(e).__name__}: {e}; details={failure}")
            raise RuntimeError(f"runway task failed for model {model!r}: {type(e).__name__}: {e}; "
                               f"details={failure}") from None
        if getattr(done, "status", None) != "SUCCEEDED" or not getattr(done, "output", None):
            raise RuntimeError(f"runway task {getattr(done, 'id', '?')} ended with status "
                               f"{getattr(done, 'status', None)!r} and no output")
        cost = getattr(getattr(done, "cost", None), "credits", None)
        log(f"[runway:{model}] task {done.id} SUCCEEDED, duration={duration}s, cost_credits={cost}")

        path = _download(done.output[0])
        try:
            raw_frames, src_fps = _read_video(path, default_fps=24)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return resample_to_target(np.asarray(raw_frames), src_fps, target_fps, n_frames, target_hw)

    return generate_fn
