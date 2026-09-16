# Adding a new model

This is the checklist for plugging a new world/video-generation model into
VITALS. Two models (NVIDIA Cosmos-Predict2, Alibaba Wan2.1) and one
diagnostic non-GPU baseline (`pixel_constant_velocity`) already go through
this exact path — read `vitals/adapters/wan.py` alongside this document if
you want a second worked example; it's the more recently added of the two
real models and the cleaner integration (no CLI/subprocess wrapping).

Everything below is grounded in real integration work already done on this
codebase, including three real bugs found while doing it (timeout
mismatch, a stale backend registry, a silent K-count mismatch). Where a
step exists specifically *because* one of those bugs happened, it says so
— skipping it reintroduces the exact failure that was already found and
fixed once.

## The contract

VITALS never talks to your model directly. It talks to a plain Python
function:

```python
def generate_fn(prefix_frames: np.ndarray, n_frames: int) -> np.ndarray:
    """prefix_frames: (T, H, W, 3) uint8 RGB, the real rendered conditioning
    video (T=30 frames at this project's own 30fps/320x240 convention).
    Returns: (n_frames, H, W, 3) uint8 RGB, your model's predicted
    continuation, resampled to this project's own fps/resolution/frame
    count before returning."""
```

That's the whole interface. `vitals/adapters/video_model.py::VideoWorldModel`
wraps this closure: it renders the real physics prefix, calls your
`generate_fn`, stitches your continuation back on, sends the full clip
through the real Phi pipeline (SAM2 tracking + reconstruction, unmodified),
and scores the result with the exact same detectors every other condition
in this project uses. **You do not touch anything downstream of
`generate_fn` — that is the whole point of the adapter boundary.**

Your model does not have to run on Modal. It does not have to be a GPU
model at all — `pixel_constant_velocity.py` is a pure-numpy `generate_fn`
with zero external dependencies, built specifically to prove this
boundary is real. Modal is just this project's own choice of GPU host;
the steps below assume it because that's what the existing reference
implementations use, but if your model runs behind your own API, `make_
<name>_generate_fn(...)` can just call that API directly — nothing else
in this document changes.

## Step by step

**1. Decide your conditioning modality, and know what you're giving up.**
The prefix handed to you is 30 frames (1s at 30fps) of real video, but
your model's own API may only accept a single image. That's fine — Wan2.1
only uses `prefix_frames[-1]`, plus a text prompt — but state this
honestly in your adapter's own docstring the way `wan.py` does, rather
than silently discarding information VITALS thinks it gave you.

**2. Write `vitals/adapters/<your_model>.py`.**
Expose one factory function, `make_<your_model>_generate_fn_modal(scenario_name, ...)`
(or `_local`, if not Modal-hosted), that returns a `generate_fn` closure
matching the contract above. Inside it:
- Look up `scenario_prompts.SCENARIO_PROMPTS[scenario_name]` for your text
  prompt — **do not invent a prompt at call time.** Every model is scored
  against the identical frozen wording per scenario; that's the whole
  point of the input-normalization protocol (compare models under matched
  settings, not under whatever prompt happened to be typed that day).
  `cosmos.py`/`wan.py` both raise a `KeyError` with an explicit message if
  a scenario has no registered prompt, rather than falling back to
  something invented — keep that behavior, don't paper over it.
- Call your model, then resample its output through
  `vitals.adapters.video_utils.resample_to_target(frames, src_fps, dst_fps,
  n_target_frames, dst_hw)` — nearest-frame resampling plus resize plus
  trim/pad, so every model's output lands in this project's own
  30fps/320x240/exact-frame-count convention regardless of its own native
  output rate.

**3. If GPU-hosted: write `remote/modal_app_<your_model>.py`.**
Mirror `remote/modal_app_wan.py`'s own structure — it's the cleaner of
the two reference apps (native `diffusers`, no CLI subprocess wrapping):
one `modal.App(...)`, an `image` with your model's pip dependencies, a
persistent `modal.Volume` for checkpoint caching so cold-downloads happen
once, and one `@app.function(image=..., gpu=..., timeout=...)`-decorated
inference function.

**Pick your GPU tier from a real measurement, not the model card.** Wan's
own card said "~16GB VRAM"; the actual pipeline needed ~22GB before a
single inference step ran. Confirmed by a real OOM, not guessed. Start
one tier above what the card claims if you're not going to load-test it
yourself first.

**Set your function's `timeout=` generously, and then go do step 6 —
this is the single most consequential number in this whole checklist.**
Wan's own deployed `image2video` has `timeout=1200`; the very first
integration used a flat 600s client-side default for every model, which
was fine for Cosmos but silently *shorter* than Wan's own server-side
ceiling. The client abandoned a Wan call that was still legitimately
running, then immediately fired an overlapping second attempt for the
same episode — which is what actually produced the failure, not a real
hang. Full incident, if you want the details: `AGENT.md`, M6.5.

Deploy once: `modal deploy remote/modal_app_<your_model>.py`. Redeploy
only when the image itself changes — your `vitals/adapters/*.py` code is
mounted at container startup, so ordinary edits need no redeploy.

**4. Register the model name in exactly one place:**
`vitals/adapters/__init__.py::MODEL_REGISTRY`. This dict is the single
source of truth `run_model_population.py` (which names `--model` accepts)
and `render_survival_report.py` (which category/color a results file
gets) both import via the `REAL_MODEL_BACKENDS` set it computes — **do
not let a second file learn your model's name independently.** The
reason this rule exists: it already broke once. Before this registry
existed, `render_survival_report.py` had a hardcoded `backend ==
"cosmos"` check as its only non-baseline category, so Wan's own real
results silently got plotted and colored as if it were a naive baseline
— a real model, mis-categorized as `ConstantVelocity`'s peer, purely from
one file not knowing a second model's name existed.

```python
"<your_model>": dict(license="<spdx-id-or-name>", restricted=False),
```

**If your model's license carries a jurisdiction or scale restriction
(a MAU threshold, a territorial exclusion, a non-commercial clause),
set `restricted=True` and add a `restriction_note` a human can act on —
not just "has a restrictive license," the actual restriction:**

```python
"<your_model>": dict(license="some-community-license", restricted=True,
                     restriction_note="Not licensed for use in <X>; see <model card URL>"),
```

A restricted model is excluded from `REAL_MODEL_BACKENDS` — and therefore
invisible to `--model`, every report, and the lambda sweep's
auto-detection — unless a human explicitly sets
`VITALS_ENABLE_RESTRICTED_MODELS=<your_model>` for that invocation. This
is not optional ceremony: VITALS is meant to be handed to other
companies, some of whom cannot legally use every open-weight model that
exists. Restricting by default and opting in explicitly means the core
benchmark never silently hands an adopter something they aren't
permitted to run. State the same restriction plainly in your own
adapter's module docstring too — the same honesty `wan.py`'s own
docstring already models for its own license tradeoffs.

**5. Add one dispatch line in `scripts/run_model_population.py::make_generate_fn()`:**

```python
if model_name == "<your_model>":
    from vitals.adapters.<your_model> import make_<your_model>_generate_fn_modal
    return make_<your_model>_generate_fn_modal(scenario)
```

**6. Add your model's own entry to `EPISODE_TIMEOUT_S_DEFAULT`** (same
file, near the top) — the client-side wall-clock deadline per episode
attempt. Set it comfortably above your own Modal function's `timeout=`
from step 3. Anything not in this dict silently gets a 600s default,
which is almost certainly wrong for a real diffusion model. This is the
single line that would have prevented the M6.5 incident.

**7. Register a prompt for every scenario you intend to evaluate on**,
in `vitals/adapters/scenario_prompts.py::SCENARIO_PROMPTS` — this is
shared across every model, so if the scenario already has an entry (most
do), you don't need to add one; you only add a new entry when the
*scenario* itself is new to this project, not when the *model* is.
Keep the wording a plain description of what's actually in the scene
(geometry, color, motion) — no adjectives aimed at flattering any one
model.

## Validate before you scale

Run a smoke-sized batch first, on a scenario that already has a full,
validated pipeline (`occlusion_corridor` or `ramp_descent_high_friction`
are the safest choices — `collision`, K=2, has its own extra sharp edge,
see below):

```bash
python3 scripts/run_model_population.py --model <your_model> \
    --scenario occlusion_corridor --seeds 1,2,3,4,5
```

Every new model integrated into this project so far has surfaced at
least one real, non-obvious bug on its first real run — that is the
expected, normal outcome of this step, not a sign something is wrong
with your adapter specifically. Don't request a full n=80 population
until a 5-seed batch completes cleanly.

## Then run the full evaluation

Once a model is registered and validated on at least one scenario:

1. **GATE 2** (instrument tax, if not already run for that scenario):
   `VITALS_MANIFEST=configs/manifests/<scenario>.yaml python3 scripts/run_gate2.py`
2. **Population scoring**: `scripts/run_model_population.py --model <your_model> --scenario <scenario> --seeds 1..N`
3. **Lambda sweep** (free once trajectories are cached — re-scores them at
   every λ level with zero new GPU calls):
   `scripts/run_lambda_sweep.py --manifest configs/manifests/<scenario>.yaml`
   — this auto-detects any `REAL_MODEL_BACKENDS` name with a populated
   `results/trajectories/<scenario>_<model>/` cache dir and adds it as an
   extra condition. No code change needed for this step once step 4 above
   is done.
4. **Regenerate the reports**: `scripts/render_survival_report.py`,
   `scripts/render_gate2_report.py`, `scripts/render_score_report.py` —
   all auto-discover from whatever is in `results/`, no manual wiring.

## Known sharp edges

- **A K>1 scenario (`collision`) only ever scores one pre-registered
  object.** Phi's video reconstruction is single-object-only project-wide
  — your model's output is scored on the same one object every other
  model is scored on there. Nothing to do on your end, but don't be
  surprised the reference ensemble is K=2 while your candidate is scored
  as K=1; that asymmetry is already handled in the scoring code, not
  something your adapter needs to account for.
- **A results file is overwritten wholesale per invocation, never
  merged.** If a batch partially fails (a dropped network connection
  mid-run has done this before, at scale — not a Modal capacity problem,
  a local connectivity one), re-run with the *full* original `--seeds`
  list, not just the failed ones, or you'll silently shrink an
  already-larger population back down.
- **`--max-workers` above ~3 for a large, slow model risks
  client-side timeouts that look like server failures but aren't** —
  each concurrent call's wall-clock includes any time it spends queued,
  not just time actually generating. If a batch shows a high failure
  rate, check `modal app logs <your-app-name>` for the real story before
  assuming it's your model or your GPU tier — the actual cause so far has
  always been either a timeout mismatch (step 6) or a local network drop,
  never a problem with the model itself.
