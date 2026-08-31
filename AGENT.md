# VITALS — Agent Handoff

**Read this before writing any code. Re-read §3 and §4 before every commit.**

This document is the operating contract for building VITALS. It exists because
the system's correctness lives in a handful of invariants that are easy to
violate while making a change that looks like an improvement. A violated
invariant does not throw — it silently produces plausible numbers that are
wrong. Most of this document is about preventing that.

---

## 1. Intent

### What this is

VITALS measures **how long a world model's rollout remains a valid prediction
of the physical world, and which necessary property fails first.**

The output is not a score. It is:
- a **duration** (validity interval, in seconds, with a confidence interval),
- a **termination profile** (which property failed, as cause-specific incidence),
- a **calibration horizon** (when the model's uncertainty stops tracking reality).

### Why it is built this way

Two problems with existing world-model benchmarks:

1. They report a scalar at a horizon the benchmark chose. Practitioners need a
   duration — how long can I plan before re-observing?
2. Their task selection is curated, not derived. Nothing explains why the
   benchmark contains the scenarios it contains.

VITALS answers (2) by starting from a formal definition and deriving what must
necessarily follow, and (1) by treating evaluation as a **time-to-event
problem** rather than a scoring problem.

### The theory, in one block

A coherent predictive model of the physical world is a **structured stochastic
transition model over a factorized state**:

```
p(s_{t+1} | s_t, a_t),   s_t = (phi, {e_i}, {rho_ij})
                                 |     |       |
                          invariant  entities  relations
                            frame
```

Six components, six necessary observable properties:

| ID | Component | Property |
|----|-----------|----------|
| P1 | invariant substrate | **frame structure** — what doesn't change, doesn't change |
| P2 | individuation | **entity persistence** — objects continue to exist, keep identity |
| P3 | relational factorization | **relational persistence** — support, contact, containment hold |
| P4 | transition operator | **physical dynamics** — motion stays plausible |
| P5 | modularity of mechanism | **causal consistency** — interventions produce their effects, and *only* those |
| P6 | distributional output | **uncertainty calibration** — uncertainty tracks real ambiguity |

**P1 ≺ P2 ≺ P3 ≺ P4** is a dependency ordering, not a preference. Motion is
undefined without a frame; relations are undefined without identified entities.
This ordering *derives* the precedence rule in §3.6.

P5 and P6 sit outside the chain because they are properties of the operator,
not of the state — which is why they need different experimental designs.

### The core measurement loop

```
M reference rollouts from perturbed initial conditions (same s_0, scale lambda)
  -> threshold = (1-alpha) quantile of how much references deviate from EACH OTHER
  -> run candidate, find first SUSTAINED crossing of that threshold
  -> (time, which detector fired) per rollout
  -> Kaplan-Meier survival curve; median = validity interval
  -> Aalen-Johansen cause-specific incidence = termination profile
```

Everything else in this repository is elaboration on those five lines.

### What this system does NOT claim

VITALS establishes P1–P6 as **necessary** conditions. The only valid inference
is the contrapositive: *a model that fails a property does not have behavioral
world understanding.* Passing is **not** sufficient and must never be described
as evidence of understanding. This is not modesty — it is what makes the
construct measurable without representational access.

---

## 2. Glossary

| Symbol | Meaning |
|--------|---------|
| `s_0` | nominal initial configuration of a scenario |
| `Sigma` | perturbation model over `s_0` |
| `lambda` | dimensionless scale of `Sigma`; swept, never fixed |
| `M` | number of reference realizations per episode |
| `N` | number of samples drawn from the model under test |
| `R` | the reference ensemble, `{r_1..r_M}` |
| `C` | conditioning prefix given to the model (`T_c` frames) |
| `Phi` | measurement operator: video -> structured state |
| `sigma_k` | deviation statistic for risk `k`, a `(T,)` array |
| `theta_k` | threshold for risk `k`, estimated not chosen |
| `alpha` | false-positive rate of each detector (0.01) |
| `pi` | persistence window; crossing must be sustained this long |
| `T_max` | rollout horizon |
| `pi*` | the target property an episode is designed to falsify |
| `VI_q` | validity interval at survival level q |
| `L0 / L1` | measured from ground-truth state / measured from video |

---

## 3. INVARIANTS — never violate

Each of these, if broken, produces wrong numbers without raising an error.
If a change appears to require breaking one, stop and escalate to a human.

### 3.1 Measurement symmetry

`Phi` runs **identically** on reference video and candidate video. The
reference ensemble is measured *through Phi from rendered frames*, never read
from simulator state.

> Why: reading simulator state for the reference gives it an instrument
> advantage the model does not have. Every threshold shrinks, and every
> validity interval in the paper inflates. This is the single most damaging
> possible bug and it looks like an optimization.

Simulator ground truth is permitted in exactly one place: **validating** `Phi`
and the detectors (`§7 M2`). Never in scoring.

### 3.2 No language model in the measurement path

No VLM judge, no MLLM scorer, no text-conditioned component, no
language-supervised features (CLIP-family excluded; DINO-family fine).

> Why: this is not about neural vs. classical. Perception components emit
> *measurements* that have ground truth and therefore a publishable error
> floor. A judge emits a *verdict* against a plausibility scale that does not
> exist anywhere, so its error is unbounded in principle and correlates with
> the model families it judges.

Where a component offers both text and visual prompting, use visual — prompt
segmentation with the ground-truth frame-0 mask from the manifest.

### 3.3 Thresholds are estimated, never chosen

`theta_k` is always the `(1-alpha)` quantile of `sigma_k` computed
**leave-one-out among reference realizations**, per scenario and per time bin.

> Why: it removes threshold choice as a free parameter, gives every detector a
> stated false-positive rate, and — critically — self-immunizes detectors
> against their own systematic weaknesses. A tracker that confuses a dark ball
> with its shadow does so on the references too, so that jitter is already
> inside the reference distribution and the threshold sits above it.

Never hardcode a threshold to make a test pass. If a detector won't fire,
the statistic is wrong, not the threshold.

### 3.4 Trajectory is the primary artifact

Everything upstream of rendering speaks `Trajectory`. Do not pass video
between modules that could pass state.

> Why: it makes mutants trivial, makes observation interventions free
> re-renders of the same physics, and makes L0/L1 comparison possible at all.

### 3.5 Mutants corrupt state, never pixels

A mutant is a corrupted `Trajectory` that is then rendered. Never add noise or
artifacts to video to simulate a defect.

> Why: a state-corrupted mutant is photometrically identical to a real render
> except for the injected defect. A pixel-corrupted one is not, and conflates
> the defect with a rendering difference.

### 3.6 The precedence rule is derived — do not reorder

```
R1 (frame) > R2 (existence) > R3 (identity) > R4 (constraint) > R5 (kinematic)
```

Follows from the dependency ordering in §1. A violation at a lower level makes
measurement at a higher level ill-defined: if the frame is broken, "the object
moved" has no referent.

### 3.7 The conditioning prefix comes from a held-out realization

`C` must be drawn from a realization **not** in `R`.

> Why: otherwise the model is scored against a distribution that contains its
> own conditioning, which biases every horizon upward.

### 3.8 Candidate and reference may have different object counts

`cand.K != ref.K` is **not** an error — it is precisely what existence failure
means. Any code that assumes matching `K`, zips object lists, or indexes by
position across candidate/reference is a bug.

### 3.9 Fair scoring rules only

Use `fair_crps` / `fair_energy_score`, never the plain versions.

> Why: the fair variants remove finite-ensemble bias so M≈20–30 approximates
> the infinite-ensemble value. Plain CRPS rewards larger ensembles and makes
> models with different N incomparable.

### 3.10 Bootstrap over episodes, never over rollouts

Rollouts within an episode are correlated. Resampling them inflates apparent
precision.

### 3.11 Report no aggregate scalar

There is no `overall_score()`. Do not add one, however convenient.

> Why: aggregating three horizons plus a five-way termination profile destroys
> the diagnostic content that motivates the benchmark, and invites optimizing
> against the metric.

### 3.12 Pre-register, then freeze

`alpha`, `pi`, precedence, detector statistics, covariate set and the lambda
grid are frozen **before** any model is evaluated, and published. Sensitivity
analysis is reported; tuning after seeing results is not.

---

## 4. ANTI-PATTERNS

Things an agent will plausibly try. All are wrong.

| Tempting | Why it's wrong |
|---|---|
| "Read object poses from the simulator for the reference — it's more accurate" | Breaks §3.1. Catastrophic and silent. |
| "Use a VLM to check if the video looks physically plausible" | Breaks §3.2. |
| "This detector never fires; lower the threshold to 0.5" | Breaks §3.3. Fix the statistic. |
| "Add an `overall_score` so we can rank models" | Breaks §3.11. |
| "Cache Phi output keyed on filename" | Must key on `(content_hash, phi_version)`, or stale results survive a Phi change. |
| "Compare candidate and reference object-by-object with `zip`" | Breaks §3.8. |
| "Tune alpha until the baselines look right" | Breaks §3.12. Baselines landing wrong means the metric is wrong. |
| "Sample mutant injection times uniformly over [0, T_max]" | Produces silent mutants where dynamics aren't changing (§8.2). Weight toward intervals where reference spread is growing. |
| "Skip the artifact-injected validation set, we can add it later" | It is the only thing bounding the reference/candidate asymmetry in §9. Without it the instrument floor is a guess. |
| "Use a generative video model to make reference data" | Circular. References must come from an engine with known dynamics. |
| "Delete `physics/synthetic.py`, we have MuJoCo now" | It keeps CI fast and deterministic and isolates blame between physics bugs and statistics bugs. Keep it. |
| "Fix the failing test by relaxing the assertion" | Every gate in §7 exists to fail loudly. A relaxed gate is a design decision made silently. |

---

## 5. Current repository state

```
vitals/
  types.py              Trajectory, Event, EpisodeSpec              DONE
  physics/
    __init__.py         make_backend("mujoco"|"synthetic")          DONE
    perturb.py          Sigma: "full" isotropic 3D, or
                        "velocity_x_only" for 1-DoF scenarios        DONE
    synthetic.py         analytic ramp double, no engine required    DONE (keep)
    synthetic_corridor.py analytic flat-corridor double              DONE (keep)
    runner.py           MuJoCo rollout + fork + GATE 0 + keyframe    DONE, GATE 0 PASSES
  mutants/library.py    null, vanish, duplicate, velocity_freeze,
                        wrong_gravity, teleport, jitter             DONE
  detect/
    statistics.py       sigma_existence, sigma_kinematic,
                        sigma_interpenetration                      2 DONE, 1 OUT OF SCOPE (P3)
    thresholds.py       whole-path calibrated, per time bin         DONE (see note below)
    events.py           first crossing + precedence (R2, R5 live)   DONE
    physical_constants.py  fit implied physical constant from
                        tracked motion, z-score vs reference band     DONE for occlusion_corridor
                                                                        deceleration (see note below);
                                                                        ramp_descent gravity NOT DONE
                                                                        (two entangled unknowns)
  stats/
    scoring.py          fair CRPS, energy score, spread-skill,
                        rank histogram                              MISSING — needed for P6 (M7)
    survival.py         KM, Aalen-Johansen, VI, bootstrap CI        DONE
  adapters/base.py      WorldModel protocol, CopyLastState,
                        ConstantVelocity                            DONE (baselines only --
                                                                      real open-weight models still MISSING, M6)
  render/
    __init__.py          Renderer protocol, Frames, GroundTruth        DONE
    mujoco_renderer.py   MuJoCo-based Renderer: RGB + segmentation
                        + depth + camera extrinsics, one instance      DONE, LOW REALISM (see note below)
  phi/                   milestone M4, IN PROGRESS (see note below)
    segmentation.py       SAM2 promptable video tracking                DONE, occlusion-recovery
                                                                          cliff at ~22-27 frames
    reidentify.py          two-tier reacquisition after the cliff:
                        physics-prior PRIMARY, DINO appearance FALLBACK  LOGIC DONE, FULL ORGANIC
                                                                          END-TO-END VALIDATION
                                                                          IN PROGRESS -- 2 real bugs
                                                                          found+fixed, 1 still open
                                                                          (see note below)
    motion_prior.py       pixel-space kinematic extrapolation +
                        candidate gating, pure numpy, no GPU required    DONE (synthetic-only,
                                                                          see note below)
    thresholds.py          DINO similarity threshold (FPR-only +
                        balanced/Youden's J)                             DONE
    reconstruct.py          Phi outputs -> real Trajectory; the M4
                        done-criterion itself                            DONE for occlusion_corridor
                                                                          (flat-ground only, see note below)
scenes/ramp_descent.xml         MJCF, Z-up, ball/ramp only, NO occluder  DONE
scenes/occlusion_corridor.xml   MJCF, Z-up, flat + long wall, NO ramp    DONE
configs/manifests/ramp_descent.yaml        P4, M=30, lambda=1.0          DONE
configs/manifests/occlusion_corridor.yaml  P2, M=30, lambda=6.0          DONE
scripts/run_l0_demo.py  manifest-driven L0 instrument-validation demo  DONE, PASSES ALL 4 (2 scenes x 2 backends)
scripts/run_eval.py     manifest-driven GATE 3: scores a WorldModel,   DONE, GATE 3 PASSES ALL 4
                        not a mutant -- the first script that
                        evaluates a MODEL rather than the instrument
tests/test_core.py      9 tests                                    PASS
tests/test_gates.py     GATE 0, both scenes, skipped if no mujoco  PASS
tests/test_render.py    M3 shape/range/segmentation sanity checks  PASS
tests/test_physical_constants.py  fit precision + corruption catch  PASS
tests/test_motion_prior.py        fit/predict/gating, no GPU needed PASS
tests/test_reconstruct.py         M4 done-criterion, position + present PASS
remote/modal_app.py     Modal GPU app (SAM2+DINOv2 inference)         DONE, replaces the CMU
                                                                        GHC SSH workflow entirely
                                                                        (see note below)
remote/call.py           Invokes the DEPLOYED app -- the correct        DONE (see note below --
                        way to call it; never `modal run` directly     "Invocation pattern")
```

**M3 is deliberately the cheap version, not the Blender pipeline AGENT.md
originally sketched.** `mujoco.Renderer` already emits RGB, per-pixel
instance segmentation, and depth from the same renderer instance, plus
camera pose is directly readable off `renderer.scene.camera[0]` after
`update_scene` -- verified: `cam_mat` is orthonormal to 1e-5, segmentation
matches the ball's actual screen position exactly (checked visually, not
just by shape), depth increases correctly with distance and respects
occlusion boundaries. This unblocks M4 (Phi) immediately instead of after
a separate rendering-engine integration project.

The tradeoff, honestly: MuJoCo's default materials/lighting are flat CG,
which is exactly the T2 residual risk (render realism / domain shift) the
spec already calls out as highest-priority to track. `realism` is accepted
by `MujocoRenderer.render()` for forward compatibility but today only
toggles shadows. Good enough to develop and validate Phi against; **not**
sufficient basis for claims about how a real video-generation model would
perform, until a realism upgrade (better materials, or swapping in a
higher-fidelity `Renderer` implementation via the same protocol) happens.

Also noted, not yet investigated: MuJoCo prints `WARNING: ARB_clip_control
unavailable while mjDEPTH_ZEROFAR requested, depth accuracy will be
limited` on this machine (likely a software/CPU rendering fallback,
common on macOS). Depth values are directionally correct and pass the
current sanity checks (positive, finite, correct occlusion ordering), but
absolute depth accuracy hasn't been validated against known distances.
Do this before depth feeds any real measurement (M4's monocular-depth-based
3D position estimate).

Two scenes, deliberately never combined (3.6): `ramp_descent` tests P4/R5
(incline kinematics) in isolation; `occlusion_corridor` tests P2/R2 (entity
persistence) in isolation, with a flat corridor and a wall that has real
length along the direction of travel — long enough that a rolling ball can
stop short of it, stop hidden behind it, or roll far enough to re-emerge on
the far side, depending only on lambda-scale perturbation of its initial
speed. There used to be one scene with both a ramp and a wall in it; it was
split into these two because a model could nail one necessary property and
hallucinate the other, and a combined scene can't tell you which one broke.

Verified working, both `synthetic` and `mujoco` backends, both scenarios:
`python3 tests/test_core.py` (9 pass), `python3 tests/test_gates.py` (GATE 0
passes on both scenes), and `python3 scripts/run_l0_demo.py` /
`VITALS_MANIFEST=configs/manifests/occlusion_corridor.yaml python3
scripts/run_l0_demo.py`, each with `VITALS_BACKEND=synthetic|mujoco` — GATE 1a
passes on all four combinations (0.000-0.033 against alpha=0.01), all GATE 1b
sensitivity cases fire on the correct risk.

**GATE 3 passes on both scenarios, both backends** (`scripts/run_eval.py`,
`--adapter constant_velocity`): VI_50 on `occlusion_corridor` (already
rolling at roughly constant decelerating speed -- close to free flight) is
~2.2s on both backends, nearly 2x `ramp_descent`'s ~1.3s (incline
acceleration then a sharp direction change at the ramp/floor transition --
contact-dominated). `constant_velocity` also beats `copy_last_state` in
both scenarios (1.03s / 1.47s), consistent with extrapolating momentum
being a strictly better assumption than freezing whenever there's real
motion. No censoring in either case -- both baselines are bad enough that
every episode eventually diverges before `T_max`, which is expected: they
are the floor, not a claim of adequacy.

**`detect/physical_constants.py` is a second, independent operationalization
of "is the physics right,"** complementary to (not a replacement for) the
R1-R5 threshold-crossing detectors: instead of comparing a raw trajectory
against a reference band (model-free), it fits a single implied physical
parameter (a deceleration) from tracked motion and compares that ONE
number, per episode, against the reference ensemble's own distribution of
fitted values — inspired by WorldBench's parameter-estimation track, but
using VITALS' free exact ground-truth camera pose instead of a real
checkerboard calibration. `occlusion_corridor`'s rolling-deceleration fit
clusters extremely tightly (CV=0.12% on M=30 references, using the
validated `DEFAULT_CORRIDOR_WINDOW=(0.5s, 1.3s)` — see defect #11) and
reliably flags a synthetically corrupted candidate trajectory (z-score in
the hundreds vs. single digits for a normal held-out episode). Not yet done
for `ramp_descent`: gravity recovery there entangles g with the incline
friction coefficient in one acceleration term, unlike the corridor's
single unknown — deferred, not attempted with a hand-waved simplification.

**`phi/reidentify.py`'s re-identification strategy was redesigned mid-build,
from DINO-appearance-primary to physics-prior-primary.** DINO similarity,
even after calibration improvements (M=20 calibration set, Youden's J
balanced threshold), measured 0/N matches on `occlusion_corridor`'s hard
velocity band (v0=4.5-4.9 m/s) — root-caused to genuinely low appearance
separability for a low-texture, rotating, rolling object, not a
calibration-sample-size problem (see `reidentify.py`/`thresholds.py`
history). While validating a physics-prior alternative, found a SEPARATE,
compounding bug: the "reappearance frame" was defined as the first frame
with ANY visible pixel, which on these same hard cases was a 1-6px sliver
(confirmed directly: 5px at the naive frame vs. ~65-70px once genuinely
re-emerged, ~6 frames later) — too small for any detector, SAM2 or DINO,
to reasonably segment or embed. Fixed by requiring >=50% of the object's
own pre-occlusion max mask area (per-episode-normalized, not a fixed pixel
count) before calling a frame "reappeared."

With that fix, kinematic (constant-deceleration) extrapolation from an
object's own recent tracked centroids — using the SAME reference-ensemble
deceleration prior `detect/physical_constants.py` validates
(CV=0.12%) — predicts the true reappearance pixel position to within
2.7-3.9px on the exact hard cases that broke DINO, and the real SAM2
automatic-mask-generator candidate nearest that prediction is the true
ball (IoU 0.55-0.75, cleanly separated from the next-nearest false
candidate at ~25px/IoU<0.02) in 10/10 tested episodes. `phi/motion_prior.py`
implements this; `reidentify.py::track_with_reidentification` now tries it
FIRST and falls back to DINO only when a track has too little
pre-occlusion history to fit one (`motion_prior.MIN_FIT_FRAMES`). DINO
stays the relevant signal for a future K>1 scenario, where position alone
can't disambiguate which of several plausible nearby objects is correct —
it just isn't primary for today's K=1 scenarios, where it's reliably
outperformed by a signal the object's own motion already gives for free.

**`motion_prior.py` is synthetic-only, and says so in its own docstring —
do not let this quietly get assumed to generalize.** Its pixel-space
kinematic fit implicitly assumes (a) the camera's distance to the object
stays roughly constant across the occlusion gap (true here: the corridor
camera views side-on, so depth barely changes as the ball travels along
x) and (b) the camera itself is static. Neither is detected or corrected
for — a real deployment with substantial motion-in-depth would need actual
camera intrinsics plus a depth estimate (monocular depth, since a single
real camera has no depth for free) to convert pixel motion back to metric
3D before this extrapolation means anything; this is exactly the "3D
positions: multi-view triangulation, else monocular depth" row the M4
component table (§7) already planned and this module does not yet
implement. A moving camera (robot ego-motion, handheld, drone) is a
further, separate requirement (visual odometry/SLAM to separate camera
motion from object motion) not built speculatively, since no scenario here
has one yet. Neither gap is a §3.1 violation waiting to be fixed — knowing
your OWN camera's calibration is normal instrumentation, not privileged
access to the object's state, unlike reading simulator ground truth would
be. See T3 (§9), expanded with this specific finding.

**`phi/reconstruct.py` closes M4's literal done-criterion** ("Phi produces a
Trajectory from video, and the same sigma_k functions run on it unmodified")
for `occlusion_corridor`: mask centroids are unprojected to 3D via ray/plane
intersection against the scene's known camera calibration and a known,
fixed resting-height plane, then assembled into a real `Trajectory` that
`detect/statistics.py::sigma_existence`/`sigma_kinematic` run on with zero
modification (verified directly, `tests/test_reconstruct.py`). Two real
findings from validating it, both worth knowing before touching this
module again:

1. **Camera elevation matters a lot for 3D reconstruction accuracy, separate
   from how good it looks for a human watching a video.** The corridor's
   visualization camera (`elevation=-10`, chosen in `visualize_reference_
   ensemble.py` for framing) views the horizontal resting-plane at a near-
   grazing angle, which amplifies a small (~1.5px) mask-centroid bias into
   ~9-10cm of 3D position error via ray/plane intersection sensitivity.
   Steepening to `elevation=-45` cuts this roughly in half (~5cm mean error,
   measured on ground-truth masks as a SAM2 stand-in, isolating this
   method's own geometric error from SAM2's separately-measured tracking
   error) and further steepening plateaus -- the residual ~5cm appears to be
   the intrinsic mask-centroid-vs-true-center bias itself, not further
   fixable by camera angle. `reconstruct.py`'s own tests use `elevation=-45`
   for this reason; do not reuse the visualization camera for measurement
   purposes without re-checking this.
2. **`Trajectory.present` must mean "exists," not "currently visible" --
   and it's easy to get this wrong even after building an entire re-
   identification system specifically to make that distinction.** An early
   version set `present = mask non-empty`, which silently conflated
   "occluded" with "gone" -- exactly the failure mode `segmentation.py`'s
   cliff and `reidentify.py`'s search-and-reprompt exist to prevent, and
   caught only because it produced NaN downstream (see defect #13 below),
   not because it looked wrong on inspection. **FIXED**: `present` now
   follows `reidentify.py`'s own reacquisition verdict via `reid_events` --
   True for directly-visible frames and for any gap a later event confirms
   was successfully closed, False only for a gap that never closes. `pos`
   stays NaN for any non-directly-visible frame regardless (position is
   genuinely unmeasured during a gap, closed or not) -- `sigma_kinematic`
   already has documented, correct handling for that.

**Build-sequence step 27 -- full organic end-to-end validation of
`track_with_reidentification` -- is IN PROGRESS, not done.** Everything
validated before this point (10/10 correct reacquisition, AGENT.md's
earlier physics-prior section) ran `motion_prior`'s candidate-scoring logic
against a hand-picked frame at the *known, true* reappearance instant, via a
standalone script -- never the real, live search loop deciding for itself
when and whether to reacquire. Running the actual integrated function,
starting from frame 0 with real SAM2 tracking, surfaced four real, distinct
bugs. Three are fixed; one is open. Record precisely, because each is a
different kind of mistake worth not repeating:

1. **`load_mask_generator` never exposed `pred_iou_thresh`/
   `stability_score_thresh`, so `track_with_reidentification`'s real,
   shipped usage was always using SAM2's strict defaults (~0.88/0.95), not
   the looser `0.5/0.5` already known from earlier calibration work in this
   project to be necessary for this small, low-texture ball.** Confirmed
   directly, not inferred: with strict defaults, SAM2's automatic mask
   generator proposed only 4 candidates per frame around the true
   reappearance point, all large background regions (2800-48000px) --
   never anything near the ball's actual size. No candidate-scoring or
   gating logic downstream can recover from a search that never sees the
   right candidate at all. **FIXED** -- `load_mask_generator` now defaults
   to `pred_iou_thresh=0.5, stability_score_thresh=0.5,
   min_mask_region_area=0`. This was a bug in shipped production code, not
   a test artifact -- every caller that didn't explicitly override these
   two (previously unexposed) parameters was silently exposed to it.

2. **The physics-prior tier accepted the nearest scored candidate
   unconditionally -- no distance-rejection criterion existed at all.**
   Combined with search firing as soon as `empty_streak > forgiveness_
   frames` (almost immediately after a loss, usually long before a real
   occlusion could plausibly resolve), this locked onto a static background
   distractor at frame 46 of a 150-frame clip (23px from the prediction,
   accepted as a match) and tracked that wrong thing for the rest of the
   video (post-reacquisition IoU ~0.002). The standalone validation never
   caught this because it only ever tested at the correct, known
   reappearance frame -- never at a frame where the object plausibly isn't
   visible yet at all. **FIXED, in two layers**: (a) `motion_prior.
   mask_diagonal_px` + a distance gate scaled to the object's own
   last-known apparent size (not an arbitrary pixel constant); (b)
   persistence-tracking (`rejected_positions`, cleared on every successful
   match) -- a distance gate alone still eventually accepted the SAME
   static distractor once the physics prediction (converging toward the
   object's estimated resting position over many extrapolated frames)
   drifted close enough to it, even though the candidate itself never
   moved across 10 consecutive search attempts. A candidate near a
   position rejected earlier in the same gap is now held to a much
   tighter absolute bound instead of the size-based one.

3. **Root-caused, then FIXED (the accumulation-blanket hypothesis
   originally suspected here turned out to be wrong -- see defect #16 for
   the full account): the per-episode PIXEL-space quadratic
   (`PixelTrackFit`) re-derives deceleration from a short window of real,
   noisy tracked pixels, which is unreliable when extrapolated far.**
   Measured directly: on this exact gap, it implied the object stops at
   pixel x=107.9 when the true object was still at pixel x=68.3 by frame
   61 and still moving (true velocity there needs ~108 frames to actually
   stop, far longer than the pre-drop window available to fit from).
   **Fix**: `motion_prior.fit_metric_track`/`MetricTrackFit` -- fits only
   (position, velocity) per-episode, in metric 3D space (well-conditioned
   over a short window), and borrows deceleration from `physical_
   constants.py`'s already-calibrated reference-ensemble constant instead
   of re-deriving it. Validated directly against MuJoCo ground truth, not
   just internal consistency: the new fit's predicted pixel trajectory
   matches ground truth to within 0.9-1.9px throughout the same real gap.
4. **Surfaced precisely because fix #3 is now accurate: a physics
   prediction can walk into a STATIC, OCCLUDER-ADJACENT feature sitting on
   the object's own (correctly predicted, but still-occluded) path --
   tracked separately as defect #17, FIXED.** At frame 48 of the same gap,
   the object's true position is still within the wall's own occlusion
   span (known scene geometry), genuinely not visible yet -- but a SAM2
   candidate at that exact accurate predicted location got accepted
   anyway, almost certainly a real visual feature of the wall itself.
   Different cause from defect #15's random-background-distractor case,
   not just a different symptom.

**Scenario-selection trap, worth recording so it isn't rediscovered:** the
first end-to-end test seed (seed=1) happened to roll the ball far enough
down the corridor that it reappears at a genuinely tiny, permanent ~5px
apparent size (confirmed directly against `gt_masks` area across dozens of
frames -- not a transient sliver that grows, per defect #12's fix; it
simply never grows past ~5px for the rest of the clip). No automatic mask
generator at any reasonable threshold could reliably detect a 5px target --
this is a fundamentally different, harder problem (target too small to
exist as a candidate at all) than the appearance-ambiguity case this test
is meant to exercise, and produced a misleading negative result before
being identified. Switched to seed=2024 (drop=37, rise=60, reappearance
area ~50px), one of the already-characterized hard-band cases from earlier
DINO-calibration work. Any future end-to-end test on `occlusion_corridor`
should verify the chosen seed's reappearance size directly before trusting
results against it.

**Remote GPU infrastructure was migrated from the CMU GHC lab machine to
Modal (`remote/modal_app.py`), and this is a confirmed, complete fix, not
a workaround.** The old machine (a shared, 7.77GB RTX 2080) hit three
separate, real infrastructure ceilings during this validation work, none
of them fixable in place: (a) GPU VRAM OOM during SAM2 propagation with
multiple models resident, worsened by `expandable_segments:True` -- the
standard PyTorch fragmentation mitigation -- being confirmed incompatible
with this specific Turing-architecture card; (b) a hard 16GB *virtual*-
memory ulimit imposed by shared-machine policy, not raisable by an
unprivileged user, hit purely from loading a second small model instance
alongside an already-resident CPU-offloaded video predictor; (c) `/tmp`
and (eventually, confirmed by a second occurrence) `/var/tmp` both wiped
on every reboot -- sometimes multiple times per day, since it's a shared
public lab machine, not a dedicated one -- costing a full environment
rebuild each time, including one mid-debugging-session reboot that lost
render state and forced restarting from scratch.

Modal (`https://modal.com`, workspace **midcentury-labs**, app
**vitals-phi**) resolves all three at once: a dedicated container per
invocation (no shared-tenancy ulimits), GPU tier chosen per job
(`GPU_TYPE="A10G"`, 24GB -- vs. the old card's 7.77GB), and a persistent
Volume (`vitals-model-cache`) holding model weights and uploaded episode
data that survives indefinitely instead of being wiped. The image build
also fixes the SAM2-clobbers-torch problem (§ earlier defects, SAM2's own
unpinned dependency spec silently upgrading torch to an incompatible
build) durably rather than by re-running a shell script and hoping: SAM2
installs first, then `torch==2.6.0`/`torchvision==0.21.0` are re-pinned as
the final image layer, baked once at build time. `vitals/phi/*.py` is
mounted at container startup (`add_local_dir`, not baked into the image),
so local edits take effect on the next call with no image rebuild --
matching the fast-iteration workflow this project actually uses.

**Invocation pattern, corrected once and worth getting right permanently:
`modal deploy remote/modal_app.py` once, then call through
`remote/call.py`, never `modal run remote/modal_app.py::<fn>` directly.**
`modal run` spins up a brand-new EPHEMERAL app per invocation that tears
down when the call finishes -- correct, documented Modal behavior, not a
bug, but it means every debugging iteration leaves behind a new,
already-dead "vitals-phi" entry in the midcentury-labs dashboard.
Confirmed directly: the workspace accumulated 8 of these during defect
#16's investigation, purely from iterating with `modal run`. `remote/
call.py` instead uses `modal.Function.from_name("vitals-phi", ...)` to
look up and call directly into the one persistent deployed app -- verified
directly (`modal app list --json`) that repeated calls through it keep
exactly one `vitals-phi` entry, not a growing pile. Redeploy only when the
image itself changes (a new pip dependency, etc.); ordinary edits to
`vitals/phi/*.py` need no redeploy, same mount-at-startup behavior as
before.

Confirmed directly: `smoke_test` (torch/CUDA/SAM2/DINOv2 all load) passed
cleanly on the first real attempt (after one trivial fix -- `debian_slim`
has no `git` by default, needed for `pip install git+...`). More
importantly, the full `run_reidentification` end-to-end pipeline --
exactly the workload that repeatedly crashed the old machine with OOM or
ulimit errors, every single time it was attempted there -- **ran to
complete, clean completion with zero infrastructure failures**, the first
time that has ever happened for this workload. Whatever is still wrong
with the reacquisition logic (point 3 above) is now unambiguously a logic
problem, not a resource problem -- exactly the separation this migration
was for.

**Threshold calibration is whole-path, not per-instant** (see
`detect/thresholds.py` docstring). A per-time-bin `(1-alpha)` quantile
controls the exceedance probability at each instant, but "first sustained
crossing anywhere in `T_max`" is a repeated test over the bins — even a
correctly-calibrated per-instant threshold inflates the whole-trajectory
false-alarm rate to roughly `1-(1-alpha)^n_bins` by a union-bound argument.
Measured directly: ~12-14% against a target of ~1%, confirmed independent of
`n_bins`, confirmed by many held-out references firing at *identical* times
clustered at bin edges. Fix: reduce each leave-one-out curve to a single
number, its own worst deviation over the whole path, and calibrate the
`(1-alpha)` quantile on *that* distribution — the statistic GATE 1a actually
measures.

---

## 6. Known defects — fix these first

1. ~~`scenes/ramp_descent.xml` (formerly `occlusion_transit.xml`) uses a
   Y-up convention.~~ **FIXED.** Scene was built Z-up from the start. A
   related but distinct bug was found and fixed instead: the ramp's
   pre-rotation box coordinates were guessed rather than solved from the
   desired post-rotation world endpoints, which placed the ball's start
   position 4cm from the rotated ramp's true top edge. That knife-edge start
   made the reference ensemble trimodal (roll forward, roll backward off the
   ramp, or land balanced and barely move) purely from lambda-scale
   perturbation — not visible as a physics bug, only as inexplicable
   detector behavior downstream. Fixed by solving box center/size
   algebraically from the target world-space top/toe points and starting
   the ball ~1m inboard of the true edge. If a scene is ever rotated via
   `euler`, solve corner placement from the rotation, don't guess
   pre-rotation coordinates and eyeball it.
2. ~~`detect/statistics.py::sigma_interpenetration` does not exist.~~
   **FIXED (2026-08, M2.5).** L0-only ("P3-lite") version added — see M2.5
   for the real design iteration this took (a first z-scored-difference
   version broke the shared threshold-calibration machinery in a genuinely
   novel way; fixed by reframing as a ratio) and its own explicit scope:
   this is NOT the FULL, learned/video-side P3 §10 still defers.
3. ~~`mutants/library.py::vanish` has dead code~~ **FIXED.** Removed.
4. **`mutants/library.py::duplicate` changes `K`.** This is intentional (see
   §3.8) but any new detector must tolerate it.
5. ~~`physics/synthetic.py` has no horizontal friction~~ **FIXED** in both
   backends: `synthetic.py` has explicit `MU_FLAT` deceleration;
   `scenes/ramp_descent.xml` needed `condim="6"` on the ball geom before
   its rolling-friction coefficient did anything at all — MuJoCo's default
   `condim=3` silently ignores the torsional/rolling friction components.
6. **Environment: requires Python >= 3.10.** MuJoCo 3.5 has no cp39 wheel;
   pip falls back to a source build and fails on `MUJOCO_PATH`. Not
   triggered here — Python 3.12 / mujoco 3.11.0 already installed.
7. **`mutants/library.py::wrong_gravity` rescaled `pos[:, obj, 1]` (Y) instead
   of `pos[:, obj, 2]` (Z).** FIXED. In a Z-up world this mutated the
   near-stationary lateral axis, making the mutant almost a no-op — found
   only because it stayed silently undetected on the MuJoCo backend after
   every other fix.
8. **`detect/statistics.py::sigma_kinematic` combined all 3 axes into one
   isotropic distance/spread ratio.** FIXED — now normalizes each axis by
   its own spread before combining (diagonal-covariance Mahalanobis, not
   combined-then-normalized). Once the ball is rolling, x-spread across the
   reference ensemble reaches ~80x the z-spread; the old isotropic version
   let x swamp the ratio and hid any defect confined to z (e.g. defect #7).
   The per-axis spread floor also matters: use ~0.01 (1cm), not ~1e-6 — an
   axis genuinely collapses to ~0 variance once the ball comes to rest, and
   a micron-scale floor turns float noise into huge spurious sigma spikes
   that corrupt the whole-path threshold calibration (§5 note) globally,
   since one such spike among the M leave-one-out curves inflates the
   single calibrated multiplier applied to every time bin.
9. **Occlusion was originally combined with the ramp scene (one scene, wall
   partway down the incline).** Split out per 3.6 -- see §5. The wall in the
   old combined scene was also geometrically too thin (a marker line, not a
   volume) to ever produce "stops occluded, never re-emerges" as a distinct
   outcome; `occlusion_corridor.xml`'s wall has real length along the
   direction of travel specifically so that outcome is reachable.
10. **Isotropic 3D Sigma (`sample_perturbation`) is wrong for
    `occlusion_corridor`.** A ball resting on a flat floor has a unilateral
    contact constraint (can't penetrate, isn't pulled back down symmetrically),
    so perturbing z-position/z-velocity at the lambda needed for useful
    along-track spread (lambda=6) doesn't cancel out on average -- it
    produces an actual bounce (z-spread measured up to ~2.8, vs a resting
    ~0.15±0.01), which corrupts R5 detection with motion the scenario never
    meant to test. **FIXED** by adding `EpisodeSpec.perturb_mode` and
    `physics/perturb.py::sample_velocity_perturbation`, which perturbs
    position and velocity along one axis only. `ramp_descent` is unaffected
    (defaults to `perturb_mode="full"`). Any future 1-DoF-motion scenario
    should use `perturb_mode="velocity_x_only"` from the start rather than
    rediscovering this.
11. **`occlusion_corridor`'s ball motion has two physically distinct phases,
    not one constant deceleration.** The keyframe gives the ball pure
    translational velocity with zero spin, so it starts SLIDING (measured
    ~2.3-2.6 m/s^2, standard rigid-body result while angular velocity spins
    up) before transitioning to ROLLING-without-slipping (measured ~1.246
    m/s^2, CV=0.12% across M=30 references) once `v = omega*r`. A naive fit
    window starting at `t=0` blends both phases into a number that matches
    neither (clustered just as tightly, ~1.264, but wrong — an artifact of
    how much transient leaked into that particular window). **FIXED** by
    `detect/physical_constants.py::DEFAULT_CORRIDOR_WINDOW=(0.5s, 1.3s)`,
    chosen with margin on both sides of the transient and the earliest
    stop across the lambda=6.0 reference ensemble. Also caught in the same
    investigation: `synthetic_corridor.py`'s `MU_FLAT=0.182` (i.e.
    `mu*g=1.785`) is a coarser whole-trajectory-averaged estimate from
    earlier calibration work, not a clean ground truth for MuJoCo's
    rolling-friction model (which isn't a simple `mu*g` formula) — do not
    feed it to `constant_recovery_report`'s `true_value` argument, that
    would silently assert a false ground truth.
12. **Re-identification test/calibration scripts defined "reappeared" as
    "first frame with any visible pixel," not "first frame meaningfully
    visible."** On `occlusion_corridor`'s hard velocity band this picked
    frames where the object was a 1-6px sliver just peeking from behind
    the occluder — too small for SAM2's automatic mask generator or DINO
    to reliably segment/embed, independent of which re-id method is used.
    Likely a real contributing factor to DINO's earlier measured 0/N-match
    failure, not just the appearance-separability ceiling characterized
    separately. **FIXED** in the design that replaced it (§5,
    `phi/motion_prior.py`): "reappeared" now requires the mask area to
    reach >=50% of that episode's own pre-occlusion max area, not a fixed
    pixel count (per-episode-normalized since apparent object size depends
    on camera distance/scenario, same reasoning as defect #11's window
    choice). Any future occlusion-recovery test should use this
    definition from the start.
13. **`estimate_threshold` (`detect/thresholds.py`) could return NaN when a
    majority of the reference ensemble is simultaneously non-visible for a
    sustained stretch -- found while validating `phi/reconstruct.py` on
    `occlusion_corridor`, not a defect in `reconstruct.py` itself.**
    **FIXED**, in three parts, none of them a quick patch of the symptom:
    1. `sigma_kinematic` previously collapsed every non-finite candidate
       deviation to `inf`, conflating "genuinely gone" (`present=False`,
       correct to treat as maximal violation -- it's what makes the
       `vanish` mutant detectable, GATE 1b) with "temporarily occluded but
       still believed to exist" (`present=True`, pos unmeasured -- a live
       re-identification gap). The second case is now left as `NaN`
       ("undefined, no evidence"), not `inf` ("worst possible") -- gated on
       `traj.present`, so the `vanish`/`velocity_freeze` mutant tests
       (`tests/test_core.py`) are unaffected.
    2. `estimate_threshold`'s whole-path reduction now uses `np.nanmax`
       (ignore NaN, correctly keep inf) instead of forcing every
       non-finite ratio to inf, and `np.quantile(..., method="higher")`
       instead of the default linear interpolation, which computes
       `inf - inf = NaN` internally whenever the true `(1-alpha)` quantile
       position legitimately falls among tied `inf` values -- exactly what
       happens when `>alpha` fraction of references carry a genuine,
       permanent violation. `_bin_scale` was never at fault; it already
       filtered to finite values per bin correctly.
    3. Even NaN-free, a single global whole-path scalar `c` still meant
       one badly-behaved region made `theta(t) = c*scale(t) = inf`
       EVERYWHERE, including an early window where the statistic was
       perfectly well-behaved (measured directly: theta was inf at every
       time bin, not just the occluded tail, until this was addressed).
       Fixed by bounding calibration to a prefix (`_defined_cutoff`) where
       the reference ensemble stays sufficiently observed; `theta(t)=inf`
       for `t` past that point is now an honest "this statistic can't be
       calibrated here," not a global collapse. The cutoff fraction
       defaults to `1 - alpha`, not an arbitrary constant like 0.5 (tried
       first, measured to still leave theta inf everywhere at M=30,
       alpha=0.01 -- with that few references and that strict an alpha,
       the quantile is essentially the single worst of M values, so even
       one in-window dropout still poisons it; `1-alpha` keeps the two
       parameters self-consistent instead of independently guessed).
    Validated on `occlusion_corridor`, M=30, Phi-reconstructed references:
    `theta` finite and non-NaN through the pre-occlusion window
    (~frame 37), `inf` (not NaN) beyond it; LOO false-positive rate 1/30
    (order-of-magnitude consistent with alpha=0.01 at this small M).
    **Not covered by this fix, and not in its scope**: full R5-sensitivity
    re-validation via Phi specifically on `occlusion_corridor` -- that
    scenario's target property is P2 (existence), not P4, and its own demo
    mutant is `duplicate`, not a kinematic one (`visualize_reference_
    ensemble.py`'s own `DEMO_MUTANT` mapping already reflects this); R5's
    real calibration home is `ramp_descent`, which `reconstruct.py`
    explicitly does not support yet (non-planar motion, see its own
    module docstring). A same-instrument `duplicate`-mutant sensitivity
    check through Phi was attempted and blocked by a separate, already-
    documented limitation: `reconstruct.py` is single-object (K=1) only,
    same limitation `reidentify.py`'s threshold docstring already names.
    R2/existence sensitivity via Phi is already evidenced by the
    re-identification validation earlier in this project (10/10 correct
    reacquisition against real SAM2 candidates on the exact hard cases
    DINO failed), not re-derived here.
14. **`phi/reidentify.py::load_mask_generator` never exposed `pred_iou_
    thresh`/`stability_score_thresh`, so real, shipped usage always got
    SAM2's strict defaults (~0.88/0.95), not the looser `0.5/0.5` already
    known from earlier calibration work to be necessary for this small,
    low-texture ball.** Found via full end-to-end validation (build-
    sequence step 27), not inspection -- confirmed directly that with
    strict defaults SAM2's automatic mask generator proposed only 4
    candidates per frame near a true reappearance point, all large
    background regions, never anything near the ball. **FIXED** --
    defaults changed to `pred_iou_thresh=0.5, stability_score_thresh=0.5,
    min_mask_region_area=0`. Every caller that didn't explicitly override
    these two previously-unexposed parameters was silently exposed to
    this; it was a production bug, not a test-script issue.
15. **`motion_prior.score_candidates_by_position` accepted the nearest
    scored candidate unconditionally -- no distance-rejection criterion
    existed.** Combined with search firing almost immediately after a
    loss (`empty_streak > forgiveness_frames`), usually long before a real
    occlusion could plausibly resolve, this locked onto a static
    background distractor and tracked it for the rest of a clip
    (post-reacquisition IoU ~0.002, measured). The standalone validation
    that preceded this never caught it because it only ever tested at the
    known, correct reappearance frame. **FIXED, two layers**: a distance
    gate scaled to the object's own last-known apparent size
    (`mask_diagonal_px`), plus persistence-tracking (`rejected_positions`)
    so a static distractor that gradually drifts inside a naive
    size-based gate over many extrapolated frames is still caught. See
    §5's build-sequence-step-27 section for the full account, including
    the still-OPEN issue (#16) this fix's own side effect may have
    introduced.
16. **FIXED. Root-caused precisely (not the accumulation-blanket
    hypothesis originally suspected): the per-episode PIXEL-space quadratic
    fit (`PixelTrackFit`) is fundamentally unreliable when extrapolated far
    beyond its own fit window, because a short window of real, noisy
    SAM2-tracked pixels is a poor basis for re-deriving DECELERATION
    (a second-derivative quantity) -- confirmed directly: on a real
    ~23-frame gap, the per-episode fit implied the object stops at pixel
    x=107.9, when the true object was still at pixel x=68.3 by frame 61
    and still moving.** The clamp added earlier (this defect's first
    attempted fix) was necessary but insufficient -- correct given the
    fit's own coefficients, but the fit itself wasn't trustworthy that far
    out.

    **Actual fix**: `motion_prior.fit_metric_track`/`MetricTrackFit`
    (new) -- fits ONLY (position, velocity) per-episode, in METRIC 3D
    space via `reconstruct.unproject_to_plane` (well-conditioned over a
    short window; position/velocity are directly observable, unlike
    deceleration), then extrapolates using `detect/physical_constants.
    py`'s already-calibrated reference-ensemble deceleration
    (a=1.246 m/s^2, CV=0.12%) instead of re-deriving it, clamps at the
    implied stop time, and projects back to pixel space via the new
    `reconstruct.project_to_pixel` (exact inverse of `unproject_to_plane`,
    validated to ~1e-7). `track_with_reidentification` gained optional
    `camera`/`deceleration` parameters to opt into this path; the old
    pixel-only path remains as an explicit fallback for scenarios without
    camera calibration.

    **Validated against ground truth, not just internally consistent**:
    compared the new fit's predicted pixel trajectory against MuJoCo
    ground truth at every frame of a real occlusion gap (seed=2024) --
    within 0.9-1.9px throughout, matching the original standalone
    candidate-check's own accuracy standard (2.7-3.9px). The closed-form
    physics itself (given true x0/v0) matches ground truth to 1.8cm at
    0.8s of extrapolation. Two new tests
    (`tests/test_metric_track_fit_recovers_known_position_and_velocity`,
    `tests/test_metric_prediction_matches_analytic_stop_position_far_
    past_extrapolation`) check this against a synthetic camera + known
    trajectory, independent of any real scene.

    **New, distinct, separately-tracked finding surfaced while confirming
    this fix — see defect #17.** The prediction being this accurate is
    exactly what exposed it: an accurate trajectory can still walk
    straight into a coincidentally-positioned static feature sitting on
    its own (still-occluded) path.
17. **FIXED. Even with an accurate (validated against ground truth)
    physics-prior prediction, real end-to-end reacquisition on
    `occlusion_corridor` could still false-match a static, OCCLUDER-ADJACENT
    feature before the object is actually observable again.** Precisely
    characterized, not a guess: at frame 48 of a real gap, the object's
    true metric position (x=5.16m) was still within the wall's own
    occlusion span (x in [4.25, 5.75], from the scene's own geometry) --
    genuinely not yet visible -- yet a SAM2 candidate at that exact
    predicted pixel location (within the accurate physics gate) got
    accepted anyway. Almost certainly a real, static visual feature of the
    wall itself (an edge, shadow, or texture patch) sitting directly on
    the object's own straight-line path, precisely because the wall was
    placed to occlude that path. Distinct from defect #15's "random
    background distractor" case in cause, not just in symptom -- the
    persistence-tracking fix from #15 doesn't prevent this, since the
    candidate is genuinely close to an accurate, not a wrong, prediction.

    **Fix**: since the occluder's own footprint is known scene geometry
    (not privileged object state -- same status as knowing your own camera
    calibration, per this repo's existing position on that distinction),
    `MetricTrackFit.predict_pixel` now refuses to offer a search target at
    all (returns `None`) while the predicted metric position is still
    within a known `occluder_bounds` span, only resuming once the
    prediction itself indicates the object should have cleared it.
    `tests/test_metric_prediction_refuses_a_match_inside_known_occluder_
    bounds` covers this against a synthetic camera + known trajectory.

    **Validated on the real gap**: zero physics search attempts logged
    during frames 40-56 (correctly suppressed -- the accurate prediction
    knows it's still behind the wall), resuming at frame 57 with predictions
    accurate to ~3.7px against ground truth.

18. **FIXED. The DINO appearance-fallback tier (tier 2) had no equivalent
    of the physics tier's (#15) persistence/suspicion safeguard, so it
    could confidently accept a candidate the physics tier had just spent
    several frames actively rejecting as implausible.** Measured directly
    on the same real gap used for #16/#17: physics rejected a static
    candidate at (95.2, 118.6) four consecutive times (frames 58-61,
    `rejected_positions` correctly flagging it as suspect each time), then
    at frame 61 DINO accepted a match at similarity 0.4127 -- barely over
    the 0.4 threshold -- with zero awareness of physics's own rejections
    in the same gap. That match produced 0.0 IoU with the true object for
    the rest of the clip (`mean_iou_post_reacquisition`).

    First attempted fix (insufficient, kept only as a lesson): gate DINO's
    best candidate on proximity to the exact points in `rejected_positions`
    (a small, shared `STATIC_SUSPECT_RADIUS_PX`). Too narrow -- SAM2 splits
    one static distractor (a wall/doorframe feature) into several distinct
    sub-masks at different pixel offsets across the same frame, so DINO's
    highest-similarity candidate can legitimately be a *different* sub-mask
    of the *same* static thing. Confirmed directly: DINO's frame-61 best
    match sat at (94.9, 104.5), 14px from the nearest rejected point --
    outside the 5px check, so it evaded it entirely and was still accepted.

    **Actual fix**: don't remember exact rejected points at all -- reuse
    physics's own size-scaled spatial gate (`gate_px`, computed from the
    object's last-known `mask_diagonal_px`) directly against DINO's
    candidate, whenever physics has a valid `predicted_pos` this frame. A
    DINO candidate outside where the object could plausibly be is
    implausible regardless of appearance similarity -- DINO can now only
    override physics within physics's own notion of "plausibly still the
    object," not anywhere appearance alone finds convincing.

    This surfaced a second, narrower gap while validating against the real
    gap with determinism pinned (see #19): during frames 40-56, physics has
    no `predicted_pos` at all -- not because it can't fit, but because
    `predict_pixel` (via #17's fix) knows the extrapolated position is
    still inside `occluder_bounds` and correctly refuses to offer a target.
    That's a *stronger* signal than "no opinion," but the spatial-gate fix
    above only fires when `predicted_pos is not None`, so it was blind in
    exactly this window -- and with determinism pinned, DINO deterministically
    false-matched an unrelated static feature at frame 40 (similarity
    0.4048, 24 frames before the object could possibly be visible again at
    rise=64). Fixed with `MetricTrackFit.occluded_at(t)`, which exposes
    "definitely still occluded" as a distinct, positive signal (reusing the
    same extrapolation `predict_pixel` already computes, factored into a
    shared `_x_motion` helper) -- `track_with_reidentification` now skips
    tier 2 entirely for a search attempt physics already knows is
    premature, not just tier 1.

    **Validated end-to-end on the real gap, with determinism pinned (#19)
    so the result is reproducible, not a lucky run**: `n_reid_events` 27 ->
    11 (dead search attempts during the known-occluded window no longer
    logged at all); frame 61's DINO candidate correctly rejected
    (`dino_rejected_implausible`); clean reacquisition at frame 62 via the
    physics tier itself, 1.02px from the true position; **`mean_iou_post_
    reacquisition` 0.0 -> 0.79**. Confirmed bit-identical across three
    repeated Modal runs post-fix.

19. **FIXED. Discovered while validating #18: three back-to-back Modal
    runs of IDENTICAL code against IDENTICAL input frames produced TWO
    DIFFERENT reid traces.** One run's DINO similarity score for the same
    static distractor came back 0.4048 (over the 0.4 threshold -- false
    match) where two other runs of the same comparison gave 0.3998 (safely
    under). This is a direct violation of 3.1 (measurement symmetry): the
    same episode must not get a different verdict on different days, and
    at the time this was found, it could. Root cause: unpinned GPU
    inference non-determinism (cuDNN's default, non-"deterministic"
    algorithm selection, plus TF32's reduced mantissa) -- not a gradient
    issue (everything here runs under `torch.inference_mode()`), but
    forward-pass floating-point results can still vary run-to-run at fixed
    code and input, which is exactly the kind of jitter that flips a
    borderline similarity score across a fixed threshold.

    **Fix**: pin `torch.manual_seed(0)`,
    `torch.backends.cudnn.deterministic = True`,
    `torch.backends.cudnn.benchmark = False`, and disable TF32
    (`torch.backends.cuda.matmul.allow_tf32` /
    `torch.backends.cudnn.allow_tf32 = False`) at the top of
    `track_with_reidentification`, before any GPU op runs.

    **Validated**: three repeated Modal runs post-fix returned bit-identical
    results, including similarity scores matching to the full float32
    precision printed (e.g. `0.4047652781009674` exactly, three times).

    **Not yet audited elsewhere**: this pinning currently lives only in
    `track_with_reidentification`. Other GPU inference paths in this repo
    (plain `segmentation.track_video()`, `diagnose_candidates`) do not yet
    pin determinism and should be audited before being relied on for GATE
    2 or model comparisons in M6/M7, where a non-reproducible measurement
    would be a much more expensive problem to discover late.

20. **FIXED (real, but turned out NOT to be the dominant cause of GATE 2's
    null false-positive rate -- see §7 M5's writeup for the actual, confirmed
    mechanism): `reconstruct.py`'s `_existence_mask` required a `reid_event`
    to mark a gap as closed, with no exception for a gap SHORT ENOUGH that
    `reidentify.py`'s own `forgiveness_frames` tolerance never treated it as
    a loss worth searching for in the first place -- so a forgiven,
    never-searched-for, single-frame SAM2 flicker would have silently read
    as an unresolved existence violation to a threshold with zero tolerance
    for one.** Fixed: `_existence_mask` (and `reconstruct_trajectory`) now
    take `forgiveness_frames` and auto-close any gap at or under that
    length, matching what `reidentify.py` itself already decided wasn't
    worth searching for. `tests/test_reconstruct.py::
    test_forgiven_flicker_stays_present_without_a_reid_event` covers this
    directly.

    **Investigated, not assumed, whether this explained GATE 2's observed
    9/10 null false-positive rate**: re-ran the same 10 null instances
    against the fix and got IDENTICAL results to before it -- the actual
    cause (confirmed against each instance's own ground-truth trajectory,
    §7 M5) was a real, longer occlusion behind the wall that never resolved
    within the observation window, not a short flicker. Kept anyway: still
    correct, load-bearing behavior for any future sweep where a forgiven
    flicker *does* coincide with a threshold-sensitive window, and the fix
    cost is a few lines plus a test, not a design compromise. A useful
    reminder that confirming a hypothesis against real data, even after the
    fix is clearly correct in isolation, is not optional.

---

## 7. Build order

Each milestone has a definition of done. Do not start the next until the
current one passes. The ordering is chosen so that the cheapest possible
failure comes first.

### M0 — Environment and GATE 0 (day 1)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[physics]"
python -c "from vitals.physics.runner import gate_fork_determinism as g; \
           print(g('scenes/ramp_descent.xml')); print(g('scenes/occlusion_corridor.xml'))"
```

**Done when:** `gate_fork_determinism` returns `True`.

**If it returns False:** do not work around it. Try fixing determinism first
(single-threaded, fixed timestep, no `mjOPT` randomness). If it still fails,
**P5 is unmeasurable in this engine** — drop causal consistency from the scope,
record the decision, and continue with P1–P4 and P6. Do not fake a fork.

### M1 — Real physics behind the existing core (week 1)

Fix the axis convention (§6.1). Then:

```bash
VITALS_BACKEND=mujoco python scripts/run_l0_demo.py
```

**Done when:** the demo runs unchanged against MuJoCo and Gate 1a still passes.
If anything downstream of `physics/` needed editing, the `Trajectory`
abstraction is leaking — fix that rather than the caller.

**Also produce:** the lambda-spread curve. For lambda in {0.25, 0.5, 1, 2, 4},
plot RMS distance from the ensemble centroid vs. time. This tells you what
lambda_0 should be and is the first real experimental output.

### M2 — Mutant validation, state space (week 2)

Build the full mutant sweep: for each mutant kind × severity × injection time,
record whether it fired, on which risk, and with what event-time bias.

**Done when:**
- null false-termination rate ≈ alpha (within a factor of ~3)
- every non-null mutant fires on its **expected** risk, not a neighbour
- event-time bias is reported per mutant with a distribution, not a single number
- ROC per risk over severity, and a stated minimum detectable defect

**This is Gate 1.** If mutants fire on the wrong risk, the detectors are not
separable and the design needs revisiting before any pixels are involved.

### M2.5 — P3-lite (interpenetration) + a high-friction P4 scenario (2026-08)

Property coverage was P2+P4 only; the score-report work (M6/M7) surfaced this
as the next real gap. §10 scopes FULL P1 (a static-keypoint/camera-pose
tracker, unbuilt) and FULL P3 (a learned contact/support relation head,
video-side) as open research, not implementation, with a sanctioned
"report as an unmeasured gap" fallback. Scoped v1, chosen directly over
attempting either in full: an L0-only (state-space) `sigma_interpenetration`
(R4), calibrated with the exact same rigor P2/P4 already had at L0 before
either ever got an L1 pass — activating `mutants/library.py::teleport`,
which has declared `risk_expected="R4"` since it was first written but had
no statistic to score against (defect #2 below, now resolved).

**Key finding that shaped the whole approach:** `teleport(traj, t_star,
obj=0, target=1)` just overwrites `pos[i:, obj]` with `pos[i:, target]` — a
post-hoc trajectory corruption exactly like `vanish`/`duplicate`/`jitter`,
needing a scene with K≥2 real objects already present, NOT real collision
physics. `scenes/occlusion_corridor_distractor.xml` already has exactly
that (ball + decoy, both present the whole clip, never colliding with each
other in ground truth) — so no new scene file was needed for P3-lite, only
a new manifest (`configs/manifests/occlusion_corridor_interpenetration.
yaml`, `target_property: P3`) pre-registering the SAME scene as its own
isolated experiment (3.6/3.12 — one manifest, one property, decided in
advance; the sibling `occlusion_corridor_distractor.yaml`, P2, is
untouched).

**`sigma_interpenetration`, real design iteration, not a first-try success:**
1. First version: `(ref_mean - cand_dist) / ref_std`, clipped to `max(...,
   0)` — a direct pairwise sibling of `sigma_kinematic`'s own centroid/
   spread pattern. Looked right, calibrated to an absurd threshold
   (~15,800, against a real defect's own natural value of ~1) once M was
   large enough to expose it. Root cause, found by direct inspection, not
   guessed: this z-scored difference is ~0 for a "normal" LOO reference
   BY CONSTRUCTION, and the explicit clip makes it EXACTLY 0 for roughly
   half of any given time bin's M references (whenever that reference's
   own separation happens to be above the mean) — `thresholds.py::
   _bin_scale`'s per-bin MEDIAN then sits right at that 50/50 boundary and
   floors to the shared `1e-6` "no signal" fallback unpredictably often,
   and whenever it does, some OTHER reference's own modest, real deviation
   at that same bin gets divided by ~1e-6 and blows the whole-path
   calibrated threshold up by 5 orders of magnitude. R2/R5 never hit this
   despite `_bin_scale`'s own comment acknowledging "R2 is exactly 0
   almost everywhere" — because their OWN near-zero bins are LITERALLY
   always exactly zero in a clean reference ensemble (object presence
   count doesn't fluctuate by chance; a 3-axis Mahalanobis NORM is
   essentially never exactly 0), never a 50/50 split for the median to
   land on unpredictably. This is a real, novel finding about the shared
   calibration machinery's implicit assumptions, not a bug in
   `thresholds.py` itself — deliberately NOT touched, to avoid any risk to
   R2/R5's own already-verified calibration.
2. Fixed by reframing as a RATIO, not a difference: `ref_mean(t) /
   max(cand_dist(t), 0.01)`. ~1.0 when normal (a stable, non-degenerate
   median — exactly the well-behaved scale `_bin_scale` was already built
   to handle), grows large as the two objects get closer than typical
   (interpenetration), stays small when they drift farther apart (one-
   sided in EFFECT — a small ratio can't cross a threshold calibrated
   near/above 1.0 — not by an explicit clip). Threshold dropped from
   ~15,800 to a sane ~6.3 immediately.
3. `STATS` gating (`scripts/run_l0_demo.py`) is on `target_property ==
   "P3"`, NOT on `K >= 2` — a real bug, found by this change's own
   verification pass (every existing manifest re-run before/after): K≥2
   gating silently added R4 to `occlusion_corridor_distractor.yaml` too
   (same K=2 scene, but pre-registered P2), which changed ITS GATE 1b
   outcome (`velocity_freeze` fired R4 instead of R5, since freezing the
   ball's own velocity incidentally changes its distance to the untouched
   decoy, and `PRECEDENCE` ranks R4 ahead of R5) and pushed its GATE 1a
   false-termination rate from 0.05 to 0.067. The SAME cross-talk, same
   root cause, was independently found a second time in `build_demo_cases`
   and the 80-episode survival-population mix (both inherited the generic
   R2/R5 mutant set for the new P3 manifest at first) — fixed the same
   way: a P3 manifest gets its OWN case list (`null` + `teleport` only)
   and its OWN 2-way population mix (50/50 `null`/`teleport`), not the
   4-way R2/R5 set. All three fixes are the identical instance of 3.6/
   3.12 applied consistently, not three unrelated bugs.
4. `n_reference: 100` for this manifest only (every other manifest keeps
   30) — GATE 1a's false-termination rate is a union across every active
   risk channel, and at M=30, `estimate_threshold`'s own whole-path
   calibration was already selecting the single MAXIMUM LOO deviation out
   of only 30 references (confirmed directly: threshold identical across
   alpha=0.01 down to alpha=0.001), too small a sample for a 3rd channel
   to share the false-alarm budget reliably. Swept M directly rather than
   guessed: 30→fp=0.067 (fail), 60→0.033, 100→0.000.

Final state, `occlusion_corridor_interpenetration`: GATE 1a fp=0.000,
GATE 1b clean (`null` and `teleport` both OK), n=80 survival population
100% R4 terminations (by construction — the manifest's own 50/50 null/
teleport mix, not yet a mixed R2/R4/R5 population; broadening that mix is
a natural follow-up, not done here).

**Known, scoped gap, stated plainly (2026-08): P3-lite's video (L1) path
is blocked, its state-space (L0) path is not.** GATE 2 and real-model
scoring both go through `vitals/adapters/video_model.py::default_
phi_reconstruct` (called by `run_gate2.py` and by `run_model_population.
py`'s own `make_modal_phi_fn`), which is single-object only — it calls
`reidentify.track_with_reidentification` once and `reconstruct.
reconstruct_trajectory` once, never `reconstruct.merge_trajectories`.
`merge_trajectories` itself is real and already exists (M5.6, "core
done") — reconstruct each tracked object independently through the
existing single-object path, then merge — but it's only ever been
exercised by one standalone validation function (`remote/modal_app.py`'s
own primary+secondary test), not wired into the actual PRODUCTION
reconstruction entrypoint every other script goes through. Checked
directly before concluding this (the `run_gate2.py` module docstring's own
"K>1 needs reconstruct.py support it doesn't have" claim turned out to be
STALE for `ramp_descent`, so it was re-verified here rather than trusted
a second time) — confirmed still accurate for the K>1 case specifically.
Wiring K=2 support into `default_phi_reconstruct` (and the `phi_fn`
contract every caller of it depends on — a second `frame0_mask`, at
minimum) is real, comparable-sized scope to M5.6 itself, not a quick
add-on — deliberately deferred as its own future task rather than done
as a side effect of this one. `occlusion_corridor_interpenetration`'s own
L0 detector (GATE 1a/1b above) is real and calibrated; only GATE 2 and
real-model scoring on THIS scenario are blocked, not the whole property.

**High-friction P4 scenario, requested directly alongside P3-lite
("variations that incorporate friction and other more complex physics"):**
`scenes/ramp_descent_high_friction.xml` + `configs/manifests/
ramp_descent_high_friction.yaml` — identical solved geometry to
`ramp_descent.xml` (not re-derived), only the default friction triple
changed (`0.6 0.005 0.03` → `1.2 0.01 0.15`, ~2x/2x/5x), so the ball
visibly decelerates to a stop on the flat section within the clip instead
of coasting at near-constant velocity — a qualitatively different
kinematic regime (decaying, not near-constant velocity), not just new
geometry on the same regime. `condim="6"` (defect #5's own lesson) already
carried over unchanged.

GATE 1b's `velocity_freeze` case (t*=1.5) stayed censored at the scene's
inherited `lam=1.0` (ramp_descent's own default) — investigated directly
(not assumed): at t=1.5 the ball is still genuinely moving (~0.17 m/s,
still on the ramp), so this wasn't a "mutant fires on an already-stopped
object" no-op. Root cause, confirmed by sweeping `lam` alone and nothing
else (this project's own documented method for exactly this symptom, first
used for `projectile`/M4.5): higher friction makes the reference
ensemble's own exact deceleration/settling behavior MORE sensitive to
`lam=1.0`'s perturbation than the base scene's (small differences in slip/
contact timing compound differently under much higher friction),
inflating R5's threshold enough to swallow the mutant. `lam: 0.25` (chosen
over a more aggressive 0.1 that also worked — less tightening of the
reference ensemble for the same result) fires cleanly. Final state: GATE
1a fp=0.000, GATE 1b all OK; the n=80 survival population's own VI₅₀ comes
back `inf` (censoring=0.62, over half the population never terminates
within the clip) — an honest, already-precedented outcome for a
population that's mostly not defective by design (`bootstrap_vi`'s own
non-finite-draw handling), not a bug.

**Tests:** `tests/test_core.py` gained `_two_obj_traj`/`_interpen_refs`
(hand-built K=2 fixtures — the synthetic backend is K=1-only, so no
existing helper covered this) and three tests: planted overlap crosses
far above the ratio's own ~1.0 typical value, NaN when either object is
absent (an existence failure is R2's job, never manufactured into a second
R4 finding), and the one-sidedness (farther-than-typical separation stays
well below 1.0, never registers). `python3 -m pytest tests/ -q` — 64
passed (61 before this + 3 new), including every pre-existing manifest
re-verified byte-identical (GATE 1a/1b unchanged) before and after every
change in this section.

### M2.6 — GATE 2 generalized to a second scenario; a real, pre-existing bug found doing it (2026-08)

`scripts/run_gate2.py` was hardcoded to `occlusion_corridor.yaml` only.
Generalizing it for `ramp_descent_high_friction` (`VITALS_MANIFEST` env
var, matching `run_l0_demo.py`'s own convention) surfaced that the remote
`run_gate2_episode` (`remote/modal_app.py`) had its OWN separate copy of
the track+reconstruct logic, hardcoded to `plane_z`/occlusion_corridor-
only constants — structurally incompatible with `ramp_descent_high_
friction`'s `plane_pieces` mode. Fixed by extracting `vitals.adapters.
video_model.track_and_reconstruct` (scene-mode-aware, shared by both
`default_phi_reconstruct` and the now-unified `run_gate2_episode`) rather
than reimplementing the same branching a second time — one implementation
now, not two that can drift apart again. `meta.json` (`run_gate2.py`'s own
per-episode render output) carries `scenario_name` now, not `plane_z`
directly, so the remote side looks up its own config the same way every
other Phi caller already does.

**Real, pre-existing bug found DOING this unification, not introduced by
it:** `OCCLUSION_CORRIDOR_DECELERATION = 1.246` (the calibrated
reference-ensemble deceleration, `detect/physical_constants.py::
fit_corridor_deceleration`, CV=0.12%, AGENT.md defect #16 — consumed by
`motion_prior.fit_metric_track`'s physics-prior re-identification) lived
ONLY as a local constant in `remote/modal_app.py`, hardcoded directly into
the OLD `run_gate2_episode` — never migrated into `scene_geometry.py`'s
own canonical per-scenario config (`OCCLUSION_CORRIDOR_WALL_BOUNDS`'s own
sibling constant, right next to it, WAS already there). Confirmed via
directly re-running GATE 2 before/after: unifying onto `scene_geometry.
get(scenario_name).get("deceleration")` (which returned `None` for
`occlusion_corridor`, since that key didn't exist yet) measurably
regressed `velocity_freeze`'s own L1 correct-risk from 5/8 to 2/8 — real,
not cosmetic, since this is exactly the mutant type (a decelerating-motion
defect) most sensitive to losing a decelerating-motion re-identification
prior.

**This means `default_phi_reconstruct` — the SAME function every real-
model score on `occlusion_corridor` already goes through — has been
silently running WITHOUT this prior the entire time GATE 2's own separate
copy quietly compensated for it**, un-noticed until GATE 2 was unified
onto the shared path and its own accuracy visibly moved. Fixed at the
root: `deceleration=OCCLUSION_CORRIDOR_DECELERATION` added to `scene_
geometry.py`'s `occlusion_corridor`/`_distractor`/`_moving` SCENES
entries (all three share the identical ball-on-flat-floor physics), and
`remote/modal_app.py`'s own now-redundant local copy removed in favor of
importing the one true value — the exact same consolidation this module's
own docstring already documents doing once for `OCCLUSION_CORRIDOR_
WALL_BOUNDS`, needed a second time for a different constant that slipped
through the first pass.

**Re-verified against the most current previously-documented GATE 2
baseline (this section's own M5 writeup, not the stale `results/gate2_
instances_occlusion_corridor_mujoco.json` snapshot on disk, which turned
out to predate an even earlier fix and was quietly out of date) after
redeploying the fix: EXACT match across all four mutant groups** — null
L1 fired=6/10 correct=4/10; vanish L1 fired=6/8 correct=6/8, bias
-0.083s±0.186s; velocity_freeze L1 fired=6/8 correct=5/8; jitter
unaffected 8/8/8/8, bias +0.000s±0.000s. The stale results file has been
replaced with this freshly re-verified run (old copy kept at `results/
archive/gate2_instances_occlusion_corridor_mujoco.STALE_2026-08-07.json`
for the record, not deleted).

**Consequence, flagged directly (not silently absorbed): every real-model
score on `occlusion_corridor` computed before this fix — Cosmos n=80,
Wan n=2, both already published in `results/score_report.pdf` and this
document's own M6 writeup — was measured through Phi reconstruction
missing this same calibrated prior.** Old result files preserved at
`results/archive/l0_demo_occlusion_corridor_{cosmos,wan}.
PRE_DECELERATION_FIX.json` for a direct before/after comparison.

**Re-run, real delta measured, not assumed:** Wan (n=2) — byte-identical
event times before/after (both episodes fail too early, t≈1.0-1.6s, for
the prior to matter). Cosmos (n=80) — VI₅₀ unchanged (1.40s, same 95% CI
[1.33, 1.43]), median unchanged; ONE of 80 episodes shifted from firing
R2 (a likely re-identification hiccup during the occlusion gap, now
resolved with the correct prior) to firing R5/staying uncensored,
changing the termination profile from `{R2:0.0125, R5:0.9875}` to
`{R5:1.0}`. **The published Cosmos finding itself (worse than both
baselines, near-immediate failure) is confirmed robust to this fix, not
invalidated by it** — the missing prior mattered for GATE 2's own
mutant-specific check and one borderline episode's risk classification,
not for the headline result.

**A second, deeper, GENUINELY NEW finding, found investigating the first
GATE 2 pass on `ramp_descent_high_friction` (also this section's own
work): `ramp_descent`-family scenes have a real, previously-undiscovered
plane-based-reconstruction gap, exposed only because GATE 2 had literally
never been run on this scene family before (not `ramp_descent` either --
no `gate2_instances_ramp_descent*.json` file has ever existed) --
confirmed by directly PROVING it with zero video/segmentation involved:
ground-truth position -> `project_to_pixel` -> `unproject_to_known_plane`
round-trip, no SAM2 in the loop at all.**

- `ramp_descent.xml`'s ball starts at `pos="-3.3 0 1.7"`, DROPPED onto the
  ramp from height for "a small, consistent settling fall" (that file's
  own comment) -- genuinely airborne, on NO registered plane, for the
  first ~5-9 frames (~0.17-0.3s) of every single episode, before settling
  onto the ramp's own idealized surface. Measured directly: perpendicular
  distance from the ball's true position to the ramp's own ball-center
  plane is 0.23m at frame 0, decaying to <0.0001m by frame 10 -- expected,
  real physics, not a bug.
- Forcing a plane-based reconstruction (`unproject_to_known_plane`) onto a
  point that ISN'T on that plane yet produces a large, real geometric
  error during exactly that window -- 0.90m at frame 0, decaying to
  <0.001m by frame 10, measured on the SAME round-trip test. `reconstruct_
  trajectory`'s `plane_z`/`plane_pieces` modes have no representation for
  "briefly airborne, on no known plane" -- this is the FIRST scenario in
  the project with a genuine drop phase before its known-plane assumption
  becomes valid (`occlusion_corridor`'s ball starts already resting on the
  floor; `projectile`'s own `ballistic` mode has no plane assumption to
  violate in the first place).
- This transient is long enough (comparable to `extract_event`'s own
  `pi=0.3s` sustained-crossing requirement) to spuriously fire GATE 2's
  own R5 detector on EVERY episode regardless of whether a real defect
  exists -- confirmed directly: `null` (no defect at all) fired 10/10 at
  L1, always at t=0.00, when it should never fire at all (L0: 0/10,
  correct). This is why GATE 2's own first pass on `ramp_descent_high_
  friction` is NOT currently usable/trustworthy as published -- a real,
  unresolved gap, not a code bug to silently patch over; a scope decision
  (truncate the scored window past the settling frames? add an airborne
  reconstruction mode? document as an unmeasured gap the way P1/full-P3
  already are?) is needed before it can be, deliberately not made
  unilaterally here.
- **Explicitly does NOT contaminate real-model scoring (Cosmos's own
  n=5 result on this scenario, reported above in M6), confirmed by
  reading the actual call path, not assumed:** `VideoWorldModel.predict()`
  reconstructs the full prefix+continuation sequence via Phi, but
  `_slice_continuation()` discards the reconstructed PREFIX entirely, and
  `run_model_population.py`'s own `concat_trajectory(conditioning,
  continuation_traj)` scores the TRUE, ground-truth conditioning
  trajectory for frames 0-29 (1.0s prefix) -- never the Phi-reconstructed
  one. Only the CONTINUATION (frame 30 onward, well past the ~9-frame
  settling window) is ever scored from Phi's own reconstruction, and
  that portion is the part measured above to be near-perfect. Cosmos's
  own "100% immediate failure at generation handoff" finding on this
  scenario stands as reported, not retracted.

**A third, separate, pre-existing bug found in the same sweep -- fixed
2026-08, dedicated investigation.** `gate2_jitter_5` (`ramp_descent_high_
friction`) crashed -- `reidentify.py::track_with_reidentification`,
`IndexError: list index out of range` at `frame_paths[frame_idx - 1]`
(the retroactive-re-embedding logic, `reidentify.py:451`) -- reproduced
deterministically by re-calling the same episode directly. Not caused by
this section's own refactor (`track_with_reidentification` itself
untouched; only which config its CALLERS pass changed).

**The originally-hypothesized mechanism turned out to be wrong, checked
directly rather than assumed true:** the guess above was that `frame_idx
- 1` goes negative when the very first frame (`frame_idx=0`) already has
an empty mask. Traced through `_recent_visible_run`'s own sibling logic:
`prev_mask = result.masks.get(prev_idx)` uses `dict.get`, which returns
`None` for a negative key rather than wrapping around or raising --
`frame_idx=0` was ALREADY safe, confirmed by reading the code, not by
assumption. The real gap is the OTHER end, which this same line has no
guard for at all (unlike the `next_idx` search-side access a few lines
down, which already checks `frame_idx + 1 < T`): `frame_paths[prev_idx]`
has no UPPER-bound check, so any `prev_idx` that is in-range for
`result.masks` (a dict, keyed by whatever `frame_idx` values SAM2 has
actually yielded so far) but somehow falls outside `frame_paths`' own
length would crash -- confirmed directly by constructing exactly that
input locally (`masks={3: <mask>}`, `frame_paths` of length 3,
`frame_idx=4`) and reproducing the identical `IndexError: list index out
of range` with the OLD code before trusting a fix. The precise live-SAM2
trigger for how `result.masks` and `frame_paths` could disagree in
length was never fully pinned down (needs a real GPU repro this dev
machine can't run) -- left honestly unresolved -- but the fix doesn't
need to know why to be correct: ANY out-of-range `prev_idx` is a
"nothing to re-embed yet" case, not a crash, regardless of mechanism.

**Fixed by extracting `_maybe_recache_embedding(frame_idx, masks,
frame_paths, dino_model, device)`** (`vitals/phi/reidentify.py`, right
before `track_with_reidentification`) -- same logic, now bounds-checked
on BOTH ends (`0 <= prev_idx < len(frame_paths)`) before ever indexing,
and pulled out specifically so it's unit-testable without mocking SAM2's
entire propagation protocol (which needs a real GPU, unavailable here).
5 new unit tests (`tests/test_reidentify.py`) -- including one that
first proves the OLD inline logic really does crash on the constructed
input above, before asserting the NEW function returns `None` on the
identical input, so the regression test is anchored to a confirmed real
crash rather than a hypothesized one. Full suite 103/103.

### M2.7 — The airborne-reconstruction gap, actually fixed; a second, deeper gap found and fixed along the way (2026-08)

M2.6 left `ramp_descent_high_friction`'s GATE 2 pass unusable (`null`
firing 10/10 at L1 when it should never fire at all). Requested directly:
"fix the ramp_descent_high_friction airborne reconstruction gap." Two
genuinely separate root causes, found and fixed one at a time -- fixing
the first one didn't resolve the symptom, it revealed the second one
underneath.

**Root cause 1: the ball is genuinely airborne for its first ~9 frames,
and plane-based reconstruction has no representation for that.**
`ramp_descent.xml`'s ball is DROPPED from height ("a small, consistent
settling fall," that file's own comment) -- on no registered plane at
all for ~0.2-0.3s before landing. Proved with zero video/segmentation
involved: ground truth -> `project_to_pixel` -> `unproject_to_known_
plane` round-trip showed 0.90m error at frame 0, decaying to <0.001m by
frame 10 -- real geometry, not noise. Fixed with `reconstruct_
trajectory`'s new `unreliable_prefix_frames` parameter (`vitals/phi/
reconstruct.py`): forces `pos` to NaN for a known, fixed prefix window,
unconditionally, after the normal reconstruction loop -- `present` stays
True (the object IS visible), only its position is honestly marked
unmeasured. Wired via `scene_geometry.py`'s new `RAMP_SETTLING_FRAMES`
constant (12 frames, 0.4s -- analytical free-fall time from the scene's
own known drop height matches the empirical measurement almost exactly:
sqrt(2*(1.7-1.28)/9.81)=0.293s=8.78 frames), applied to `ramp_descent`
and `ramp_descent_high_friction` only -- strict no-op (default 0) for
every other scenario, confirmed by direct config inspection.

A per-frame size-consistency check (reject a plane candidate whose
implied depth disagrees with the object's own known-radius/apparent-size
depth cue, `fit_ballistic_trajectory`'s own established signal) was tried
FIRST and abandoned -- measured to carry its own ~1.0-1.3m systematic
bias on real rendered data even at ALREADY-settled frames, comparable in
magnitude to the ~1-2m of EXTRA discrepancy the airborne window itself
produces. Too weak a signal-to-noise ratio for a hard per-frame reject
test; a known, fixed frame count is strictly more reliable for a
deterministic, scene-governed phenomenon than a noisy per-frame heuristic
re-detecting it from scratch every episode.

`tests/test_reconstruct.py::test_ramp_descent_reconstruction_error_is_
bounded` previously ABSORBED this exact transient into a loosened 0.15m
mean-error tolerance rather than fixing it (its own comment already
documented the symptom -- "seed=1's own first ~7 frames show error
decaying from 1.08m down to the normal ~0.11m floor" -- but the fix
applied at the time was loosening the assertion, not addressing the
cause). Updated to assert the real fix instead: the settling window must
be NaN (not just diluted into an acceptable average), tolerance tightened
back to 0.10m now that it's honestly warranted.

**Root cause 2, found only after fixing the first one (GATE 2's own
`null` STILL fired 10/10 post-fix, now at a suspiciously exact t=0.40s --
right where the NaN prefix ends): this scene's true lateral (Y) motion is
essentially zero for every reference, and `sigma_kinematic`'s per-axis
normalization makes that a real problem specifically for L1 candidates.**
Traced the actual sigma values (not guessed): the reference ensemble's
own natural Y-variance is tiny (motion is confined to the ramp's own X-Z
plane by scene design -- no lateral dynamics exist in this scenario at
all), so Phi's own ALREADY-ACCEPTED ~0.10-0.16m reconstruction noise
(matching this scenario's own already-published ~0.11-0.13m error floor,
confirmed harmless on X/Z where real physics variance is much larger)
becomes enormous in units of Y's own near-zero natural variance --
enough to spuriously cross a threshold that was calibrated assuming only
real physics variance would ever appear on that axis, never instrument
noise.

**This was a materially bigger problem than "GATE 2 doesn't pass yet":
the offset persists well past the settling window (measured through
frame 100+), meaning it also contaminates the CONTINUATION region real-
model scoring actually reads from.** Directly means the earlier Cosmos
n=5 "100% immediate failure at generation handoff" finding on this
scenario (M6) could not be trusted as evidence about Cosmos specifically
-- indistinguishable from the instrument firing on anything, model output
or real physics alike, until this was fixed. Explicitly walked back
rather than left standing once this was understood.

**Fix:** `sigma_kinematic` (`vitals/detect/statistics.py`) gained an
optional `axes` parameter -- restrict the per-axis normalize-then-combine
to only the given axis indices, `None` (default) using all 3, identical
to every existing call site's behavior. `SCENARIO_KINEMATIC_AXES = {
"ramp_descent": (0, 2), "ramp_descent_high_friction": (0, 2)}` plus
`kinematic_axes_for(scenario_name)` live in `statistics.py` itself (not
`scene_geometry.py` -- this is a DETECTION-semantics concept, "which axes
are physically meaningful for THIS scenario's own kinematics," not a
reconstruction/camera one, and L0/GATE 1 needed it as much as L1/GATE 2
does for the SAME theta to stay valid across both channels). Bound ONCE
via `functools.partial(sigma_kinematic, axes=kinematic_axes_for(SPEC.
name))` in `STATS`'s own construction, identically in `run_l0_demo.py`,
`run_gate2.py`, and `run_model_population.py` -- deliberately the SAME
construction in all three, not independently chosen per script, since
theta is computed ONCE in state space and applied unchanged to every L0
AND L1 candidate; a mismatched axes restriction between scripts would
silently compare incompatible statistics.

Excluding Y isn't lowering a bar to make a defect-free population pass
(AGENT.md 3.3/4) -- Y was never a physically meaningful axis for this
scenario's own defects to begin with; every mutant this scenario's own
GATE 1b/GATE 2 sweep already exercises (`velocity_freeze`, `wrong_
gravity`, `jitter`) is fully expressed in X/Z, none of them rely on Y at
all.

**Both fixes verified together, real GPU runs, not assumed from the
math alone:** `occlusion_corridor`'s own GATE 2 re-run twice (once per
fix) came back byte-identical to the published baseline both times
(`kinematic_axes_for` returns `None` there -- confirmed strict no-op).
`ramp_descent_high_friction`'s own GATE 2, final state: `null` L1
fired=0/10 correct=10/10 (was 10/10 firing, 0/10 correct) --
`velocity_freeze`/`wrong_gravity` both L1 fired=8/8 correct=8/8 with
TIGHT, plausible bias (-0.046s±0.044s and +0.221s±0.016s respectively);
`vanish` 8/8 fired, 7/8 correct; `jitter` 7/8 fired, 7/8 correct. A
genuinely usable, trustworthy GATE 2 pass on this scenario for the first
time.

Cosmos's own real-model population on this scenario was re-run with the
fully fixed pipeline (`STATS` construction now includes the axes fix in
`run_model_population.py` too) -- old (potentially-contaminated) result
preserved at `results/archive/l0_demo_ramp_descent_high_friction_cosmos.
PRE_AXES_FIX.json` for a direct before/after comparison, not silently
overwritten.

**The re-run came back IDENTICAL to the old one: 5/5 episodes still fail
at exactly t=1.0s, the earliest possible instant, right at generation
handoff.** This is the good outcome, not a null result -- it CONFIRMS the
"100% immediate failure" finding was never actually the Y-axis artifact
in the first place: Cosmos's own generated continuation apparently
diverges badly enough on X/Z alone (a real, large kinematic error, not a
Y-only one) that removing the spurious Y contribution doesn't change the
verdict at all. The earlier walked-back claim is restored, now on solid
ground -- verified with the corrected instrument, not merely consistent
with a broken one.

**Scaled to statistical power immediately after (2026-08): Cosmos n=80,
Wan n=5 (first-ever Wan population on this scenario).** Cosmos: VI₅₀=1.0s,
95% CI **[1.0, 1.0]** (zero-width -- genuinely zero variance, not a
rounding artifact), censoring=0%, 100% R5, 80/80 episodes firing at the
identical earliest-possible instant. As strong and unambiguous a real-
model finding as this project has produced -- Cosmos's own generated
continuation on this scenario is deterministically wrong from its very
first frame, every time, across 80 independent episodes. Wan: n=5
(illustrative, below the n=10 significance floor), VI₅₀=1.07s, CI
[1.03, 1.07] -- same qualitative failure (fails almost immediately at
handoff), slightly more per-episode variance than Cosmos's exact zero.

**Known, not-yet-filled gap in the score report for this scenario:** no
`ConstantVelocity`/`CopyLastState` baseline population exists yet for
`ramp_descent_high_friction` (only `mujoco`/`cosmos`/`wan`), so `results/
score_report.pdf`'s own model cards for this scenario show VI₅₀ alone,
with no `Δ vs. baseline` comparison cards -- confirmed to degrade
gracefully (`render_score_report.py`'s own empty-baselines-list handling,
not a crash), but genuinely incomplete until those two populations are
run here too, the natural next step.

**UPDATE (2026-08): that baseline gap is now filled.** `scripts/run_eval.
py` (GATE 3 -- the script that runs `ConstantVelocity`/`CopyLastState`)
turned out to be a FOURTH place building `STATS` with `sigma_kinematic`,
missed in M2.7's own audit of `run_l0_demo.py`/`run_gate2.py`/`run_model_
population.py` -- would have silently scored baselines against a
DIFFERENT theta (all 3 axes) than Cosmos/Wan's own results (X/Z only),
an apples-to-oranges comparison in the score report. Fixed identically
(rebind `STATS["R5"]` via `kinematic_axes_for(spec.name)` once the
manifest is known, inside `main()`, matching the other three scripts'
own construction exactly). `ConstantVelocity`/`CopyLastState` now both
run on `ramp_descent_high_friction` (n=40 each, real MuJoCo backend):
VI₅₀=2.40s and 1.43s respectively -- same directional pattern as
occlusion_corridor's own baselines (ConstantVelocity > CopyLastState).
`results/score_report.pdf`'s own Cosmos/Wan cards for this scenario now
show real `Δ vs. baseline` comparisons: Cosmos vs. ConstantVelocity
-1.40s (CI [-1.60,-1.27], significant), vs. CopyLastState -0.43s (CI
[-0.57,-0.37], significant) -- both confirming what the raw VI₅₀=1.0s
number already suggested, now with a real floor to compare against.

**A real mistake made and fixed in the same turn, stated plainly, not
quietly corrected:** a "quick sanity check" run of the newly-fixed
`run_eval.py` (`--backend synthetic --n-episodes 10`, meant as a fast
local smoke test) was pointed at the SAME output path the real,
already-published `occlusion_corridor`/`ramp_descent` baselines live at,
overwriting `results/l0_demo_occlusion_corridor_constant_velocity.json`
(n=40, real MuJoCo -- referenced throughout this project's own score
report and M6 writeup) with throwaway n=10 synthetic-backend data.
Caught immediately, not silently left -- both files regenerated with the
correct parameters (n=40, `--backend mujoco`) and confirmed to match the
original published value exactly (occlusion_corridor's own VI₅₀=2.33s).
`copy_last_state` for both scenarios was untouched (never re-run in the
mistaken invocation). No archive backup existed for these two files
specifically (unlike the deceleration/axes-fix backups elsewhere in this
document) -- a real gap in this session's own backup discipline, worth
noting: back up BEFORE any write to a live results path, not just before
larger, more obviously risky pipeline changes.

`render_annotated_comparison_video.py` (the border-video feature) had
its own dead argument found and removed the same way: `--prefix-path`
was `required=True` but never once read in the function body -- the
prefix is always re-rendered locally (`renderer.render(conditioning,
...)`, cheap, needs no cached file) a few lines below where the unused
arg was declared. Removed; only `--continuation-path` (the one real,
paid-for model call) is genuininely required now. First real border
video generated and verified directly (not just unit-tested in
isolation): `results/videos/occlusion_corridor_wan_seed1_annotated_
with_border.mp4`, reusing the already-cached raw Wan continuation (zero
new GPU cost) -- confirmed frame-by-frame that the border transitions
exactly at the prefix/continuation boundary (frame 29: green `[0,219,0]`,
frame 30: red `[228,39,39]`, held through frame 89).

### M2.8 — Video provenance gap closed; occlusion_corridor's wall restyled with a "lit opening" (2026-08)

Real gap found directly ("why are the videos only getting saved from
occlusion_corridor... where are the ones from ramp/ramp w friction"):
`run_model_population.py` -- the script EVERY statistically-powered
Cosmos/Wan population in this project comes from -- generates, scores, and
DISCARDS every episode's frames in memory by design, for every scenario,
occlusion_corridor included. Confirmed nothing was ever saved: the one
existing video (occlusion_corridor, Wan, seed 1) came from a one-off
manual `remote/call_wan.py` invocation during early validation, not from
any real population run. Nothing existed for `ramp_descent`/`ramp_descent_
high_friction` at all.

Fixed by adding `--model {cosmos,wan}` to `render_annotated_comparison_
video.py` as an alternative to `--continuation-path`: calls the SAME
`make_generate_fn` dispatch `run_model_population.py` itself uses (one
real, freshly-paid-for model call, not reusing anything), saves the RAW
output to `results/videos/<scenario>_<model>_seed<seed>_raw.mp4` for
provenance BEFORE annotating -- so this real call is never silently
un-recorded again. `--continuation-path` changed from `required=True` to
optional, with a `parser.error` if neither or both of `--model`/
`--continuation-path` are given. Verified end-to-end: generated and saved
both `ramp_descent_high_friction_cosmos_seed1_annotated.mp4` and
`ramp_descent_high_friction_wan_seed1_annotated.mp4` (bordered, matching
the occlusion_corridor precedent).

Separately, a real confound raised directly ("in occlusion corridor, the
generating models may be failing because they think that it's a solid
surface... make the inside of the corridor lit... so it's clear that it's
open"): `SCENARIO_PROMPTS["occlusion_corridor"]` already states in text
that the ball "continues to roll on the other side," but the wall's own
VISUAL geometry (a plain solid gray box, `alpha=1` by design -- the
occlusion test's own honesty requirement, can't be relaxed) gives a real
generative model no visual cue that it's a passage rather than a dead end,
and video models are known to weight pixels over prompt text, especially
for continuation.

First approach tried (hollow tunnel -- side walls + roof, same outer
footprint, open front/back) and REJECTED after direct measurement, not
assumed to work: built `scripts/measure_occluder_visibility.py` (sweeps
the ball through a fixed x-range at 5mm resolution, reads real rendered
segmentation -- the same technique the original `OCCLUSION_CORRIDOR_WALL_
BOUNDS` measurement used, made reusable this time since it wasn't saved as
a script before), validated it reproduces the documented (4.395, 5.870)
span on the untouched wall, then used it plus a raytrace of the scene's
own camera geometry (cam_pos/cam_mat extracted directly from the
renderer, not assumed) to confirm: this camera's line of sight to the ball
passes through the wall's NEAR face (not the roof, not the far wall) --
because the camera sits far off-axis in y at a steep elevation
(elevation=-45), so hollowing the box still rendered as a solid silhouette
(confirmed by an actual render, not inferred), and genuinely opening the
near face would remove the exact surface creating the occlusion, letting
the ball show through and breaking the test outright.

Correct fix: a purely cosmetic decal. A second, thin, non-colliding geom
(`wall_opening` / `occluder_opening` for the moving variant, child of the
SAME body so it tracks the occluder's own motion), positioned just in
front of the unchanged wall face, pale-colored (`rgba="0.95 0.95 0.75 1"`)
and inset from the wall's edges -- reads as a lit doorway/archway rather
than a blank dead end, literally answering "make the inside lit." The
occluding geometry itself is untouched: re-measured directly with the
decal present on both `occlusion_corridor.xml` and `occlusion_corridor_
distractor.xml` -- (4.395, 5.870), byte-identical to before, so `scene_
geometry.OCCLUSION_CORRIDOR_WALL_BOUNDS` needed no update and no GATE 1a/
1b/GATE 2 recalibration. Applied identically to all three occlusion_
corridor scene variants (`occlusion_corridor.xml`, `_distractor.xml`,
`_moving.xml` -- the last as a child geom of the existing `occluder` body,
verified by rendering mid-slide that it moves correctly with the body and
still fully hides the ball). Full local test suite re-run clean (66
passed) as a regression check.

**Re-run on the decal scene, real numbers, not assumed (2026-08):** Cosmos
n=80 -- VI_50=1.40s, 95% CI [1.40, 1.40], termination profile R5:100%,
censoring=0.00. **Byte-identical to the pre-decal n=80 result** (same
VI_50, same CI, same termination profile). Honest finding, not spun as a
win: the wall's visual styling made ZERO measurable difference to Cosmos
on this scenario. Combined with the termination profile being 100% R5
(kinematic) and 0% R2 (existence) both before and after, this is further
evidence Cosmos was never "reading the wall as solid and stopping" in the
first place -- it's producing kinematically-wrong motion almost
immediately regardless of what the wall looks like, a different failure
mode than the one the decal was built to address. Wan re-run at n=5 (up
from n=2): VI_50=1.47s, 95% CI [1.03, 2.40], termination profile
R2:20%/R5:80% -- termination profile now shows some R2 (existence)
alongside R5, unlike Cosmos, but n=5 is too small to treat this as a real
model-vs-model difference rather than sampling noise. The SAM2-candidate-
disambiguation risk flagged above was NOT specifically instrumented for in
this re-run (no per-episode mask-count diagnostic added) -- still open,
still worth watching, not ruled out just because the aggregate numbers
came back clean.

### M2.9 — Grid videos (zero extra GPU cost) + a real ~12h stuck-job incident (2026-08)

Real gap found directly ("all of the videos should be being generated
possibly in one single video that shows all of the rollouts to save
space"): saving one video per episode doesn't scale (80 Cosmos episodes =
80 files). Fixed by extracting `draw_border`/`draw_marker` out of
`render_annotated_comparison_video.py` into a new shared module,
`vitals/render/annotate.py` (`annotate_episode_frames`, `tile_grid_video`)
-- the same "one implementation, not two" choice already made for
`track_and_reconstruct`. `VideoWorldModel.predict()` gained an optional
`capture_frames` flag (default `False`, zero behavior/memory change for
every existing caller) that stashes the pixels+reconstructed positions
it already computed on `self.last_capture`, so `run_model_population.py`'s
new `--n-video-episodes N` can build a grid video for the first N seeds
(in `--seeds` order) at literally zero extra GPU cost -- it's the SAME
episodes already being scored, just also captured, not a second pass.
Partial grids (e.g. Wan's 5-episode run against a 9-tile request) fill
the leftover tiles with a gray separator color rather than padding by
re-running extra episodes just to fill a requested shape. 6 new unit
tests (`tests/test_annotate.py`, `tests/test_video_model_adapter.py`'s
own new capture test) -- all pure numpy/PIL, no GPU needed, full suite
72/72 passing. Produced `results/videos/occlusion_corridor_cosmos_grid.mp4`
(9 episodes, 3x3) and `..._wan_grid.mp4` (5 episodes, 2x3, one tile
gray-filled).

**Real operational incident, worth remembering:** the first Cosmos n=80
re-run attempt hung for **~12 hours** with no error, no crash, no result
file -- `ps`'s own CPU-time column stayed essentially flat (0:31 total)
across the whole span, which at first looked like conclusive proof of a
hang, until the Wan re-run (which DID complete successfully) turned out
to show the exact same flat-CPU pattern the whole time it was genuinely,
correctly working -- this workload is almost entirely network-wait, not
local compute, so CPU time alone is NOT a reliable stuck-vs-slow signal
for it. What WAS reliable: `modal app list`'s own Tasks column showed
`0` active tasks for `vitals-cosmos` while the local process claimed to
still be waiting on it -- a job genuinely in flight shows up there; ours
didn't, meaning the local process was blocked on a connection that had
already died, not on real remote compute. Killed it and relaunched with
`python3 -u` (unbuffered stdout) specifically so the NEXT time this
happens, real progress is visible immediately rather than sitting fully
buffered until process exit (which is also why the first attempt's local
log looked empty the whole time -- not itself evidence of anything, since
the same buffering hid Wan's genuine progress too, confirmed by direct
comparison once Wan's own log flushed at completion). The retry completed
normally. Lesson for next time a background Modal job looks stalled:
check `modal app list`'s Tasks column before assuming either "it's fine,
just slow" or "it's stuck" from local process metrics alone -- both
guesses were wrong here on first pass, in opposite directions, until
checked directly.

### M3 — Rendering (week 3, first half)

`render/` module: `Trajectory -> (frames, ground_truth)`.

Define it as a protocol from the start so a different renderer can be dropped
in later without touching callers:

```python
class Renderer(Protocol):
    name: str
    deterministic: bool
    emits_ground_truth: bool
    def render(self, traj, cameras, realism) -> tuple[Frames, GroundTruth | None]: ...
```

**Done when:** Blender emits, per frame, RGB + cryptomatte instance IDs + depth
+ camera extrinsics, and re-rendering the same trajectory at a different
lighting/viewpoint produces identical physics by construction.

### M4 — The measurement operator (week 3, second half)

`phi/` module: `video -> Trajectory`. Components:

| Output | Method | Feeds |
|---|---|---|
| instance masks + IDs | promptable video segmentation, prompted with GT frame-0 masks | P2 |
| point tracks + visibility | joint point tracker, N points per object | P4, P2 |
| 6-DoF pose | Kabsch fit over ≥3 visible tracked points | P4, P3 |
| 3D positions | multi-view triangulation, else monocular depth | P3, P4 |
| appearance descriptors | DINO-family, mask-pooled | P2 |
| static keypoints + camera | learned keypoints + matcher on non-instance pixels, then PnP/BA | P1 |
| contact/support graph | **learned relation head — see §10** | P3 |

Two rules: track points, not mask centroids (a centroid is biased the instant
an object is partially occluded); and visibility flags are what distinguish
"occluded" from "gone", which is the entire content of P2.

**Done when:** `Phi` produces a `Trajectory` from video, and the same
`sigma_k` functions run on it unmodified.

### M5 — GATE 2, the instrument tax (week 3, end)

Run the M2 mutant sweep again, but measured through `Phi` from video instead
of ground-truth state.

**The event-time difference between M2 and M5 on identical mutants IS the
instrument error floor.** Publish it. This is free and no existing benchmark
reports it.

**Done when:** per-component error floors are published (mask IoU, ID-switch
rate, track position error, occlusion-flag accuracy, per-risk precision/recall
and event-time bias), plus the artifact-injected sweep — reference renders
degraded with graded generation-style artifacts, showing how each component's
error scales with artifact severity.

**FIRST PASS DONE (occlusion_corridor only; ramp_descent and the
artifact-injected sweep are follow-ups, not done here — see below).**
`scripts/run_gate2.py`. Scope: null/vanish/velocity_freeze/jitter (10/8/8/8
instances; `wrong_gravity`/`duplicate`/`teleport` excluded for this scenario
specifically, not removed from `mutants/library.py` — see the script's own
docstring), each clip truncated to 6.67s (T_RENDER=200 frames) for GPU-time
reasons, t_star=2.0s. Same reference ensemble, thresholds, and detector math
as GATE 1 (`scripts/run_l0_demo.py`), reused unmodified; the ONLY thing that
differs between an instance's L0 and L1 event is the measurement channel
(state vs. real SAM2/DINOv2-on-Modal + `phi/reconstruct.py`).

Published per-mutant results (n as above):

| mutant | L0 fired | L1 fired | correct risk (L0/L1) | event-time bias (L1-L0) | mean mask IoU | mean occlusion-flag acc. | mean position error |
|---|---|---|---|---|---|---|---|
| null | 0/10 | 9/10 | 10/10 · 1/10 | n/a (no joint fires) | 0.79 | 0.88 | 0.046m |
| vanish | 8/8 | 8/8 | 8/8 · 8/8 | -0.079s ± 0.165s (n=8) | 0.93 | 1.00 | 0.046m |
| velocity_freeze | 7/8 | 8/8 | 7/8 · 5/8 | -0.786s ± 0.882s (n=7) | 0.78 | 0.88 | 0.053m |
| jitter | 8/8 | 8/8 | 8/8 · 8/8 | +0.000s ± 0.000s (n=8) | 0.83 | 0.96 | 0.215m |

Raw per-instance results: `gate2_results.json` (scratchpad, not committed —
regenerate via `scripts/run_gate2.py`).

**vanish and jitter are the clean, publishable instrument-tax numbers.**
vanish's R2 detection is accurate to -0.079s ± 0.165s through the real
pipeline — a small, tight, genuinely informative error floor. jitter fires
at t=0 identically on both sides (its own severity is large enough to be
instantaneous either way), so its event-time bias is a true (if trivial)
zero; its real signal is elsewhere -- see the ID-switch note below.

**velocity_freeze surfaced a qualitative failure mode, not just a timing
bias**: 3 of 8 instances fired R2 (existence) through Phi where the correct
answer was R5 (kinematic) — the drift from frozen velocity apparently moved
the ball somewhere tracking couldn't follow (off its predictable path, or
off-camera), and precedence (3.6: R2 outranks R5) reports the resulting
tracking failure instead of the kinematic violation it was supposed to
measure. ID-switches=0/5 rules out wrong-object identification as the
cause. This means for THIS mutant, on THIS scenario, Phi doesn't just add
timing noise to the right answer -- it can report the wrong risk entirely,
which is a more severe form of instrument tax than a bias number captures.
Root-caused precisely, not left as a guess -- see the null false-positive
writeup below, follow-up 3.

**jitter's elevated ID-switch rate (9/61, ~15%) comes with a caveat**:
jitter injects continuous positional noise into the GROUND TRUTH itself, so
the ID-switch heuristic (position error at a reid-match frame exceeding
0.5m) can be triggered by the ground truth legitimately having moved
unpredictably at that exact instant, not only by Phi genuinely matching the
wrong object. This number should be read as an upper bound on true
ID-switch rate for this mutant, not a clean measurement -- unlike vanish/
velocity_freeze/null, where ground truth position is otherwise smooth and
the same heuristic is trustworthy.

**null's 9/10 false-positive rate (vs. GATE 1's calibrated ~1% target in
state space) is real, fully root-caused, and CONFIRMED ROBUST -- not a
truncation artifact.** All three follow-ups flagged after the first pass are
now closed, with two corrections to the original hypotheses below (both
found by checking against real data rather than trusting a plausible-looking
first theory).

*Follow-up 1 -- re-run at the full 8s horizon.* Same 10 null instances
(identical seeds), re-rendered and re-measured at the manifest's native 240
frames instead of the 200-frame truncation. Result: **bit-identical** --
9/10 fired, at the exact same times, same risk, same n_reid_events (just
proportionally more search attempts for the instances that thrash, since
there are more frames to thrash across). This DISPROVES the original
writeup's caveat that the 90% rate was likely inflated by a too-short
window -- extending the window changes nothing, because (see below) the
balls that end up falsely flagged have already settled into their final
resting position within the first ~2-3s, well inside even the shorter
window. **The true, window-length-independent R2 false-positive rate for
this scenario at lam=6.0 through Phi is ~90%**, not GATE 1's state-space
~1%. Full-horizon results: `gate2_null_full_rerun.json`.

*Follow-up 2 -- root-cause the "thrashing" instances (`null_1/2/3/4/8`).*
**The original partial-occlusion-flicker hypothesis was WRONG** -- checked
directly against the rendered ground-truth masks (not assumed): the
visibility transition at the wall is a single, clean, mostly-binary event
(area ramps 65->53->36->17->3->0 over five frames, then stays exactly zero
for the whole occluded stretch -- reconstruct.py's own "binary occlusion"
assumption was actually RIGHT). The real mechanism is entirely different
and was found by comparing each instance's own final resting x-position
against `OCCLUSION_CORRIDOR_WALL_BOUNDS=(4.25, 5.75)`, the coarse
world-x-interval approximation the physics-prior's occluder check uses:
  - `null_0/6/7/9` rest COMFORTABLY inside the interval (final x = 4.51-4.95m)
    -- physics permanently and correctly (by its own simplified model)
    believes the object is still behind the wall, `known_occluded` stays
    True forever, both tiers stay suppressed (defect #19's fix working
    exactly as designed) -- `n_reid_events=0`, no further harm beyond the
    unresolved-existence signal itself.
  - `null_1/2/3/8` rest just OUTSIDE the interval (final x = 5.85-5.94m,
    only 0.10-0.19m past the coarse boundary) while STILL not actually
    visible to the camera -- the true visual occlusion boundary (a function
    of the wall's real 3D geometry and this camera's oblique viewing angle)
    doesn't line up exactly with the simplified world-x interval used for
    the occluder check, which is precise enough for objects that clearly
    stop well inside or well past the wall, but not for one resting in this
    ~0.2m margin. `known_occluded` incorrectly releases, so physics offers a
    (wrong) target and search retries every remaining frame in vain --
    246-344 logged attempts, all-but-permanently failing. Two of these four
    (`null_2/8`) additionally hallucinate a wrong match often enough during
    that retrying to drag mask IoU and occlusion-flag accuracy down to
    0.29-0.35 (Phi confidently reports something that isn't there); the
    other two (`null_1/3`) fail cleanly (never lock onto anything wrong,
    `occlusion_flag_accuracy=1.00`) -- wasted compute, not corrupted output.
  - `null_4` is a partial exception to this otherwise clean split (final x
    = 5.64m, inside the coarse interval, yet still shows high search churn)
    -- not fully explained; likely the predicted position transiently
    crosses the interval boundary before settling, but this wasn't traced
    further. Noted honestly rather than papered over.

  Root mechanism, in one sentence: `OCCLUSION_CORRIDOR_WALL_BOUNDS` is a
  reasonable approximation of the wall's occlusion footprint, but it is
  still an approximation, and this scenario's own lam=6.0 calibration
  (chosen specifically to split outcomes across "stop short / stop behind
  wall / re-emerge") puts a large fraction of any reference or null
  population resting exactly in the margin where that approximation is
  weakest.

*Follow-up 3 -- root-cause velocity_freeze's R2-preemption (3/8
instances).* Confirmed directly by re-simulating what the physics prior's
OWN decelerating model (calibrated deceleration a=1.246 m/s^2, unaware any
mutant exists) would have predicted from each instance's (x0, v0) at the
drop instant, and comparing to the TRUE frozen-velocity trajectory: for 7 of
8 instances, physics's own model predicts the ball permanently STOPS before
ever reaching the far side of the wall (`t_stop` reached first) -- while the
true, frozen-velocity ball never decelerates at all and does eventually
cross back into view (confirmed against ground truth: true re-emergence
times of 2.5-5.6s). When physics believes an object has permanently
stopped, `known_occluded` never releases, search stays suppressed for the
rest of the clip exactly like the null mechanism above, and the resulting
unresolved-existence signal preempts the kinematic (R5) detection this
mutant was designed to trigger. The 5 instances that DID fire R5 correctly
show high search churn instead (135-181 events) -- meaning their gap
structure broke `known_occluded`'s permanent lock at some point (likely the
same coarse-boundary margin effect as follow-up 2, not traced further per
instance), allowing eventual reacquisition. This is a genuine, somewhat
ironic methodological finding: the SAME machinery that correctly handles
genuine occlusion (defect #17/#19) creates a NEW failure mode specifically
on the one mutant class designed to violate the physics assumption that
machinery depends on -- a physics-informed re-identification prior will
systematically misattribute failures to the wrong risk category when the
failure itself is a physics violation. Worth a note for the eventual
writeup (§12), not just a bug entry.

See defect #20 (§6) for a real, kept, but (per follow-up 1/2 above)
ultimately not-the-dominant-cause fix found while investigating the null
false-positive rate.

**Remaining follow-ups, still genuinely open:**
- ramp_descent's own GATE 2 sweep (needs non-planar 3D reconstruction, a
  separate, parallel, already-tracked gap -- see §5/§6).
- `null_4`'s partial exception to the coarse-boundary-margin story (above).
- The artifact-injection sweep (graded pixel-level degradation of reference
  renders) -- no code exists for this yet.

**Attempted (2026-08): recalibrating the existence threshold against
Phi-measured references instead of reusing GATE 1's state-space one.**
Built a fresh M=30 reference ensemble (`scripts/gate2_recalibrate.py`,
disjoint seed range), ran each through the real Phi pipeline, calibrated
`theta` from those instead, then re-evaluated all 34 original GATE 2
instances against it (both null AND the non-null mutants, not just the
false-positive case -- a threshold change must be checked for sensitivity,
not only specificity).

**Result: this fixed specificity perfectly and destroyed sensitivity
almost completely -- not a fix, a different failure mode.** null's
false-positive rate went 9/10 -> 0/10 (correct_risk 1/10 -> 10/10) exactly
as hoped. But `vanish` -- a mutant that unambiguously SHOULD fire R2 every
time, a genuine, permanent disappearance -- went from correctly firing
8/8 to firing **0/8**. `velocity_freeze` went 8/8 -> 0/8 fired too. Only
`jitter` was unaffected (its own signal is large enough to be
instantaneous regardless of threshold, so it was never testing this
particular boundary).

**Mechanism, not just observed -- `estimate_threshold`'s own documented
design** (`detect/thresholds.py`, `_defined_cutoff`): calibration
truncates to `theta=inf` for the remainder of the clip once too large a
fraction of the calibrating reference ensemble goes permanently undefined
at some point -- an honest "can no longer be meaningfully calibrated
here," not a bug (this is the SAME mechanism defect #13 already
documented, now hitting the reference ensemble itself instead of just the
candidate). Since a large share of the Phi reference ensemble legitimately
never resolves existence (the identical mechanism behind the original
90% false-positive finding), that cutoff almost certainly triggers early
-- making theta effectively infinite for most of the clip, which means
NOTHING can ever cross it there, genuine violation or not. The fix didn't
make R2 detection more accurate through Phi; it made the risk category
nearly inert.

**Open, unresolved as of this entry -- genuinely a design fork, not a
mechanical next step:**
- Combine both candidate fixes -- narrow the false-positive rate via a
  more precise `OCCLUSION_CORRIDOR_WALL_BOUNDS` FIRST (fewer Phi references
  legitimately go permanently unresolved), then recalibrate on a
  less-degenerate reference ensemble.
- Tune `estimate_threshold`'s own `min_defined_frac` parameter for the Phi
  channel specifically, rather than assuming its state-space-tuned default
  (1-alpha) transfers -- it exists exactly to trade off calibration-window
  length against tolerance for undefined references, and was never
  validated against a reference ensemble this non-stationary.
- Reconsider whether a single whole-path scalar threshold (this module's
  entire design, by its own docstring) is the right statistical shape for
  Phi's existence signal at all, given how much more STRUCTURED (concentrated
  specifically in the wall-occlusion region, not diffuse) its
  undefined-ness is compared to state space's comparatively uniform
  kinematic noise.
- A more fundamental reconsideration: `sigma_existence` currently collapses
  "provably still hidden, existence honestly unresolved" and "provably
  gone" into the same scalar (documented, deliberate, per
  `detect/statistics.py`'s own docstring) -- recalibrating the THRESHOLD
  can't fix a problem that's really about the STATISTIC conflating two
  different things. Worth asking whether Phi needs a genuinely different
  existence signal, not just a different number to compare it against.
**Attempted (2026-08, continued): the first candidate above -- narrow the
wall bounds, then retry.** Measured the TRUE empirical visual occlusion
span directly (not assumed): swept the ball through fixed x positions at
5mm resolution and recorded actual rendered visibility. True span is
(4.395, 5.870) (last/first visible just outside), vs the coarse box-extent
value of (4.25, 5.75) -- the far edge alone was 0.12m short.
`OCCLUSION_CORRIDOR_WALL_BOUNDS` updated to `(4.39, 5.87)`. Re-ran all 64
already-uploaded episodes (34 original + the 30 Phi references) against
the corrected bounds -- no re-render needed, just fresh
`run_gate2_episode` calls against the redeployed app.

**Result: no improvement, and a real, understood reason why.** With the
corrected bounds and GATE 1's ORIGINAL state-space theta (untouched),
`null`'s false-positive rate was EXACTLY unchanged: 9/10 fired, the same
instances, at nearly identical times. The interval didn't just widen, it
SHIFTED (near edge moved in by 0.14m at the same time the far edge moved
out by 0.12m) -- fixing the far-edge misclassification this was targeting
while introducing new near-edge misclassification elsewhere, visible
directly in the data as a large increase in search churn for previously
undisturbed instances (`vanish_0/1/2` went from ~0 search attempts to 276
each). Net effect on the actual outcome: nothing. Recalibrating theta on
the corrected-bounds Phi reference ensemble reproduced the EXACT same
failure mode as the original (uncorrected-bounds) attempt above: specificity
fixed perfectly (null 0/10 fired), sensitivity destroyed identically
(`vanish` 0/8, `velocity_freeze` 0/8) -- with 20/30 of even this
corrected-geometry reference ensemble still never resolving existence by
clip end.

**This is a real, informative negative result, not just bad luck: it
confirms the theoretical argument above rather than just restating it.**
No amount of boundary precision can fix a case like `null_0` (rests at
x=4.72, comfortably, unambiguously behind the wall by ANY reasonable
boundary) -- the clip simply ends before that gap resolves, which is a
property of this scenario's own physics and the chosen observation window,
not of where exactly the occluder's edge is measured to be. The
wall-bounds-precision hypothesis is now considered CLOSED, not just
deprioritized -- tested directly, not merely reasoned about, and did not
work. The two untried options -- tuning `min_defined_frac` for the Phi
channel, and reconsidering whether `sigma_existence` needs to distinguish
"unresolved" from "gone" as a genuinely different signal rather than one
scalar -- remain open. Given two threshold-side attempts have now failed
identically, the statistic-level option is looking more likely to be the
real fix than a threshold-tuning one, but this is not yet decided or
attempted.

**Attempted (2026-08, third attempt): reconsider the statistic itself,
rather than the threshold.** A pure gap-DURATION/survival statistic
(reusing `stats/survival.py`'s censoring machinery) was designed first and
then rejected on paper, before writing any code: it can't actually
separate `vanish` from `null` either, because `vanish`'s `t_star` fires
independent of the object's position, so a real vanish that happens to
occur near the wall produces a same-duration, same-timing, never-closing
gap as a genuinely, correctly occluded `null` instance -- duration alone
carries no discriminating signal in that case, any more than a raw
existence count does. The distinguishing signal that survives this check:
whether the gap is SPATIALLY explained by a known occluder throughout,
using the `occluder_bounds`/`known_occluded` check `motion_prior.py`
already computes during search (defect #17) but never recorded.
`null_0`'s gap has `known_occluded=True` for literally every frame --
that's WHY it shows `n_reid_events=0` (physics never even attempted a
search). A `vanish` event, triggered independent of position, is usually
spatially unexplained (physics actively searches in open space and finds
nothing) -- except for the coincidental cases where it happens to occur
near the wall, which remain genuinely ambiguous by design, not a bug in
this fix.

**Implementation**: `reidentify.py`'s `ReidentifyResult` gained
`known_occluded_frames: {frame_idx: bool}`, recorded for every search
attempt on the primary object regardless of outcome (previously this
information existed only transiently inside the search loop, used once
and discarded). `reconstruct.py`'s `_existence_mask` gained a new
`_gap_explained_by_occluder` check: a gap counts as present if EVERY
recorded search attempt within it found `known_occluded=True` -- exactly
the same status ordinary, invisible-in-state-space occlusion already has.
Threaded through `reconstruct_trajectory`'s new `known_occluded_frames`
parameter and `remote/modal_app.py`'s two callers.
`tests/test_reconstruct.py::test_occluder_explained_gap_counts_as_present`
covers all three cases (fully explained, partially explained, no info at
all) directly.

**Validated against the SAME 34 original GATE 2 episodes, GATE 1's
ORIGINAL untouched state-space theta (no recalibration this time -- this
fix changes what existence MEANS for Phi, not a threshold):**

| mutant | fired (old→new) | correct (old→new) | notes |
|---|---|---|---|
| null | 9/10 → 6/10 | 1/10 → 4/10 | 3 fully-explained instances (`null_0/7/9`, all `n_reid=0`) now correctly censored |
| vanish | 8/8 → 6/8 | 8/8 → 6/8 | the 2 lost (`vanish_3/5`) are the predicted philosophically-ambiguous case -- coincidental occlusion-timing overlap, not a regression; remaining 6 kept a tight bias (-0.083s ± 0.186s, same quality as before) |
| velocity_freeze | 8/8 → 6/8 | 5/8 → 5/8 (unchanged count, real qualitative win) | `velocity_freeze_2/6` -- the ORIGINAL R2-misattribution cases (§7 M5's own finding) -- now correctly abstain (censored) instead of confidently reporting the WRONG risk category. A silent miss is a materially less harmful failure than a wrong, confident answer, even though it doesn't move the raw "correct" count |
| jitter | 8/8 → 8/8 | 8/8 → 8/8 | completely unaffected, as expected (fires at t=0, before any of this machinery is relevant) |

**This is the first attempt at the R2/existence problem that produces a
real, principled improvement rather than trading one failure mode for
another.** null's false-positive rate: 90% -> 60%, not eliminated, but
substantial, with the theoretical reason for the remainder well understood
(see below). Unlike both threshold-side attempts, sensitivity loss is
small, explainable, and concentrated exactly where theory predicted it
would be, not total collapse.

**The remaining 6/10 null false positives are the ALREADY-IDENTIFIED,
SEPARATE "thrashing"/coarse-boundary-margin group** (`null_1/2/3/4/8`,
`n_reid` 232-256 -- high search churn, not `n_reid=0`), plus `null_6`
(`n_reid=4`, a small number of attempts, at least one apparently NOT
`known_occluded`, worth a closer look but not investigated further this
session -- possibly a brief transient at gap onset before the physics fit
stabilizes). This fix was never expected to help that group -- it targets
gaps physics never even tried to search (fully explained throughout), not
gaps where physics searched extensively and inconsistently. That group
remains the "coarse-boundary-margin"/small-object-tracking-instability
problem documented above, unresolved by (and unrelated to) this fix.

**DONE (2026-08): the artifact-injected sweep** -- the last item on M5's
own "done when" checklist. Implemented as a scratchpad script
(`gate2_artifact_sweep.py`, not yet moved into `scripts/`), matching this
sweep's own exploratory/first-pass status. Three artifact types, chosen as genuine
proxies for known generative-video-model failure modes, not arbitrary
choices: Gaussian blur (low denoising quality), Gaussian pixel noise
(sampling artifacts), and temporal flicker (independent per-frame
brightness jitter -- a real camera never does this, but frame-to-frame
consistency is a known generative-model weak point). 4 severity levels
each, applied to 4 already-rendered, already-validated GATE 2 episodes'
CLEAN frames (no re-simulation -- only the frames are degraded;
`gt_masks.npy`/camera calibration copied unchanged, since the artifact is
a rendering-quality effect, not a physics or camera change), chosen for
diversity: `vanish_0` (clean tracking baseline), `null_1` (occlusion-heavy
re-identification churn), `velocity_freeze_0` (kinematic drift), `jitter_0`
(already-noisy ground truth). 48 real end-to-end Modal runs total.

**Finding 1 -- blur has a sharp CLIFF, not a gradual decline, and the
cliff point is episode-dependent.** All 4 episodes show the same shape:
near-baseline IoU at low severity, then a catastrophic drop of 0.6-0.9 IoU
between two adjacent severity steps, not a smooth curve:

| base episode | blur=1 | blur=2 | blur=4 | blur=8 |
|---|---|---|---|---|
| vanish_0 | 0.927 | 0.891 | **0.190** | 0.085 |
| null_1 | 0.941 | 0.918 | 0.909 | **0.006** |
| velocity_freeze_0 | 0.865 | 0.787 | 0.758 | **0.044** |
| jitter_0 | 0.676 | 0.590 | 0.301 | 0.010 |

Confirmed by `n_reid` collapsing to near-zero at the cliff (e.g.
`vanish_0`: 276 -> 1 search attempts) -- this is NOT increased
re-identification struggle, it's SAM2's own mask generator failing to
propose any reasonable candidate at all once the object is blurred past
some episode-specific threshold. There is nothing left to search FOR, not
more searching happening.

**Finding 2 -- temporal flicker is remarkably well tolerated, EXCEPT for
one clear anomaly.** 3 of 4 episodes (`null_1`, `velocity_freeze_0`,
`jitter_0`) show almost NO measurable degradation across the entire
severity range (IoU within ~0.01-0.02 of baseline even at the maximum
severity tested) -- a real, well-reasoned finding, not a null result: SAM2's
video predictor uses temporal memory across frames, so a per-frame
INDEPENDENT brightness shift gets absorbed by the tracker's own multi-frame
reasoning, unlike blur/noise which corrupt the CURRENT frame's own spatial
signal directly. `vanish_0` is the exception -- flat through severity=20,
then a sharp drop at severity=40 (0.933 -> 0.838 -> 0.279). Not
investigated further, but `vanish_0` also has the highest baseline
re-identification churn of the four episodes (276 search attempts even at
zero artifact severity) -- plausible, unconfirmed hypothesis: it has the
least headroom before any additional degradation destabilizes an already-
marginal candidate-matching process.

**Finding 3 -- noise degrades mostly monotonically, but with the SAME
episode (`vanish_0`) showing a real non-monotonic anomaly.** `null_1`,
`velocity_freeze_0`, `jitter_0` all decline smoothly and monotonically
with noise severity. `vanish_0` does not: 0.936 -> 0.427 -> 0.730 -> 0.225
-- a partial RECOVERY at severity=40 between two worse neighbors. Recorded
honestly rather than smoothed over or discarded; not root-caused this
session. That this is the SAME episode showing the anomaly for BOTH noise
and flicker strengthens the "least headroom" hypothesis above, but doesn't
confirm it.

**occlusion_flag_accuracy tracks mean_iou closely throughout** (both
measure detection/segmentation quality) -- no additional independent
pattern beyond what's described above.

Raw results: `artifact_sweep_results.json` (scratchpad, not committed --
regenerate via the script). GATE 2's own "done when" checklist (§7 M5) is
now fully satisfied: per-component error floors published, event-time bias
measured, and the artifact-injected severity sweep complete.

### M5.5 — Scene diversity, phase 1: distractor robustness (2026-08)

Not an AGENT.md-original milestone -- added after GATE 2 landed, per an
explicit decision to broaden scene coverage before wiring in real models
(M6), and to build toward AGENT.md P5 ("modularity of mechanism" --
"interventions produce their effects, and *only* those"). Full P5 needs a
causal-cone propagation bound and stays out of scope (§7 M5/§11 already say
so, for good reason -- it is a real, separate research design problem, "not
an implementation task" per §10's own words). What IS buildable now,
without a causal cone, is the SPECIFICITY half of that property in
isolation: does re-identification correctly ignore a visually salient but
causally inert nuisance? (The SENSITIVITY half -- does tracking correctly
react to a real causal event, e.g. a collision -- needs a second body that
actually interacts physically, which is phase 2/M5.6 below, not this.)

**New scene**: `scenes/occlusion_corridor_distractor.xml` /
`configs/manifests/occlusion_corridor_distractor.yaml`. Identical to
occlusion_corridor in every scored respect (same lambda, same perturbation
mode, ball stays body index 0) plus one addition: a second, non-colliding
body ("decoy") -- same size as the ball, different color (blue vs red) so
appearance-based re-identification has real signal to use, different
starting lane and a comparable (not identical) push so it settles near the
SAME x-region as the ball across the reference ensemble, deliberately
targeting the exact coarse-wall-bounds-margin zone GATE 2 (M5) found
problematic. contype/conaffinity bitmasks give it its own collision group so
it rolls on the floor but passes through the ball -- caught directly during
setup: an earlier version zeroed its contype/conaffinity entirely to
disable collision with the ball, which also silently disabled collision
with the FLOOR, sending it into freefall.

**No new code was needed in `phi/` or `detect/` for this phase** --
`physics/runner.py::rollout`, `Trajectory`, the renderer, and
`detect/statistics.py`'s `sigma_k(traj, others, obj=0)` signature are all
already either fully K-agnostic or specifically parameterized to `obj=0`,
which stays the ball regardless of what else is in the scene. This is
scene design plus evaluation, not an engineering lift.

**Validated, not assumed**: found a seed (4) by direct search where the
decoy is fully, continuously visible right at the ball's own reacquisition
frame (72) -- a genuine, not a strawman, temptation. Real end-to-end run
(existing `track_with_reidentification`, zero code changes): physics and
DINO both correctly rejected the decoy for 10 consecutive frames (66-75,
`physics_rejected_suspect` / `dino_rejected_implausible` -- defects #15/#18's
machinery holding up on a scene it was never built or tuned for), then
correctly reacquired the real ball at frame 76, 2.93px from ground truth.
`mean_iou_post_reacquisition=0.805`.

Broader sweep, N=8 fresh seeds (200-207): checked EVERY directly-visible
frame's reconstructed position against both the ball's and the decoy's true
3D position (not just a summary IoU number, which turned out to be
insufficient -- see below). **0/8 instances ever showed a frame closer to
the decoy than the true ball, out of 725 directly-visible frames checked
across the sweep.** `n_reid_events` ranged 0-142 (some instances never lost
the ball at all; one, seed=203, searched heavily but never locked onto the
decoy regardless). One instance (seed=201) showed a low `mean_iou_all=0.37`
despite zero decoy-lock, confirmed across all 150 of its own visible
frames -- a separate, minor, not-yet-investigated tracking-quality issue
(plausibly the same small/far-object instability GATE 2's null_2/8 already
surfaced), explicitly NOT a specificity failure.

**Caught and fixed a bug in the sweep script itself, worth recording**: the
first version of the N=8 sweep left `meta["drop"]`/`meta["rise"]` as empty
placeholders, silently making `mean_iou_post_reacquisition` always `None`
(the field that window is keyed off). Re-checked with a position-based
method (`run_gate2_episode`'s reconstructed trajectory vs. ground truth
directly) instead of patching the summary stat -- more precise anyway,
since it checks every frame instead of one aggregate window.

**Phase 1 conclusion: specificity holds on this scene, with zero new
`phi/` code.** The existing re-identification machinery -- built and tuned
entirely on the distractor-free scene, iterated through defects #14-#20 --
generalizes to a genuinely tempting adversarial addition without
modification. This is a real, positive result, not a foregone conclusion:
GATE 2's own null_2/null_8 already showed this same machinery CAN
hallucinate wrong matches under the right (or wrong) circumstances.

**A correction to this section's own earlier claim, and the FULL GATE 2
mutant-sweep pass this scenario never actually had (2026-08, "broaden to
more scenarios and see if it generalizes").** This section's own text
above claimed `detect/statistics.py`'s `sigma_k(traj, others, obj=0)` was
"already either fully K-agnostic or specifically parameterized to
obj=0" -- true of `sigma_kinematic`/`sigma_interpenetration`, WRONG for
`sigma_existence`, which had no `obj` parameter at all until today. Only
re-identification SPECIFICITY (above) had ever actually been validated on
this scene -- the full GATE 2 mutant sweep (null/vanish/velocity_freeze/
jitter through real existence/kinematic scoring, not just "does the
decoy get falsely matched") had never been run here before this session.

**A real, instrument-breaking bug found running it for the first time,
not a tuning issue:** `null` fired R2 10/10, immediately, regardless of
tracking quality -- confirmed directly by checking `position_error_n`
(zero for every one of 42 episodes, including episodes with IoU=0.79-
0.95 and zero re-identification search events at all). Root cause:
`sigma_existence`'s whole-scene object-COUNT comparison (`traj.present.
sum(axis=1)`) against a K=2 reference ensemble (ball + an always-present
decoy), scored against a K=1 real-video candidate (`track_and_
reconstruct`'s own single-object return, M5.6's scoped design -- the
decoy is never merged into the returned Trajectory). `candidate_count=1`
can never match `reference_mean~2`, at every frame, regardless of how
well the ball itself is tracked.

**Fixed by giving `sigma_existence` an optional `obj` parameter**
(`vitals/detect/statistics.py`) restricting the comparison to one fixed
object index's own presence (candidate vs. every reference, same index)
instead of the whole-scene count -- a strict no-op for every K=1
scenario, and deliberately NOT applied to `run_l0_demo.py`'s own STATS
(GATE 1's `duplicate` mutant -- an EXTRA object at a K the reference
never had -- genuinely needs the whole-count comparison to stay
detectable; GATE 2 never tests `duplicate` in the first place, M2.5's own
documented K>1-reconstruction scope limit). `run_gate2.py` binds
`obj=0` unconditionally. 2 new unit tests (`tests/test_core.py`) --
including one that first confirms the OLD whole-count logic really does
explode on the exact K-mismatched input before trusting the fix, the same
"prove the old code crashes, then prove the new code doesn't" discipline
already used for `reidentify.py`'s own IndexError fix this session.

**Re-ran GATE 2 with the fix: real, substantial improvement, not a
complete cure.** `null` false-termination dropped from 10/10 to 4/10;
`jitter` went from 8/8-fired-but-0/8-correct to a PERFECT 8/8 fired, 8/8
correct, bias exactly +0.000s, matching state-space precisely. `vanish`/
`velocity_freeze` now UNDER-fire (most instances go censored) --
root-caused, not just observed: every held-out seed checked enters the
wall's own KNOWN-occluder zone (`OCCLUSION_CORRIDOR_WALL_BOUNDS`) at
t=1.3-2.0s, almost exactly coinciding with GATE 2's fixed defect-
injection time (`t_star` = 25% of an 8.0s horizon = 2.0s). The "known
occluded, still plausibly present" logic (M5's own existence-statistic
hardening, correctly built to avoid flagging NORMAL wall-passage) also
swallows a genuine vanish landing at the same moment -- a real, if
partially pre-existing, limitation: base `occlusion_corridor`'s own
ALREADY-PUBLISHED, ALREADY-ACCEPTED GATE 2 numbers show the identical
effect, just milder (`vanish` L1 fired=6/8, not 8/8; `null` L1 fired=6/10,
NOT near-zero either -- GATE 2 has never been held to GATE 1a's own
~1% false-termination bar; its own persisted `gate_1a.passed` field is
always `None` by design, since it characterizes real-video measurement
noise honestly rather than gating on it).

**Accepted as GATE 2 "done" for this scenario, matching the SAME bar
base `occlusion_corridor` is already published and accepted under, not a
stricter one invented for this pass** (confirmed directly, not assumed:
post-fix numbers are BETTER than base's own on `null` [4/10 vs. 6/10] and
worse on `vanish`/`velocity_freeze` [3/8 vs. 6/8 fired] -- same general
"real video measurement is imperfect, characterize it honestly" story,
not evidence of a still-broken instrument). The decoy's own possible role
in the vanish gap specifically -- a plausible but unconfirmed hypothesis
that its static appearance occasionally gives the DINO re-identification
fallback something real, if wrong, to lock onto during a genuine
disappearance -- is left as a scoped, not-yet-investigated follow-up, not
something this session's own time was spent chasing further. Full suite
111/111.

### M5.6 — Scene diversity, phase 2: multi-object (K>1) support (core done)

Needed for: `duplicate`/`teleport` mutant scoring (both currently excluded
from GATE 2, §7 M5), the SENSITIVITY half of the P5-lite property phase 1
started (a real collision needs a second physical body), and a proper
ID-switch metric (GATE 2's jitter result already showed the current
position-heuristic ID-switch check is shaky -- §7 M5's own caveat).

**Not a scene-design problem this time -- a real `phi/` engineering gap.**
`reidentify.py::track_with_reidentification` hardcodes `obj_id=1` in
exactly two places (`add_new_mask` calls) and reads `mask_logits[0, 0]`
elsewhere (`segmentation.py` too) -- a single fixed object slot. Per-object
state (`empty_streak`, `cached_embedding`, `rejected_positions`) are plain
scalars/lists, not per-object dicts. `reconstruct_trajectory` hardcodes a
single-object output (`pos = np.full((T, 1, 3), ...)`, one `name`). SAM2's
own `SAM2VideoPredictor` API already supports multiple simultaneous
`obj_id`s natively (that parameter exists for exactly this) -- the wrapper
code here has just never exercised it. The capability isn't missing at the
model layer, only in this repo's use of it.

**Scoping decision, made explicitly before writing code**: fully general
simultaneous multi-object tracking (each object independently losing and
regaining tracking, inside ONE shared `propagate_in_video` loop) would need
a real redesign of the search-and-reprompt control flow -- the current
structure (propagate until a loss, pause, search, reprompt, resume) was
built around exactly one object being lost at a time, and doesn't obviously
extend to "object A is mid-search while object B is still propagating
normally." Deferred as too large a lift for what's needed right now.

**Built instead: the SECOND object gets simple tracking, no independent
search-and-reprompt of its own.** Whatever SAM2's own propagation
naturally tracks (including its own short-gap self-recovery,
`segmentation.py`'s own documented ~1s memory-window cliff) is all it ever
gets; if lost longer than that and it never comes back on its own, it just
stays lost for the rest of the clip. This is enough for `duplicate`/
`teleport` scoring and a basic real-collision sensitivity test, without
solving simultaneous-loss as a prerequisite.

**Implementation**: `track_with_reidentification` gained an optional
`frame0_mask_secondary` parameter -- when given, SAM2 obj_id=2 is prompted
alongside obj_id=1 in the SAME shared state/propagation loop; its masks
land in the new `ReidentifyResult.masks_secondary`, completely untouched by
the existing search machinery (which stays scoped to obj_id=1, unchanged).
Mask-slot lookup switched from a hardcoded `mask_logits[0, 0]` to
`mask_logits[obj_ids.index(1), 0]` (a no-op for the K=1 case, required once
a second obj_id can share the same propagation call -- SAM2 doesn't
guarantee slot order matches obj_id order). `reconstruct.py` gained
`merge_trajectories`, which stacks independently-reconstructed
single-object Trajectories (each object still goes through the EXISTING,
unmodified `reconstruct_trajectory` -- smaller diff, keeps each object's
reconstruction independently testable) into one K-object Trajectory.
`tests/test_reconstruct.py::test_merge_trajectories_stacks_along_object_axis`
covers the merge logic; the SAM2-side indexing change has no local unit
test (this repo's established pattern for `reidentify.py` changes is real
Modal validation, not a mocked SAM2 predictor -- see below).

**Validated end-to-end on Modal, first try, zero surprises**: new
`remote/modal_app.py::run_multiobject_episode`, run against the already-
uploaded `occlusion_corridor_distractor_seed4` episode (phase 1's own test
case) -- tracking the SAME decoy that phase 1 proved Phi correctly
IGNORES, now deliberately AS a real second tracked object instead. Primary
object (the ball) tracked identically to the single-object run before this
change (`mean_iou_primary=0.817`, matching exactly) -- confirms the
refactor didn't perturb the existing, already-validated path. Secondary
object (the decoy) tracked correctly while genuinely visible (frames 0-37,
`mean_iou_secondary=0.373` reflecting the frames it's truly gone), then
correctly reports `present=False`/`pos=NaN` for the entire remainder of the
clip once actually lost -- exactly the scoped design, no search ever
attempted for it, no privileged shortcut either.

**Explicitly NOT done in this pass** (separate, real follow-up work, not
core plumbing gaps):
- Rendering `duplicate`/`teleport` mutant instances at all. Discovered
  while scoping this: MuJoCo can only render a body that actually exists in
  the loaded scene XML -- a mutant's clone object (`traj.names[obj] +
  "_clone"`) has no corresponding body in ANY current scene file, so
  `mujoco_renderer.py`'s name-matching (`mj_name2id`) would silently fail
  to find it. Rendering these mutants needs either a scene with a
  pre-existing second body whose declared name matches the mutant's own
  naming convention, or a renderer-side name-mapping adapter. Phase 1's
  distractor scene's second body ("decoy") is a plausible substrate to
  reuse for this, but isn't wired up for it yet.
- The real-collision sensitivity scene (a second body that DOES physically
  interact, unlike phase 1's deliberately non-colliding decoy) -- this is
  what actually exercises the SENSITIVITY half of the P5-lite property;
  phase 2's own validation above only proves the PLUMBING works, not that
  any collision scenario has been built or tested yet.
- A proper multi-object ID-switch metric using this new K=2 output (GATE
  2's own position-heuristic caveat, §7 M5) -- not attempted.

### M5.7 — Scene diversity, phase 4: moving occluder (scoring wired, GATE 1b calibrated)

**Design decision, made explicitly (2026-08)**: the occluder is REAL,
independently-simulated, Sigma-perturbed motion, not a scripted/known-in-
advance schedule -- chosen over the cheaper scripted alternative because a
scripted occluder's position would be knowable in advance (safely "known
scene geometry" under the same precedent as the static wall), which
sidesteps the actually-interesting question this phase exists to ask: can
the ball's own occluder-aware search logic correctly fall back to "no known
occluder this frame" when the occluder's OWN position isn't currently
measured, rather than either assuming privileged knowledge of it or
ignoring it entirely? That's a real, new capability, not a restatement of
what M5's static wall already tests.

**New scene**: `scenes/occlusion_corridor_moving.xml` /
`configs/manifests/occlusion_corridor_moving.yaml`. Same ball, same wall
geometry/nominal x-position as `occlusion_corridor.xml` (occlusion geometry
already empirically validated there, reused rather than re-derived), but
the wall is now a real second free body ("occluder") sliding across the
corridor's width (y=-2.5 -> y=+2.5 over the 8s clip) instead of sitting
fixed at y=0. Because the ball's own x-perturbation and the occluder's own
y-sweep timing are independent, whether they actually overlap in space AND
time is genuinely contingent per episode -- confirmed directly: 5 of 14
sampled seeds produced a real occlusion event, 9 did not, with drop times
clustering around the occluder's fixed y=0 crossing (~frame 86) but rise
times varying with the ball's own position (133-158).

**A real bug found and fixed while building this, worth recording**: a box
given a real freejoint and a sliding initial velocity on the floor
decelerated to a dead stop within ~20-60 simulation steps (~0.1s)
REGARDLESS of the friction coefficient -- confirmed directly by setting
friction to exactly 0 and observing the identical stall. Root cause,
confirmed via `ncon` (contact count): the box was bouncing/tumbling on
contact (`ncon` dropped to 0, i.e. briefly airborne, then back to 4), not
sliding to a friction-limited stop -- a box-plane contact dynamics issue,
not a friction-tuning one. Fixed by making the occluder body `gravcomp="1"`
(weightless) and fully non-colliding (`contype="0" conaffinity="0"`, the
SAME treatment the original static wall already had, for the same
reason -- it never needed to physically interact with anything) -- still a
genuinely simulated, independently-perturbed free body, just without
fighting the solver over box-sliding physics this scene has no actual need
to model realistically. `tests/test_render.py::
test_moving_occluder_slides_without_stalling` is a direct regression guard
-- it would have caught this immediately (asserts the occluder actually
crosses the corridor, not merely twitches near its start).

**Now scorable end-to-end, all three of the previously-scoped items done
(2026-08):** `motion_prior.py`'s `occluder_bounds` was a single FIXED
(lo, hi) world-x interval, valid only for a permanently-stationary wall.
It's now EITHER that (unchanged, kept fully backward compatible for
occlusion_corridor's own static wall) OR a callable, built by the new
`occluder_bounds_from_track(masks_secondary, dt, cam_pos, cam_mat,
fovy_deg, width, height, occluder_plane_z, x_half_width, y_half_width,
max_staleness_frames=3)`:

1. **Tracked, not privileged.** The callable reads the occluder's CURRENT
   position from phase 2's (M5.6) live `result.masks_secondary` dict --
   unprojects its tracked centroid via the same `unproject_to_plane` ray/
   plane method everything else in this module already uses, at the
   occluder's own known (fixed) center height and half-extents (e.g.
   occlusion_corridor_moving.xml's occluder box: size (0.75, 0.6, 0.5), so
   `occluder_plane_z=0.5, x_half_width=0.75, y_half_width=0.6`) -- the
   occluder's SIZE is legitimate known scene geometry (same status as the
   original static wall's own extent), only its position is tracked.
2. **2D, not 1D.** `MetricTrackFit._occluder_box_at` now returns
   `(x_lo, x_hi, y_lo, y_hi)`, and `occluded_at`/`predict_pixel` check
   BOTH axes: the ball's predicted x against the occluder's (near-static)
   x-extent, AND the ball's own (fixed) lateral position against the
   occluder's CURRENTLY-tracked y-extent -- the thing that's actually
   time-varying here. The legacy static (lo, hi) tuple is treated as
   unbounded in the lateral axis (`(lo, hi, -inf, inf)`), so
   occlusion_corridor's own existing behavior is provably unchanged --
   confirmed by `tests/test_motion_prior.py::
   test_static_occluder_bounds_tuple_stays_unbounded_in_lateral_axis`.
3. **Graceful fallback, with one real refinement found while implementing
   it.** The spec called for "no known occluder this frame" whenever the
   occluder's position isn't currently measured -- but reading
   `track_with_reidentification`'s own propagation loop shows a search
   triggered for the primary object at frame `next_idx` always runs
   BEFORE `next_idx` itself has been propagated for the secondary object,
   so an EXACT-frame lookup would be permanently blind (always answering
   "unmeasured") right when it matters most. Fixed by looking backward up
   to `max_staleness_frames` (default 3) for the most recent tracked
   position instead of requiring an exact match -- justified by the
   occluder's own slow nominal speed (~0.023m/frame at 30fps, 0.7m/s),
   not a silent loosening of the honesty requirement: still returns None,
   not a fabricated position, once nothing is available even within that
   window (e.g. the secondary track has itself been lost, which per
   M5.6's own scope has no recovery machinery).

Wired into `track_with_reidentification` via a new `occluder_geometry`
parameter (`(occluder_plane_z, x_half_width, y_half_width)`, requires
`frame0_mask_secondary` and `camera`) -- when given, it builds the dynamic
lookup once at the top of the run and it REPLACES `occluder_bounds` for
every search in that run (a scene has one relevant occluder or the other,
never both).

Validated with real, direct unit tests against synthetic tracked masks --
`tests/test_motion_prior.py::test_dynamic_occluder_bounds_tracks_a_moving_
occluder` (position + staleness fallback + eventual honest None) and
`::test_metric_prediction_dynamic_occluder_respects_lateral_axis` (the
actual point of this work: a ball permanently inside the occluder's
x-range is correctly NOT occluded while the occluder is out of its lane,
and correctly IS occluded once the occluder's tracked y-position moves
into it -- exactly the case a 1D-only check gets wrong). Not yet run
through the real SAM2/DINO GPU pipeline end-to-end on rendered video
(this machine has no GPU) -- these tests validate the new interface logic
directly against controlled synthetic tracking data, the same isolation
this module's whole docstring already argues for (pure numpy, no torch/
CUDA, importable and testable without a GPU); a real-video run is
`scripts/run_gate2.py`-style follow-up work on the GPU box, not attempted
here.

**GATE 1b calibrated (2026-08).** Both `occlusion_corridor_distractor`
and `occlusion_corridor_moving` were silently falling through to the
`wrong_gravity` demonstration mutant in `scripts/run_l0_demo.py`'s
`build_demo_cases` -- a real bug, not a design gap: the scenario-name
check was an EXACT match on `"occlusion_corridor"`, so both variants
missed the "ball barely leaves the floor, use `duplicate` instead"
special-case that the base scene itself already had, and `wrong_gravity`
(which corrupts vertical motion) is a near no-op on this near-flat
geometry. Fixed by matching on `.startswith("occlusion_corridor")`
instead. Both variants now pass GATE 1a (false-termination rate 0.05,
within the generous ≤5×α bound) and GATE 1b (all five demonstration
mutants fire on the expected risk) at the state-space (L0) level --
video-level (L1/GATE 2) scoring is the dynamic-occluder work above.

**GATE 2 run for the first time (2026-08, "broaden to more scenarios and
see if it generalizes") -- found and fixed a real wiring gap, then
found and characterized (not fixed) a separate, deeper search-robustness
limitation.** `track_and_reconstruct` (the function shared by GATE 2 and
real-model scoring) never actually threaded a second tracked object
through at all -- `occluder_geometry`/`frame0_mask_secondary` support
existed in `track_with_reidentification` (M5.7 above) but nothing ever
called it with them, so this scenario's own dynamic-occluder-awareness
was silently UNREACHABLE from any real video path, neither crashing nor
visibly wrong (`occluder_bounds=None` is a valid, degraded input).
Fixed: `track_and_reconstruct` gained `frame0_mask_secondary`, resolving
`occluder_geometry` from `scene_geometry.py` whenever a camera is also
resolved (`track_with_reidentification` itself requires both together);
`scripts/run_gate2.py`'s `render_episode` now saves the occluder's own
frame-0 ground-truth mask (`secondary_gt_masks.npy`, object index 1,
only for scenarios that register `occluder_geometry`) and `remote/
modal_app.py::run_gate2_episode` reads and passes it through. 4 new
unit tests (`tests/test_track_and_reconstruct.py`, monkeypatched, no
GPU needed) verify the wiring logic itself: `occluder_geometry` is
passed through correctly when a camera resolves, forced to `None` when
it doesn't (the safety gate `track_with_reidentification` itself would
otherwise raise on), and stays `None` for every scenario that never
registered it (`occlusion_corridor_distractor`'s own inert decoy,
unaffected). `default_phi_reconstruct` (real-model scoring) deliberately
does NOT get this fix in the same pass -- documented as an explicit,
separate scope boundary in its own docstring, not silently implied fixed.

Combined with the SAME `sigma_existence` `obj=0` fix built for
`occlusion_corridor_distractor` (see that scenario's own M5.5 writeup),
GATE 2 ran cleanly enough to be genuinely informative: `velocity_freeze`
and `jitter` both fire PERFECTLY (8/8 fired, 8/8 correct; `jitter`'s own
event-time bias exactly +0.000s, matching state-space precisely) --
proof the wiring fix and the existence fix both work correctly together
on a real K=2, dynamically-occluded scene. `vanish` is partially working
(8/8 fired, 4/8 correct). `null` is the real remaining problem: 8/10
false-positive, ALL via R5 (kinematic), zero via R2 (the existence side
is clean -- confirms the `obj=0` fix, not this section's own new work,
is what's responsible).

**Root-caused with real ground truth, not guessed -- and the first,
"obvious" hypothesis was WRONG, corrected before being trusted.**
Traced the highest-churn `null` false-positive (`null_2`, 145 total
search attempts) directly: the ball is genuinely lost from frame 88 to
frame 160 (2.4 real seconds, 144 consecutive rejected re-identification
attempts, both the physics-prior and DINO tiers), then reacquires via a
low-confidence physics-prior match (6.37px reprojection error) at
EXACTLY the frame GATE 2's own R5 event fires. Initial hypothesis --
the deceleration-based extrapolation (`MetricTrackFit._x_motion`)
doesn't correctly clamp velocity to zero once the ball has actually
stopped, so its predicted position keeps drifting the longer a gap
runs -- checked directly against the actual code and found WRONG: `t_eff
= min(t, t_stop)` already clamps correctly, confirmed by reading, not
assumed. Checked against GROUND TRUTH instead (GATE 2 is a mutant
validation harness -- the true 3D trajectory is available locally by
re-rolling the same seed, zero GPU cost, even though `run_gate2_episode`
itself deliberately never returns it to keep GATE 2's own measurement
honest): the ball has come to a COMPLETE, permanent stop by frame ~91 at
world position `(6.029, 0.0, 0.15)` -- confirmed the reacquired position
at frame 160, `(6.184, -0.250, 0.15)`, is a REAL ~0.29m error, well
outside this project's own established ~0.05-0.15m reconstruction-
accuracy floor. Not a reference-ensemble-calibration artifact (a
plausible alternative explanation, checked and ruled out this way, not
assumed away) -- a genuine bad match. The remaining open question,
narrowed but not yet answered: the LATERAL-axis error (0.25m) is larger
than the motion-axis error (0.155m), which is the more surprising half,
since the lateral position is supposed to be a simple constant carried
forward from the pre-occlusion fit, never predicted to drift at all --
whether this is a pre-occlusion fit-accuracy issue (velocity estimated
imprecisely right as the ball approaches zero velocity) or something in
how the eventual match got accepted despite being far from the physics-
prediction on both axes is NOT yet determined.

**Left open, deliberately, not chased further this session** (2026-08,
confirmed directly: "document as a characterized, open gap and move
on"). A real fix needs more investigation than this session's own
remaining budget -- redesigning how much confidence a long-gap
reacquisition should be given, or fixing whatever produces the lateral-
axis error specifically, are both real, scoped follow-ups, not
one-line patches. Real-model scoring on `occlusion_corridor_moving`
should be read with this caveat in mind until it's resolved: an
occasional spuriously-high R5 rate on this scenario specifically may
reflect this measurement gap, not real model behavior -- same "know the
instrument's own error floor before trusting what it reports"
discipline as everywhere else in this document.

### M4.5 — Non-planar 3D reconstruction, scoping (2026-08)

Before generalizing `reconstruct.py` past its current flat-ground-only
limit for ramp_descent, deliberately built a SECOND new scene first, so
the generalization is designed against the full requirement set rather
than ramp_descent's case alone and needing rework later (explicit user
decision, 2026-08).

**Two requirement classes, not one, confirmed by direct analysis, not
assumed:**
1. **Known fixed plane** (ramp_descent): the ball's true 3D position is
   always confined to ONE known, fixed surface (the incline). Geometrically
   almost identical to occlusion_corridor's flat ground -- an extension of
   the exact `unproject_to_plane` ray/plane math already built, generalized
   to accept an arbitrary plane equation instead of a hardcoded `z=const`.
   Not attempted yet (task tracked separately), but this is a small,
   well-understood extension, not a new capability.
2. **True free 3D motion** (new): no fixed known plane exists at any point
   -- needs a real depth-recovery method. Rather than general monocular
   depth or multi-view triangulation, the plan is to reuse this project's
   own established philosophy (borrow known physics, don't re-derive it --
   same trick `physical_constants.py`/`motion_prior.py` already use for
   deceleration): given known gravity and an assumed ballistic model, a
   partial 2D pixel trajectory with known camera calibration can be fit to
   recover the full 3D trajectory analytically. Not yet built.

Two scenes explicitly considered and rejected as the second requirement's
test case: articulated linkage (needs real 6-DoF pose estimation, already
flagged as deferred in `reconstruct.py`'s own module docstring -- too big
a scope for this round) and container occlusion (flat-ground compatible,
doesn't need non-planar reconstruction at all -- orthogonal to this
question, not a useful design target for it).

**New scene: `scenes/projectile.xml` / `configs/manifests/projectile.yaml`.**
P4 (dynamics) -- same target property as ramp_descent, deliberately
different kinematic regime (free-flight parabolic motion + floor bounces,
vs. ramp_descent's constrained sliding-then-rolling descent), isolated per
AGENT.md 3.6. A real bug was found and fixed while tuning the bounce
physics: MuJoCo's usual `solref=(timeconst, dampratio)` contact form
produced almost NO bounce here regardless of how underdamped it was set
(swept dampratio down to 0.05, timeconst, and friction independently, all
had negligible effect) -- switching to `solref`'s NEGATIVE form (direct
stiffness/damping in N/m, N·s/m, rather than a reference time-constant)
fixed it immediately, giving a clean, clearly decaying 6-bounce arc.
Validated end-to-end through the real `EpisodeSpec`/rollout/render
pipeline: ball stays visible 90/90 frames, GATE 1a passes (false-alarm
rate 0.017 vs target ~0.01).

**GATE 1b -- now calibrated and passing (2026-08); the earlier failure's
root cause was the reference ensemble's own spread, not mutant severity.**
At default mutant severities and lam=1.0 (copied from other scenes), only
`vanish` fired; `velocity_freeze`, `wrong_gravity`, and `jitter` all
stayed censored, and this scene's calibrated R5 threshold (median 7.17)
was substantially higher than occlusion_corridor's (2.70) or
ramp_descent's (6.52).

Confirmed directly, not assumed, by decoupling the two possible causes
(mutant-signal strength vs. reference-ensemble noise): fit ONE fixed
mutant trajectory once, then re-ran ONLY the reference ensemble at
different `lam`, isolating which one the threshold actually responds to.
Root cause: `perturb_mode="full"` (isotropic, unlike occlusion_corridor's
`velocity_x_only`) means initial LAUNCH VELOCITY is perturbed too, and for
a bouncing trajectory that swings bounce-apex height and bounce COUNT
wildly across the 30-reference ensemble -- measured directly, reference
z-position std up to ~3.5m at t=1.5s, against the ball's own ~1m total
height range for a SINGLE trajectory. Whole-path R5 calibration
(thresholds.py) inflates to swallow that natural chaos, and even a fully
frozen-velocity trajectory (physics stopped outright) reads as
statistically ordinary against an envelope that wide.

Fix: `configs/manifests/projectile.yaml`'s `lam` cut from 1.0 to 0.15 --
shrinking the reference ensemble's own initial-velocity spread shrinks the
chaotic bounce-timing scatter it produces, which is what actually tightens
the threshold (confirmed: GATE 1a's false-termination rate stays 0.000 at
every `lam` tested down to 0.08, so this never risked false alarms, only
sensitivity). `velocity_freeze` and `vanish` now fire reliably at
`lam=0.15` alone; `wrong_gravity` and `jitter` needed their own severity
bumped too (factor 0.6->0.4, sigma 0.05->0.15) since this scene's R5
threshold, even after the lam fix, is still higher than occlusion_
corridor's -- the SAME pattern already used once for occlusion_corridor
itself (jitter's sigma bumped 0.05->0.15 there, GATE 2's own writeup), now
applied here for the same underlying reason. Incidentally, the exact same
jitter gap (sigma=0.05 not firing) turned out to be pre-existing on
ramp_descent too once actually checked -- fixed identically, not left
half-done. All three scenes' `scripts/run_l0_demo.py` now show every GATE
1b demonstration mutant firing on its expected risk.

**Requirement class 1 -- known fixed plane -- DONE.** `reconstruct.py`
gained `unproject_to_known_plane(px, py, ..., plane_point, plane_normal)`,
the general ray/plane intersection `unproject_to_plane(..., plane_z)` was
already a special case of (now a thin wrapper over it -- behavior-preserving,
confirmed by the full existing test suite passing unchanged). More
importantly, `reconstruct_trajectory` gained `plane_pieces`: a list of
`(plane_point, plane_normal, valid_fn)` tried in order per frame, the first
whose result satisfies its own `valid_fn` wins -- a self-consistency check
using each piece's own known geometry, not privileged knowledge of which
piece is "correct" (same status as the wall-bounds precedent, defect #17).
Needed because ramp_descent's motion isn't confined to ONE plane for the
whole clip -- it's piecewise: the incline, then flat ground once the ball
reaches the toe -- so a single-plane generalization wouldn't have been
enough on its own.

Ramp geometry cross-validated independently, not trusted from the scene's
own comment alone: computed the top-surface plane directly from the ramp
geom's own center/size/euler rotation and got edge points (-0.900, 0,
0.00007) and (-4.2856, 0, 1.80002), normal (0.46947, 0, 0.88295) -- matches
the scene comment's claimed (-0.9, 0, 0) / (-4.286, 0, 1.8) / slope
tan(28°)=0.5317 to 4-5 significant figures.

**Validated against real ramp_descent ground truth (ground-truth masks as
Phi stand-in, same convention as the flat-ground test), across a real
rollout that actually crosses both plane pieces, not just tested on the
ramp alone**: mean position error 0.11-0.13m across 5 seeds, comfortably
under the same 0.15m bound the flat-ground test uses. Per-segment
breakdown (seed=1): ~0.08-0.09m on the flat-floor segment (matches
occlusion_corridor's own flat-ground floor), ~0.23m on the ramp segment
itself (higher, as expected for a more oblique, worse-conditioned viewing
angle on a tilted surface). `tests/test_reconstruct.py::
test_ramp_descent_reconstruction_error_is_bounded`.

**A real, localized limitation found and characterized precisely, not
smoothed over by the mean bound passing anyway**: the first ~7 frames
(~0.23s) of every rollout show error decaying from ~1.0-1.3m down to the
normal floor -- because the ball is genuinely still airborne (the scene's
own "drop from 1.7m for a small settling fall" keyframe design) before it
first touches the ramp surface. The known-plane assumption is honestly
violated during this brief window -- the ball's true position isn't on
ANY known plane yet, it's in free-fall, structurally the SAME problem
M4.5's second requirement class (true free 3D motion, task below) exists
to solve. Not fixed here -- the mean-error bound already absorbs it
because it's brief, and fixing it properly means detecting "not yet on a
known plane" and falling back to the ballistic reconstruction below --
NOT done (see that section's own scoping note: the two modes are
currently mutually exclusive per call, not automatically switched
per-frame within one clip).

**Requirement class 2 -- true free 3D motion -- DONE, scoped to
whole-clip ballistic motion (not automatic per-frame mode-switching).**
`reconstruct.py` gained `fit_ballistic_trajectory(pixel_obs, times, cam_pos,
cam_mat, fovy_deg, width, height, gravity_z, ...)`: given known gravity, a
2D pixel trajectory, and camera calibration, recovers the full 3D
ballistic trajectory `(x0, v0)` by multi-start Levenberg-Marquardt
(hand-implemented, plain numpy, numerical Jacobian -- no scipy dependency,
same "no new dependency for one well-scoped problem" pattern as
`motion_prior.py`'s own solver). `reconstruct_trajectory` gained
`ballistic=True` as a third mode, mutually exclusive with `plane_z`/
`plane_pieces` (exactly one of the three required) -- unlike the plane
modes, which unproject each frame independently, ballistic mode fits ONE
global trajectory jointly across every observed frame, then evaluates it
at each frame's own time, because a single ray alone carries zero depth
information (the classic monocular scale ambiguity) -- only the whole
trajectory's parametric shape, fit jointly, is over-determined enough to
resolve it.

**A genuine, structural finding, not a bug**: position-only residuals
turned out to be insufficiently constraining on their own. Confirmed
directly on real rendered data -- a fit ~0.7m from the TRUE 3D trajectory
had a BETTER pixel-position residual (cost/n=0.166) than the true
parameters themselves (cost/n=3.355). The optimizer was correctly doing
what it was told (minimize pixel error); a genuinely different 3D
trajectory explained the same 2D pixel path equally well or better.
Position alone cannot distinguish "closer and slower" from "farther and
faster" along a near-collinear viewing ray -- an inherent monocular
limitation, not something more Levenberg-Marquardt iterations or a better
initial guess alone can fix.

**Fix: reuse the object's own known physical size as an independent depth
cue** (apparent size shrinks/grows with depth, similar triangles) --
`size_obs`/`object_radius` add a third residual term
(`predicted_radius_px = f * object_radius / depth`) AND, more
importantly, re-center the optimizer's restart-depth sweep on the
size-implied depth (`depth0 = f*object_radius/size_obs[0]`, confirmed
accurate to ~6% against ground truth) instead of blindly sweeping 2-20m.
Both parts mattered, tested independently on real rendered data across 4
seeds: the residual term alone (`size_weight=1.0`) fixed 2 of 4 seeds
(~0.08-0.10m error) but left 2 stuck in a wrong local minimum (~0.48-0.50m)
-- a bad starting depth put the optimizer in a different basin entirely,
which a residual term can't reach out of. Re-centering the restart sweep
on the size cue fixed those too. `size_weight` default set to 3.0 (not
1.0), chosen by sweeping real data: mean error 0.126m across 6 seeds vs
0.161m at weight 1.0.

**Validated against real projectile.xml ground truth** (ground-truth
masks as Phi stand-in, same convention as the other two scenes), first 20
frames (~0.67s) of each rollout, before the ball leaves this camera's
field of view: mean position error 0.10-0.16m across 6 tested seeds,
matching Requirement class 1's own validated floor (0.11-0.13m).
`tests/test_reconstruct.py::test_fit_ballistic_trajectory_recovers_known_
3d_motion` (synthetic, noiseless, isolates the optimizer/geometry from
rendering error -- recovers `(x0, v0)` to <0.05 in both position and
velocity) and `::test_projectile_ballistic_reconstruction_error_is_
bounded` (real scene, 4 seeds, <0.2m bound).

**Scope boundary, stated plainly**: this does NOT solve ramp_descent's own
airborne-transient limitation above, or any scenario that mixes known-plane
and free-3D segments within one clip -- `reconstruct_trajectory` requires
the caller to pick exactly one mode for the WHOLE call, there is no
automatic "not on a known plane right now, switch to ballistic" detection.
Doing that properly would need a per-frame or per-segment mode decision
(e.g. residual-based agreement checking, or a known launch/landing event)
that wasn't built this round -- recorded here rather than silently implied
by "requirement class 2 is DONE."

**A real gap this scope boundary predicted, actually found (2026-08,
"broaden to more scenarios and see if it generalizes"): GATE 2 on
`projectile` fails completely, root-caused, not yet fixed.** Running GATE
2 (M5) on `projectile` for the first time -- never attempted before this
-- came back 0/42 successful L1 reconstructions across EVERY mutant type,
including `null` (no defect at all): `vanish`/`velocity_freeze`/`jitter`
all fired 8/8 correctly in state-space (L0) but 0/8 through the real
video pipeline (L1). Ruled out tracking as the cause FIRST, not assumed:
`null`'s own mean mask IoU was 0.79-0.81 (SAM2 tracking is fine), yet
`position_error_n=0` for every single one of all 42 episodes -- the
reconstructed trajectory was ENTIRELY NaN, every time, regardless of
tracking quality.

**Root-caused with real diagnostics, not guessed.** Added `VITALS_RECON_
DEBUG` (env-var-gated, same convention as `reidentify.py`'s own
`VITALS_REID_DEBUG`) to `fit_ballistic_trajectory` -- prints the actual
normalized residual cost (not just pass/fail), the fitted `(x0, v0)`, and
the raw tracked pixel/size observations. `run_gate2_episode` gained a
matching `debug_recon` param (same pattern as `run_reidentification`'s
own `debug`). Re-ran ONE already-uploaded episode (`gate2_projectile_
null_0`, no re-render/re-upload needed) with it on: **normalized cost =
5368 against an accept threshold of 4.0 -- roughly 1300x over, not a
near-miss** -- with a fitted `x0=(-2.02,-7.69,-3.66)` (z=-3.66m, BELOW
the floor) and `v0_z=+14.2 m/s` (would send a real ball ~10m into the
air) -- a physically absurd local minimum, not a plausible-but-imprecise
fit.

**The actual mechanism, confirmed directly, not hypothesized:** checked
the TRUE (ground-truth) height trajectory for the exact seed/window GATE
2 observed (cheap, local, zero GPU cost -- just rolling out the same
MuJoCo episode) and found it clearly bounces MULTIPLE times within the
observed 89-frame/2.93s window (rises to z=0.95m by frame 10, falls to
z=0.19m by frame 25, bounces back up to z=0.54m, falls again, several
more smaller bounces after that) -- exactly matching `projectile.yaml`'s
own scenario description ("bouncing... several times, losing height with
each bounce"). `fit_ballistic_trajectory`'s own docstring is explicit
that it fits **ONE continuous global parabola** (`x0 + v0*t + 0.5*g*t^2`)
across every observed frame jointly -- correct for a single flight
segment, but a bounce is a real velocity discontinuity a single parabola
cannot represent AT ALL. **This exact scope boundary is why the earlier
validation above ("Validated against real projectile.xml ground truth...
first 20 frames (~0.67s)... before the ball leaves this camera's field
of view") never hit this**: those 20 frames stay entirely within the
FIRST arc, deliberately or not, before any bounce occurs. GATE 2's own
observation window (bounded by `T_RENDER`/how long the object stays
visible, not by "before the first bounce") naturally extends well past
that boundary, into a regime the function was never designed for, tested
against, or even exercised in this way before.

**Left open, not patched over.** A real fix is a genuinely new,
scoped feature -- multi-segment/piecewise ballistic fitting (detect
bounce events, fit a separate parabola per flight segment), conceptually
analogous to `plane_pieces`' existing piecewise design for
`ramp_descent`, but for velocity-discontinuous ballistic segments instead
of position-continuous plane switches -- not something to attempt as a
quick patch (e.g. loosening the >4.0 threshold would accept physically
absurd fits like the one measured above, not fix anything). `VITALS_
RECON_DEBUG` and `debug_recon` are now real, reusable diagnostic tools
for whenever this gets picked up. Real-model scoring on `projectile` is
correctly still blocked on this -- GATE 2 must pass first, same
discipline already enforced for every other scenario -- and `occlusion_
corridor_distractor`/`occlusion_corridor_moving` (both flat-ground,
`plane_z` mode, the already-validated reconstruction path) remain the
lower-risk scenarios for broadening real-model coverage in the meantime.

**Built (2026-08, "scope and build piecewise ballistic fitting now").**
`fit_piecewise_ballistic_trajectory` (`vitals/phi/reconstruct.py`) --
segment-growing changepoint detection, not a bounce-specific pixel
heuristic (no reliable geometric "this is a bounce" signal exists from
noisy 2D observations alone): grows a single-segment fit frame by frame
for as long as it stays acceptable (reusing `fit_ballistic_trajectory`'s
own `>4.0` normalized-cost threshold), and the moment extending the
window would break an otherwise-good fit, that IS the segment boundary --
starts a fresh segment (fresh restart-depth sweep, fresh (x0,v0); post-
bounce velocity is a genuinely new unknown, no restitution coefficient
assumed) from there. `_fit_ballistic_core` extracted from `fit_ballistic_
trajectory` (identical optimization logic, verified byte-identical
behavior via the existing test suite) so the piecewise search can see the
RAW cost at every candidate window size, not just a collapsed accept/
reject bit. `piecewise_ballistic=True` opt-in parameter on
`reconstruct_trajectory` -- deliberately NOT folded into `ballistic=True`
unconditionally, keeping every already-validated single-segment result
byte-identical; `projectile`'s own `scene_geometry.py` entry opts in,
nothing else does.

**A real bug caught by this function's OWN test suite before shipping,
not found in production:** the coarse growing search can overshoot the
true boundary by a few frames before finally breaking (confirmed
empirically on a planted-bounce synthetic case: normalized cost grew
0.0001 -> 0.0001 -> 0.40 -> 2.40 -> 6.34 per added contaminating frame,
comfortably under the 4.0 threshold for two frames before finally
breaking it on the third) -- fixed with a backward-refinement pass that
re-examines a small trailing range and keeps whichever nearby window has
the LOWEST cost, not just the longest one that cleared the coarse
threshold. That fix itself then broke a DIFFERENT test (`piecewise_
ballistic=True` must degenerate exactly to the single-segment result
when nothing actually bounces) -- refining unconditionally sometimes
trimmed a few frames off a perfectly clean single arc purely from
restart-optimizer numerical noise, with nothing actually contaminating
it. Root-caused and fixed properly: refinement now only runs when the
coarse search broke due to a genuine failed fit, never when it simply
ran out of observed frames. 3 new tests (`tests/test_reconstruct.py`): a
planted-bounce synthetic case (recovers each segment's own (x0,v0) to
<0.1), a no-bounce degeneration case (exactly 1 segment, matching `fit_
ballistic_trajectory`'s own output to 1e-6), and an opt-in-equivalence
case on `reconstruct_trajectory` itself. Full suite 114/114.

**Verified against real GATE 2 data, not just synthetic tests -- and it
works, with a real, now-different characterized limitation.** Re-ran the
SAME already-uploaded `projectile` episodes (no re-render needed): every
single one of 89 frames on the diagnosed episode now has a DEFINED
position (was 0/89 before this fix). Full 42-episode GATE 2 re-run:
`velocity_freeze` and `wrong_gravity` both now fire 8/8 correct.
`null`'s own false-positive rate is still 10/10 -- but for a DIFFERENT,
now-understood reason: mean reconstruction position error is ~0.8-0.9m
(vs. this project's own established ~0.05-0.15m floor on flat-ground
scenes), large enough that R5's own threshold -- calibrated against
CLEAN state-space physics variance, never against real ballistic
reconstruction noise -- fires immediately at t=0.00s on nearly every
episode regardless of mutant type. Traced the accuracy gap directly:
later segments (found via the debug segment-boundary printout,
`VITALS_RECON_DEBUG`) are progressively SHORTER (23, 20, 12, 14, 9, 11
frames on the diagnosed episode) as bounces get more frequent later in
the clip (physically sensible -- each bounce loses energy, shortening
the next flight arc) -- shorter windows are less well-constrained,
degrading accuracy for later segments specifically (error grows from
~0.23m early to ~1.2m late on the same episode, confirmed against ground
truth). This is a genuinely different, more tractable problem than the
original 0/42 total blank-out: a threshold-calibration gap (the SAME
"instrument tax" category GATE 2 exists to characterize, not a
structural failure), not chased further this pass -- left as the next,
smaller, scoped follow-up whenever real-model scoring on `projectile` is
actually attempted.

### M5.8 — Scene diversity, Phase 1 (2026-08): rigid-body collision, the first genuinely new P4 regime

Not an AGENT.md-original milestone -- started after a direct strategic
question ("what are all of the next steps... to make this a fully
functional, state of the art benchmark") led to a scoped decision:
prioritize genuine property/regime diversity over raw scenario count
("fewer, more diverse scenarios... quality over 75-100"), and this
project's own prior "future scene ideas" scoping (§11) had already
flagged rigid-body collision as the single most tractable next addition
-- MuJoCo's actual strength (real contact solving), and a direct
extension of the K>1 tracking machinery already built and validated this
session (occlusion_corridor_distractor/_moving), not a new engine or a
new measurement capability.

**`scenes/collision.xml`**: two balls (K=2, "ball"/cue at index 0,
"target" at index 1), flat floor, `contype`/`conaffinity` DELIBERATELY
left at MuJoCo's own default (1/1) for every geom -- unlike occlusion_
corridor_moving's own occluder (explicitly zeroed to guarantee it never
collides, since that scene's whole point is occlusion, not impact), this
scene's whole point IS the collision, so no bitmask tricks are needed at
all, a genuine simplification relative to every other multi-body scene
in this repo so far.

**v1 scope, decided directly before writing any scene code:** only the
cue ball is reconstructed/scored (`object index 0`, `plane_z`
reconstruction, reusing the SAME flat-ground mode `occlusion_corridor`
already has validated -- no new reconstruction capability needed at
all, unlike `projectile`'s own ballistic case). Scoring the target
ball's own post-collision motion too is a real, separate future
extension (would need the secondary object's own FULL search-and-
reidentify path wired through GATE 2/real-model scoring, not just the
position-only tracking `occlusion_corridor_moving`'s own occluder gets)
-- not attempted here, the same "single-object first, extend later"
discipline already used for P2 (K=2 distractor/moving as later
extensions) and P3 (state-space-only P3-lite before any learned video-
side relation head).

**Real physics tuning needed on the first attempt, matching this
project's own established track record for every new scene type** (not
assumed to "just work"): the cue ball's initial launch speed (3.0 m/s)
decelerated to a stop 0.15m short of actually touching the target ball
-- confirmed directly by rolling out locally and inspecting the position
trace, not guessed. Raised to 3.5 m/s; re-checked and confirmed a real
collision now occurs (target ball's own x-position, constant for the
first 36 frames, starts increasing exactly when the gap between the two
balls' centers reaches 2×radius=0.30m -- genuine, physically-caused
momentum transfer, not scripted).

**GATE 1 calibration: clean pass, most of it on the FIRST real attempt.**
`lam=1.0` (the default starting point, never tuned) already gives GATE
1a false-termination=0.017 (target ~0.01, well within the 5x tolerance)
and 4/5 GATE 1b demonstration mutants firing correctly. The one failure
(`wrong_gravity`, censored) was immediately recognizable as the SAME
symptom already root-caused for `occlusion_corridor*` (M5.5/M5.7):
`wrong_gravity` corrupts vertical motion specifically, a near no-op on
any scene where the tracked object stays close to the floor the whole
clip (both balls here sit at z~0.15 throughout). Generalized the
existing fix rather than duplicating it: `run_l0_demo.py`'s own
`startswith("occlusion_corridor")` check became a named, documented
`NEAR_FLAT_SCENARIOS = ("occlusion_corridor", "collision")` tuple, used
for the `duplicate`-vs-`wrong_gravity` branch. Re-ran: **5/5 GATE 1b,
clean, no further lam tuning needed at all** -- `vanish`/`velocity_
freeze`/`duplicate`/`jitter` (sig=0.15, already correct without
adjustment) all fire on the expected risk. Re-verified `occlusion_
corridor`'s own GATE 1a/1b immediately after, byte-identical to its own
previously-published numbers (VI_50=4.77s, CI[3.27,5.50]) -- confirms
the generalization introduced zero regression. Full suite 114/114.

Registered in `vitals/phi/scene_geometry.py` (`COLLISION_CAM`, `plane_z`
mode) for the eventual GATE 2 pass -- camera framing confirmed directly
(both balls' own segmentation pixel counts checked across the whole
clip, not just endpoints: consistently ~72-74px each, never zero) rather
than assumed from the XML alone. **Not yet run through GATE 2 or real-
model scoring** -- GATE 1 (state-space) is done; the real-video
validation pass (this session's own track record: 3/3 new scenarios
attempted this way each surfaced real, non-trivial measurement-pipeline
work) is the natural next step, not attempted in the same pass as
standing the scenario up for the first time.

### M6 — Models (week 4)

Implement `WorldModel` adapters for 3–4 open-weight image-to-video models.
Fixed input-normalization protocol: resolution, frame rate, prefix length,
sampling/seed policy, conditioning modality — published, and models compared
only within matched settings.

**Done when:** the full baseline ladder runs and **GATE 3** passes —
`ConstantVelocity` shows a long validity interval in free flight and a short
one under contact. If it doesn't, the metric is wrong, not the baseline.

**Input-normalization protocol — DRAFTED, frozen pending sign-off before the
first real model runs (2026-08, per §3.12: decide once, do not adjust
per-model later).** Every field below is either already an established
project convention (matched to what GATE 1/GATE 2 already use, so a real
model's results are directly comparable to the populations already
measured) or a new, explicit decision made here for the first time.

1. **Conditioning modality: RGB pixels only.** No privileged `Trajectory`
   state, no camera intrinsics, no segmentation/depth handed to the model
   even if its own API offers to accept them. This is 3.1 (measurement
   symmetry) applied to the INPUT side, not just the output side: the
   reference/mutant path Phi already scores also only ever sees rendered
   RGB, never ground truth. A model that got privileged state as
   conditioning would not be measuring the same thing GATE 1/GATE 2 do.
2. **Resolution: 320x240**, matching every scene/test in this repo
   (`MujocoRenderer`'s own convention throughout). If a model's native
   training resolution differs, resize ONLY the frames handed to the
   model (letterbox, not crop, to avoid silently changing field of view)
   and resize its OUTPUT back to 320x240 before Phi ever sees it -- the
   same resize policy applied identically to every model, never tuned
   per-model.
3. **Frame rate: 30fps**, matching every manifest. Same resample-in/
   resample-out rule as resolution if a model's native rate differs --
   Phi and the calibrated thresholds are built against this project's own
   30fps clock and must never be handed anything else.
4. **Prefix length: 1.0s (30 frames), fixed globally, not scaled per
   scenario's own horizon.** Grounded in an existing project constant, not
   picked arbitrarily: `motion_prior.MIN_FIT_FRAMES=5` is the documented
   floor below which even a simple linear kinematic fit refuses to trust
   itself -- 30 frames is 6x that floor, comfortably enough to establish
   motion (direction, speed, whether contact/bounce already happened) for
   a model with far more capacity than a linear fit, while still leaving
   most of each scenario's own horizon for the model's OWN generation to
   be tested. Fixed in ABSOLUTE seconds, not a fraction of each scenario's
   horizon_s, so every model gets the identical amount of help regardless
   of scenario.
5. **Scored horizon: each scenario's own already-frozen `horizon_s` MINUS
   the 1.0s prefix** (e.g. occlusion_corridor: 8.0s total, 1.0s prefix,
   7.0s generated and scored) -- total clip length matches exactly what
   GATE 1/GATE 2's own populations already use for that scenario, so a
   real model's survival curve sits on the same clock as the curves
   already in `results/gate1_survival.html`.
6. **Sampling: n_samples=1 per episode for survival/event scoring.** The
   `WorldModel` protocol already supports n_samples>1 (`adapters/base.py`'s
   own docstring), but pooling multiple correlated samples from one
   episode into the event/survival pipeline would violate 3.10 (bootstrap
   over episodes, never over rollouts) -- n_samples>1 is reserved for the
   not-yet-built P6 distributional scoring (M7), a different pipeline, not
   this one. For a stochastic model, its OWN internal generation seed must
   be a deterministic function of the episode's own seed (same discipline
   already forced onto `reidentify.py`'s own torch/cudnn determinism
   pinning, AGENT.md defect #19) -- a rerun of the identical episode must
   reach the identical verdict.
7. **Held-out prefix source: ONE fixed set of held-out clean rollouts per
   scenario, drawn once, reused identically across every model tested on
   that scenario** -- never the calibration reference ensemble (3.7).
   Reusing the identical set (not a fresh draw per model) is what makes
   cross-model comparisons PAIRED (`bootstrap_vi_delta`'s own docstring
   already names this as the strictly more powerful version of itself
   that isn't built yet) rather than independently-sampled populations
   compared bootstrap-vs-bootstrap.
8. **Render realism, logged as an explicit graded covariate on every
   result -- not left implicit (T2's own mitigation).** `MujocoRenderer.
   render()`'s `realism` parameter (currently mostly a stub -- only
   toggles shadows) gets recorded in every result file alongside the
   model's own name, so a future realism upgrade can never retroactively
   confuse which results were measured under which rendering fidelity.
   This does not resolve T2 -- it keeps the project honest about it,
   exactly the difference §12 draws between "we built the measuring
   device" and "here is the leaderboard."
9. **Model output: raw RGB frames only**, symmetric with input -- Phi
   (segmentation -> reidentify -> reconstruct) runs on a real model's
   generated video with the SAME zero-privileged-access discipline it
   already runs under for the reference/mutant path. A model's own API
   offering "helpful" masks/depth/boxes alongside its pixels must be
   ignored, not accepted -- accepting them would make L1 measurement
   easier for that one model in a way no other model (or the reference
   path itself) gets.
10. **Camera: identical (position, orientation, fovy) to whatever rendered
    that scenario's own prefix and reference ensemble, held fixed for
    scoring -- never told to the model explicitly.** Mirrors real
    deployment (a video model is never hand-fed camera intrinsics as a
    separate input, only pixels); Phi still needs the true calibration to
    reconstruct 3D afterward, same "known instrument calibration, not
    privileged access to the tracked object's own state" status every
    other use of camera parameters in this repo already has.

**Real-model adapter is more than wrapping one API call -- worth knowing
before scoping the work, not discovered mid-implementation.** The existing
`WorldModel` protocol (`adapters/base.py`) is `Trajectory -> Trajectory`,
which is exactly right for the analytic baselines (`CopyLastState`,
`ConstantVelocity`) but NOT what a real video model natively speaks. A real
adapter's `predict()` must internally: render its `conditioning` Trajectory
to RGB frames (protocol items 1-3 above) -> call the model's own video-
generation inference -> run the returned frames through the SAME Phi
pipeline already validated this session (segmentation -> reidentify ->
`reconstruct_trajectory`, whichever of flat-ground/known-plane/ballistic
fits the scenario) -> return the reconstructed Trajectory. The `WorldModel`
interface itself does not need to change; the adapter implementation is
real, non-trivial glue code, not a thin wrapper.

**Per-scenario reconstruction geometry consolidated into one canonical
module, `vitals/phi/scene_geometry.py` (2026-08, requested directly: "one
adapter with functions per scene, or is that not possible").** The answer
was yes, it's possible and is the right shape -- ONE adapter per real
model, parameterized by scene-specific DATA, not one adapter class per
scene. But that only actually holds if the scene-specific half
(reconstruction: which plane mode, which camera, which object radius,
which occluder geometry) is centralized as data rather than duplicated at
every call site -- confirmed it genuinely was NOT, before this: camera
dicts, `BALL_RADIUS`, and `RAMP_PLANE_PIECES` were independently copied
across `tests/test_reconstruct.py` and `scripts/run_gate2.py`, and a real,
substantive drift had already happened silently -- `remote/modal_app.py`'s
own `OCCLUSION_CORRIDOR_WALL_BOUNDS` was the EMPIRICALLY-CORRECTED value
`(4.39, 5.87)` (measured via a 5mm-resolution visibility sweep, see that
module's own history), while `motion_prior.py`'s own docstring still
quoted the naive geometric-box value `(4.25, 5.75)` as if it were the same
number. Both were "the wall's occluder bounds" for the identical scene;
only one was right.

`scene_geometry.py`'s `SCENES` dict now holds one entry per scenario
(camera, reconstruction mode + kwargs matched to `reconstruct_trajectory`'s
own XOR contract, object radius, occluder geometry where relevant) and
`get(name)` raises rather than guessing for an unregistered scenario --
this is the literal shape a real adapter's `predict()` would use:
`cfg = scene_geometry.get(scenario_name)` before calling
`reconstruct_trajectory(**cfg["reconstruct_kwargs"])`, no per-scene
branching in the adapter itself. `tests/test_reconstruct.py` and
`scripts/run_gate2.py` now import from it instead of keeping their own
copies; `remote/modal_app.py`'s three GPU-function bodies do too (matching
that file's own established per-function `sys.path.insert` + import
pattern, not a new top-level import, since that module's top level runs
locally at deploy time and its own convention deliberately keeps vitals
imports out of it). New: `tests/test_scene_geometry.py` -- validates every
registered scenario's config actually satisfies `reconstruct_trajectory`'s
mode contract, and a dedicated regression guard that the wall-bounds value
can never silently regress back to the naive one that caused the GATE 2
null false-positive mechanism in the first place.

**Real-model adapter scaffold built, `vitals/adapters/video_model.py`
(2026-08, explicit user choice: build the model-agnostic scaffold first,
no specific model/API chosen yet).** `VideoWorldModel` implements the
existing `WorldModel` protocol unchanged (`predict(conditioning, horizon_s,
n_samples) -> list[Trajectory]`), structured around the two genuinely
different halves this milestone's own protocol draft named: rendering +
calling the model (scene-agnostic, pixels only) and reconstructing its
output (scene-specific, needs `scene_geometry`'s per-scenario config).
Two independently swappable injection points:

- `generate_fn(prefix_frames, n_frames) -> continuation_frames`: THE REAL
  MODEL plugs in here. No default -- there is no generic stand-in for "a
  real video model." Contract: given the rendered prefix and a frame
  count, return exactly that many new RGB frames, nothing else (protocol
  item 1 -- pixels only).
- `phi_fn(...) -> Trajectory`: defaults to `default_phi_reconstruct`, a
  REAL (not stubbed) implementation of the production path -- writes
  frames to a temp dir, calls the actual `reidentify.
  track_with_reidentification` (SAM2 + DINOv2, needs GPU) prompted with
  ONLY the prefix's own true frame-0 mask (`render/__init__.py`'s own
  documented limit on what Phi is ever allowed to see), then
  `reconstruct.reconstruct_trajectory` using `scene_geometry`'s config.
  Stated plainly, not hidden: `fit_metric_track`'s physics-prior
  re-identification tier only understands a single flat plane today, so
  for `ramp_descent` (piecewise plane) and `projectile` (no plane at all)
  this correctly falls back to the pixel-only `PixelTrackFit` re-id tier
  (a real, already-tested, just less accurate path) rather than silently
  mishandling or crashing on scenarios physics-prior re-id was never
  extended to cover.

**Validated end to end on this GPU-less machine, honestly scoped**: with
`phi_fn` overridden to a ground-truth-mask stand-in (same isolation
convention as `tests/test_reconstruct.py`) and `generate_fn` overridden to
a "perfect" stub that re-renders the TRUE continuation (a round-trip check
of the adapter's OWN new glue code -- render, stitch, reconstruct, slice
back to a continuation with time re-zeroed -- not a claim about any real
model), `predict()` recovers the true continuation to <0.15m mean position
error, matching every other reconstruction-accuracy floor already
published this session. Also covers the wrong-frame-count contract
violation (must raise, not silently mis-stitch) and confirms construction
alone -- even with the REAL default `phi_fn`, which imports torch/SAM2
only inside a lazily-evaluated closure -- needs no GPU, only calling
`predict()` for real does. `default_phi_reconstruct` itself has NOT been
run against a live SAM2/DINOv2 pipeline this session (no GPU on this
machine) -- that, plus actually wiring a real model's own `generate_fn`,
remain the concrete next steps once a specific model is chosen.

**Model chosen: NVIDIA Cosmos-Predict2 (Video2World), 2026-08.**
`vitals/adapters/cosmos.py` builds a `generate_fn` closure for
`VideoWorldModel` against Cosmos-Predict2's own published repo
(github.com/nvidia-cosmos/cosmos-predict2) -- real code written directly
against its documented CLI interface, not yet executed against a live
install (this machine has no GPU; Cosmos needs Ampere-or-newer, CUDA 12.6,
and its own checkpoint download). Facts this design is built on, pulled
from Cosmos's own docs, not assumed:

- Cosmos's documented entry point is a CLI (`python -m examples.
  video2world`) -- an internal `Video2WorldPipeline` class exists but its
  call signature isn't in Cosmos's own published docs, so this wraps the
  CLI via `subprocess`, the more stable documented surface.
- Video2World is NOT purely "frames in, frames out" -- it also requires a
  text prompt. `SCENARIO_PROMPTS` freezes one plain, physically-
  descriptive prompt per scenario (input-normalization protocol
  discipline: decided once, never adjusted per episode).
- "Multi-frame conditioning" only uses the LAST 5 frames of whatever video
  file it's handed -- this project's own 30-frame (1.0s) prefix (protocol
  item 4) needs no change for this model specifically; Cosmos just uses
  the tail of it.
- Native output is 16fps, 480p/720p, NOT this project's 30fps/320x320 --
  `_resample_to_target` handles both directions (fps resample + resize),
  same discipline the protocol already specifies for any model whose
  native format differs (items 2-3), applied here for real, not just
  described.
- One genuine UNRESOLVED question, stated plainly rather than guessed:
  Cosmos's own docs do not say how many frames one `video2world` call
  actually returns, or whether reaching a full `horizon_s` needs one call
  or several chained ones. `_resample_to_target` trims or hold-last-frame
  pads to the requested count as an honest stand-in -- confirmed this
  actually triggers correctly on synthetic data (see below), but this is
  NOT a claim that padding is a real generative capability; it exists so
  `VideoWorldModel`'s own frame-count contract doesn't break while this
  stays unconfirmed. Resolving it for real needs one live Cosmos run.

**Validated: everything that doesn't need Cosmos itself or a GPU**
(`tests/test_cosmos_adapter.py`, needs the new optional `cosmos` dependency
group -- `imageio[ffmpeg]`, added to `pyproject.toml`, the one deliberate
exception to this project's "never compressed video" rule, `render/
__init__.py`'s own `Frames` docstring -- since Cosmos's documented
interface requires an actual video file, not raw arrays): the video file
write/read round trip preserves shape, resampling preserves temporal order
(a moving marker keeps moving the same direction after fps/resolution
conversion, not just a shape check), and the hold-last-frame padding
behavior actually engages when requested duration exceeds source content.
`generate_fn` itself (the actual `subprocess` call into a real Cosmos
checkout) is NOT covered -- untestable without a live GPU install, the
same honest boundary `default_phi_reconstruct` already has.

**NOT done, and NOT attempted without explicit go-ahead first**:
provisioning a GPU environment for Cosmos (extending `remote/modal_app.py`
the way SAM2/DINOv2 already run there, or another host), downloading its
checkpoint (Cosmos's own docs: ~250GB for its FULL checkpoint set across
every model size/variant, not this one model alone -- exact 2B Video2World
size not found in its own docs), and the real GPU time to run it. These
have real cost and provisioning implications this document does not
decide unilaterally -- see AGENT.md's own build-sequence notes for the
open question.

**Model-agnosticism verified directly, not just argued (2026-08, before
committing GPU/checkpoint cost to Cosmos: "the adapter and this evaluation
suite should be model-agnostic, is there no way to do that").** Confirmed
by grep: zero mentions of Cosmos anywhere outside `cosmos.py` itself --
`VideoWorldModel`, `scene_geometry.py`, and everything downstream of
`predict()` have no idea Cosmos exists. To demonstrate the claim rather
than assert it, built a SECOND `generate_fn`,
`vitals/adapters/pixel_constant_velocity.py` -- deliberately as unlike
`cosmos.py`'s own implementation as reasonably possible: pure numpy,
in-process, no subprocess, no video files, no per-scenario config of any
kind (vs. `cosmos.py`'s subprocess + video-file I/O + frozen per-scenario
text prompts). It is also a real, reusable artifact, not a throwaway test
double: the pixel-space analog of `adapters/base.py`'s own
`ConstantVelocity` baseline -- locates the tracked object by color
threshold in the raw RGB pixels it's given (no privileged state, no
segmentation, matching protocol rule 1) and linearly extrapolates its
PIXEL velocity forward, falling back to a CopyLastState-style repeat when
the object isn't detectable at all. Same diagnostic role GATE 3 already
established for its state-space counterpart: expected to look reasonable
while true motion stays close to constant pixel velocity, and expected to
fall apart the moment it doesn't (occlusion, a bounce, a ramp transition)
-- a short validity interval there is correct, not a bug.

`tests/test_pixel_constant_velocity_adapter.py::
test_two_different_generate_fn_implementations_both_work_through_the_same_
adapter` is the actual proof: constructs `VideoWorldModel` with THIS
generate_fn (ground-truth-mask `phi_fn` stand-in, same isolation
convention as every other Phi test this session) on a real occlusion_
corridor episode, and confirms it runs end to end -- correct output shape,
time re-zeroed, and a sensible (if less accurate than the "perfect" stub
used for the Cosmos-adapter's own round-trip test) position estimate for
the frames immediately following the prefix -- with ZERO changes to
`VideoWorldModel`, `default_phi_reconstruct`, or anything else in the
pipeline. Two structurally unrelated `generate_fn` implementations both
working through the identical, unmodified adapter is what "model-agnostic"
means here, demonstrated rather than asserted. Full 11-file test suite
still passes.

**Cosmos GPU provisioning, real code written, deployment BLOCKED on two
things only the account owner can do (2026-08, "let's move on Cosmos
provisioning now").** New: `remote/modal_app_cosmos.py` + `remote/
call_cosmos.py`, mirroring `modal_app.py`/`call.py`'s own established
pattern exactly (persistent deployed app, `modal.Function.from_name(...)
.remote(...)`, never `modal run` -- see that file's own docstring for
why). A SEPARATE Modal app from `vitals-phi`, deliberately -- Cosmos needs
its own CUDA 12.6/torch stack, likely incompatible with `modal_app.py`'s
own pin (chosen for SAM2's needs specifically). `vitals/adapters/cosmos.py`
gained `make_cosmos_generate_fn_modal` (the RECOMMENDED path -- calls the
deployed app, matching how every other real GPU call in this project
already works) alongside the original local-subprocess builder, renamed
`make_cosmos_generate_fn_local` for clarity (kept for a hypothetical
future machine with a local Cosmos+GPU install, not the path anything in
this project actually uses today).

Two facts, pulled directly from Cosmos's own Hugging Face model card
(huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World), that changed the
design from what was assumed before checking:
- **32.54GB VRAM for the 2B model alone** -- past `modal_app.py`'s own
  A10G (24GB) entirely, a hard non-starter, not a "probably fine" guess.
  `modal_app_cosmos.py` uses A100-80GB instead (real headroom above the
  base figure; CUDA context + generation-time activation memory aren't
  included in it). GPU tier is a real cost/turnaround tradeoff here, not
  a minor knob -- Cosmos's own published benchmarks show an 11x spread
  between GPU tiers for the identical call (H100 SXM ~229s vs. L40S
  ~2567s at 720p/16fps).
- **The model is GATED.** Confirmed directly on its own HF page: "agree
  to share your contact information to access this model." This blocks
  everything downstream of it -- no token, however valid otherwise, gets
  past a gate that hasn't been accepted.

**NOT deployed, NOT downloaded, NOT run -- stopped here on purpose, not
because of missing code.** Two things stand between this and a real first
call, and only the account owner can do either:
1. Visit the model's HF page while logged in and accept its gated-access
   terms -- a one-time human action on Hugging Face's own site, nothing
   here can script around it.
2. Create the Modal secret `download_checkpoints`/`video2world` both
   require: `modal secret create vitals-cosmos HF_TOKEN=hf_...`,
   using that same account's token.
Once both are done: `modal deploy remote/modal_app_cosmos.py`, then
`python3 remote/call_cosmos.py download_checkpoints` (downloads ONLY the
2B Video2World checkpoint, not Cosmos's full ~250GB set across every
size/variant -- confirmed via `download_checkpoints.py`'s own
`--model_sizes 2B --model_types video2world` flags), then a real
`video2world` call. Treat the FIRST deploy as a real debugging pass, not
a guaranteed clean run -- the image's own dependency install
(`uv pip install --system -e '.[cu126]'`, following Cosmos's own setup.md)
has not been verified to succeed inside this exact base image, the same
honest caveat `modal_app.py`'s own SAM2 image needed one real fix for
before it worked.

**FIRST REAL MODEL RESULT, end to end (2026-08).** Provisioning went
live: HF gated access accepted, `vitals-cosmos` deployed, checkpoint
downloaded. The deploy needed two real fixes past what was written
blind, both found by actually running it, not by more reading:
1. `debian_slim` can't satisfy `transformer_engine`'s need for the real
   CUDA toolkit's `libnvrtc` at import time (torch's own pip-bundled CUDA
   runtime isn't the same thing) -- fixed by building from an NVIDIA CUDA
   `devel` base image instead (Modal's own documented pattern for this
   exact situation).
2. Cosmos's own downloaded Guardrail1 checkpoint is missing its
   `blocklist/whitelist` subdirectory (a gap in NVIDIA's own packaging,
   confirmed via `modal volume ls`, not this project's bug) -- fixed with
   `--disable_guardrail`. Also added `--disable_prompt_refiner` for a
   real, separate reason, not just avoiding another failure surface:
   letting another model silently rewrite this project's own frozen,
   pre-registered `SCENARIO_PROMPTS` would undermine the input-
   normalization protocol's own discipline (3.12).

Also corrected before any of this ran: the GPU tier. Originally chosen as
A100-80GB purely for VRAM headroom, with no actual cost comparison done
("wait why aren't we using a h100 gpu for this?"). Checked Modal's real
per-second pricing (A100-80GB $0.000694/s vs. H100 $0.001097/s -- H100
costs 58% more per second) against Cosmos's own published H100 benchmark
(~229s per 720p/16fps call) and a conservative 2x Ampere-vs-Hopper
slowdown estimate for A100: H100 comes out CHEAPER per call (~$0.25 vs.
~$0.32) and meaningfully faster wall-clock, with equal or better VRAM
headroom -- switched to H100, worse on no axis actually compared.

**First real `video2world` call, and what it showed.** One episode
(occlusion_corridor, seed=1): the raw generated video LOOKED nearly
static by simple whole-frame pixel-difference stats (~0.24/255 mean
frame-to-frame change) -- but that read turned out to be misleading, not
a real finding, confirmed by actually measuring it properly rather than
trusting the eyeball check. Ran the SAME output through the real Phi
pipeline (`default_phi_reconstruct`, newly wired as its own callable
Modal function, `run_phi_reconstruct_from_frames` on the `vitals-phi`
app, since that function was real production code but had never itself
been invoked from anywhere GPU-less before) -- SAM2 successfully tracked
the ball through 48/60 frames, and it HAD moved (1.49m), just
substantially less than true physics (2.43m over the same window, ~61%
of the true distance), with mean position error 0.92m against the known
true continuation -- 6-18x this project's own established ~0.05-0.15m
reconstruction-accuracy floor, so this gap is real model behavior, not
measurement noise. Whole-frame pixel stats undersold this because most
of any frame is flat, unchanging background -- exactly why this project
measures through reconstruction, not pixels.

**A real, small population (n=5, illustrative not powered), scored
through the EXACT SAME calibrated detector machinery as every other
occlusion_corridor curve.** `scripts/run_cosmos_population.py`: prefix
rendered locally -> real Cosmos call -> real Phi reconstruction -> stitch
(`adapters.base.concat_trajectory`, the same function the baselines use)
-> scored with `sigma_existence`/`sigma_kinematic` against the SAME
30-reference ensemble and calibrated thresholds `results/l0_demo_
occlusion_corridor_mujoco.json` already uses. Hit one real bug on the
first attempt (found and fixed BEFORE it burned through more than one
episode's GPU cost): `extract_event` needs `theta` sliced to the
candidate's own (shorter) length, or the comparison doesn't broadcast --
`scripts/run_gate2.py` had already solved this identical problem for its
own truncated candidates; reused that fix rather than rediscovering it.

Result, all 5 episodes: **VI50 = 1.43s, 95% CI [1.0, 1.63], censoring =
0%, termination profile 100% R5.** Every episode failed, always via the
kinematic statistic, always early (median 1.43s into a ~3s clip) --
consistent with the single-episode finding above (a real, reliable
undershoot of true motion), not a fluke: 5/5, not 1/1.

Written to `results/l0_demo_occlusion_corridor_cosmos.json` in the SAME
schema every other population uses, so it appears automatically in
`render_survival_report.py`'s output -- the first entry there that isn't
ground-truth-only. Two small template fixes needed for this to render
honestly: `gate_1a.passed` is `null` (not applicable) for a real-model
population, not `false` -- the template previously collapsed that into
"FAIL," actively misleading rather than merely missing; now renders
"N/A" distinctly. Series labels now say "REAL MODEL" for anything with
`backend != "mujoco"`, so a real-model curve is never visually
indistinguishable from a ground-truth validation curve at a glance.

**Stated plainly, per this project's own §12 framing**: n=5 is
illustrative, not a powered result -- this is "the instrument works end
to end and produced a real, consistent, quantitative measurement on a
first real model," not "Cosmos's true validity interval on this
scenario is 1.43s." A real claim needs the same scale of population
already used for the synthetic-mutant curves (n=80), which is real,
additional GPU cost, not yet spent.

**Sharper analysis of the n=5 result, and two new local-only viewing
tools built directly from it (2026-08).** Pulled the raw per-episode
event times rather than trusting the summary stats alone: prefix ends at
t=1.0s, and EVERY episode fires within 0.63s of the model's own generated
content starting -- 2 of 5 fire at literally t=1.0s, the first possible
instant. Not "drifts over time," but "never establishes correct velocity
at the handoff point." Real, unisolated confounds stated plainly:
guardrail/prompt-refiner disabled, 480p, 5-frame conditioning, one
scenario/one prompt -- any could be contributing, not yet separated out.

**`scripts/plot_survival_native.py`** -- a non-browser alternative to
`render_survival_report.py`'s HTML output, requested directly ("i would
rather a plotting tool without touching a browser at all"). Native
matplotlib GUI window (this machine's own backend: macosx, confirmed not
a browser), same data/conventions as the HTML version (fixed categorical
colors, solid-vs-dashed-past-censoring, "REAL MODEL" labeling) -- hover
tracing reimplemented as a `motion_notify_event` handler instead of JS,
same "exact value at any instant" behavior. Real, honest limitation found
running it FROM this session's own sandboxed tool execution: launching it
through that path opens no persistent visible window (the process exits
without a window ever appearing on screen) -- confirmed by checking `ps`
after launch, process gone. The script itself is correct (verified via a
headless Agg-backend smoke test -- all data loads, all curves build, no
runtime errors); it needs to be run directly in a real terminal, not
through this tool's own sandboxed shell, to actually open and hold a
window.

**`scripts/render_annotated_comparison_video.py`** -- the video
visualizer ("is there a visualizer that produces the videos? ... build
the video visualizer"). Produces a real local mp4 (no browser, no HTML)
with two markers drawn directly on the actual stitched (prefix +
generated continuation) frames: green = true physics position, red =
Phi's reconstructed position from the generated video -- makes the
population's own 0.92m/61%-displacement finding watchable, not just a
number. Costs exactly ONE real GPU call (Phi reconstruction) when a
cached prefix+continuation pair already exists locally -- does NOT call
Cosmos again; saves reconstructed positions to a local `.npz` so
re-annotating (different marker style, etc.) never re-pays for GPU.
First real output: `results/videos/occlusion_corridor_cosmos_seed1_
annotated.mp4` -- both markers confirmed actually present in the pixels
(not just theoretically drawn), checked directly rather than assumed.

**Baselines discovered to already be genuinely comparable to Cosmos --
just never persisted (2026-08: "can you not compare it against all the
ground truth stuff that we did before").** `scripts/run_eval.py` already
computed the full survival stats (VI50, bootstrap CI, termination
profile, Kaplan-Meier) for `ConstantVelocity`/`CopyLastState` through the
IDENTICAL `WorldModel.predict()` pipeline Cosmos uses -- it just printed
to console and never saved a results JSON, so the report's own
auto-discovery never found it. Fixed (same persistence pattern `run_l0_
demo.py` already established). This surfaced an important terminology
distinction worth being precise about: **Truth ≠ Baseline.** Truth
(`backend="mujoco"`) is real physics deliberately, artificially corrupted
with a KNOWN planted defect -- no model or prediction involved at all,
exists to validate the DETECTOR. Baseline (`backend=<adapter name>`) is a
genuine, if simple, PREDICTION from a real conditioning prefix, through
the same interface Cosmos uses -- differs from Cosmos only in how
sophisticated the predictor is, not in kind. Comparing Cosmos to Truth is
not really meaningful (different populations answering different
questions); comparing Cosmos to Baseline is exactly the right comparison,
and was sitting unused the whole time.

Report's own color/label scheme reworked to make this distinction visible
at a glance (2026-08, requested directly): every series gets a `(T)` /
`(C)` / `(B)` prefix by category, each category its own sequential ramp
(red / blue / green respectively) rather than one shared cycled
categorical palette -- multiple curves within a category get different
SHADES, not different hues, so "which category" reads instantly and
"which specific curve within it" still stays distinguishable.

**Real comparison, n=40 baselines vs. n=5 Cosmos (first pass):**
ConstantVelocity VI50=2.33s, CopyLastState VI50=1.67s, Cosmos VI50=1.43s
-- Cosmos already looked WORSE (shorter validity interval) than both
trivial baselines, `bootstrap_vi_delta` confirmed the ConstantVelocity gap
was real (-0.90s, CI [-1.53,-0.63], nowhere near zero) but the
CopyLastState gap was borderline (-0.23s, CI [-0.97,-0.03], barely below
zero) -- exactly the comparison n=5 wasn't powered enough to fully trust.

**Cosmos scaled to n=80 to resolve that borderline case.** `run_cosmos_
population.py`'s own episode loop was sequential (160 real GPU calls one
at a time) -- reworked to run episodes concurrently via a
`ThreadPoolExecutor` (max_workers=8 by default, deliberately modest since
this is a shared team Modal workspace, not e.g. 80) -- both Cosmos and
Phi calls are network-bound, safe across threads, and each episode
already constructs its own fresh renderer/backend/temp files (verified
before parallelizing, not assumed). Real result, all 80 episodes
completed cleanly: **VI50=1.40s, 95% CI [1.33, 1.43], censoring=0%** --
the median barely moved from the n=5 estimate (1.43s), but the CI
collapsed from 0.63s wide to 0.10s wide. The termination profile also
changed in a way n=5 couldn't have shown: 1 of 80 episodes (1.25%) failed
via R2 (existence), not R5 -- rare enough that a 5-episode sample could
easily have missed it entirely, a real example of why sample size matters
for characterizing a FULL failure-mode distribution, not just a median.

Re-ran `bootstrap_vi_delta` with the n=80 population: **Cosmos vs.
ConstantVelocity -0.93s, CI [-1.23,-0.80]; Cosmos vs. CopyLastState
-0.27s, CI [-0.73,-0.17].** Both now solidly below zero -- the previously
borderline CopyLastState comparison is confirmed real, not noise. This is
now the project's first STATISTICALLY POWERED real-model finding, not
illustrative: on occlusion_corridor, Cosmos-Predict2-2B-Video2World's
generated continuations become physically implausible faster than even a
"do nothing" baseline, under this specific (guardrail/prompt-refiner
disabled, 480p, 5-frame-conditioning) calling convention.

**Second real model integrated: Wan2.1 (Alibaba), Image2Video-14B-480P
(2026-08, "start on Wan2.1").** Chosen over other open-weight candidates
after actually verifying claims rather than trusting a curated list's own
framing -- ruled out V-JEPA 2 directly (confirmed via its own repo: it
predicts future LATENT representations, not pixels, so it can't satisfy
`generate_fn`'s "pixels in, pixels out" contract at all, a real
disqualification found by checking, not assumed) and the game-focused
entries (Oasis, Matrix-Game, Hunyuan-GameCraft -- likely poor domain match
for generic rigid-body physics, not verified further since the mismatch
was clear on its face). Wan2.1 confirmed directly against its own model
card and diffusers' own docs before writing any code: NOT gated (no HF
token/Modal secret needed, unlike Cosmos), Apache 2.0, native `diffusers`
Python class (`WanImageToVideoPipeline`) instead of a CLI wrapped in
`subprocess`.

**Refactored the shared pieces out of `cosmos.py` before writing a second
model against them** -- `SCENARIO_PROMPTS` moved to `vitals/adapters/
scenario_prompts.py`, `resample_to_target`/`_write_video`/`_read_video`
moved to `vitals/adapters/video_utils.py`, both imported (not
reimplemented) by `cosmos.py` and the new `wan.py`. Not optional
tidiness: two models needing the IDENTICAL prompt wording for a fair
comparison, maintained as two independent copies, is exactly the kind of
drift this project has already been burned by once this session
(`scene_geometry.py`'s own wall-bounds consolidation writeup) --
`tests/test_wan_adapter.py::test_wan_and_cosmos_share_the_identical_
prompt_text` asserts object identity, not just equal values, so the two
can never silently diverge again.

**Real, structural differences from Cosmos, found by actually building
this, not assumed going in:**
- Single-IMAGE conditioning only (`WanImageToVideoPipeline.__call__`'s
  own signature), not Cosmos's 5-frame option -- only the prefix's LAST
  frame actually reaches Wan2.1, stated plainly in `wan.py`'s own
  docstring, not hidden.
- Output comes back as a raw numpy array (`output_type="np"`) directly
  from the Python call -- no video file round trip needed at all for this
  model, unlike Cosmos's CLI/file-based interface.
- No CUDA "devel" base image needed -- confirmed directly: modern
  `torch`/`diffusers` pip wheels bundle their own CUDA toolkit
  dependencies (`nvidia-cuda-nvrtc` etc. installed automatically),
  `debian_slim` sufficed on the first image build.

**Real bugs found and fixed on the first real calls, same discipline as
every other model integration this session:**
1. `from_pretrained(..., dtype=...)` -- wrong keyword for this installed
   diffusers version (0.39.0); the correct name is `torch_dtype`. Fast,
   cheap fix (found before any GPU time was spent generating anything).
2. **The model card's own "~16GB VRAM" figure was wrong for this actual
   pipeline** -- confirmed by a real OOM, not guessed: loading the full
   pipeline (CLIP image encoder + VAE in fp32, the 14B transformer in
   bf16, nothing offloaded) onto an L4 (24GB) alone consumed ~22GB before
   a single inference step ran. Moved straight to A100-80GB (already a
   known-working tier from the Cosmos integration) rather than guess a
   second, narrower tier and risk a third debug cycle -- worked
   immediately.

**First real generation confirmed genuinely valid, not just non-empty
bytes** (same verification discipline as Cosmos's own first output):
81 frames, 544x720 (the aspect-ratio-preserving resize correctly
preserved this project's own ~4:3 input ratio rather than distorting to
Wan's native 16:9 default), full pixel dynamic range, real frame-to-frame
motion (mean abs diff 3.4, not a static/degenerate output).
`results/videos/occlusion_corridor_wan_seed1_raw.mp4`.

**Wan run through Phi and scored (2026-08).** `run_phi_reconstruct_from_
frames` (the remote `vitals-phi` counterpart to `default_phi_reconstruct`,
callable from this GPU-less dev machine) confirmed working on real Wan
output, not just Cosmos's. Population currently n=2 only -- illustrative,
explicitly flagged as below the n=10 statistical-significance floor
everywhere it's shown (score report, native viewer) -- scaling it to n=80
like Cosmos is still open (M6 item below), not because Wan is lower
priority in principle but because the architectural question below made it
non-blocking for the project's actual purpose.

**Population-building generalized, not duplicated per model.**
`scripts/run_cosmos_population.py` (Cosmos-only, hand-rolled render ->
generate -> reconstruct -> slice sequence) replaced by `scripts/run_model_
population.py`, which properly reuses `VideoWorldModel` (`vitals/adapters/
video_model.py`) via a `generate_fn` dispatch keyed off `--model`. Adding a
third real model means one dispatch line, not a new script. `vitals/
adapters/__init__.py` now holds the single canonical `REAL_MODEL_BACKENDS
= {"cosmos", "wan"}` set that both this script (which `--model` values it
accepts) and `render_survival_report.py`/`render_score_report.py` (which
category/color a result gets) import -- **found and fixed a real bug this
way**: `render_survival_report.py`'s own color logic had hardcoded
`backend == "cosmos"` as the only non-Truth category, so Wan's results
silently miscategorized as a Baseline (a real model grouped with
ConstantVelocity/CopyLastState) the first time a second model's results
existed. `tests/test_render_survival_report.py` regression-guards this
specific failure mode directly.

**Architectural question resolved directly (2026-08, asked directly: "it
shouldn't need 2 separate models to validate results... is that still
possible with this architecture, or does this current architecture
require multiple real models to be tested?"): NO, one real model is
sufficient.** A single team evaluating their own model needs Truth (once,
free, local, to confirm the harness itself is trustworthy) and Baseline
(always, free, local, as the actual comparison point) -- both already
computable with zero dependency on any other party's model. This was
already proven complete by the Cosmos-vs-baseline result (M6, above)
BEFORE Wan was ever integrated; Wan's value is a second data point that
the architecture generalizes, not a requirement the architecture had been
missing. This directly supersedes any earlier framing of "a second real
model" as a blocker -- it is not one, for the project's actual purpose
(one company evaluating one model). It remains useful for VITALS' own
future credibility (showing the harness produces sane, differentiated
results across genuinely different models), which is why Wan was still
worth finishing.

**`results/score_report.pdf` -- the actual per-model results/analysis
deliverable, real model vs. Baseline only** (2026-08, four corrections in
a row before landing: (1) "i wanted a full score report with analysis",
not the milestone-status document first built by mistake; (2) real model
vs. Baseline only -- Truth dropped entirely, since it calibrates the
detector rather than being a meaningful comparison point for a model's own
prediction quality; (3) no HTML/browser output -- PDF via matplotlib,
opens in any native viewer, per this project's established "no browser"
preference; (4) color key, degradation curves added to the PDF itself,
consistent series ordering across every chart on a page, and axis scales
computed GLOBALLY across every model page in a run so e.g. Cosmos's and
Wan's pages are directly comparable rather than each auto-scaling to its
own data). `scripts/render_score_report.py`, one page per real model found
in `results/l0_demo_*.json`: VI50 + CI, bootstrap delta vs. each Baseline
with a plain significance verdict, a Kaplan-Meier degradation curve, a
failure-time boxplot, an R2/R5 failure-mode breakdown, and a "COMPUTED
READOUT" block of discrete `key = value` lines (deliberately not prose --
every number here is templated from floats already computed by `vitals.
stats.survival`/`vitals.detect.events`, no LLM involved anywhere in this
file or its inputs, verified directly by grepping the whole codebase).
Small-n populations (Wan's current n=2) get an explicit warning band and
are excluded from significance testing rather than silently shown as if
comparable.

**Live second-by-second tracing extended to real models.**
`scripts/plot_survival_native.py --model <name>` filters the existing
native (non-browser) survival viewer to one real model + its own two
Baselines, in the SAME colors as that model's own PDF page (imported
directly from `render_score_report.py`, not redefined, so the two
artifacts cannot drift apart) -- gives the PDF's own degradation curves a
live, crosshair-hover, second-by-second equivalent. Omit `--model` for the
original unfiltered GATE 1 exploratory view (every scenario/backend,
generic cycled palette) -- unchanged.

### M6.1 — Is the "no good results" finding a real benchmark-parameter artifact? Tested directly, not assumed (2026-08)

**The question, asked directly: "how are there absolutely no good results
though? these are state of the art models, doesn't that mean something
with the benchmark's parameters are off."** A fair challenge, worth
actually testing rather than arguing about. Checked what could genuinely
be an "off" parameter before concluding either way: Cosmos-Predict2 was
running its **2B** Video2World checkpoint, deliberately chosen over the
**14B** variant "for the FIRST real run... start with what's cheapest to
validate the PIPELINE with, not the biggest model available" (cosmos.py's
own docstring, M6 above) -- a real, honest limitation on the "state of
the art" framing, not a settled non-issue.

**Real numbers, fetched directly from Cosmos-Predict2's own
`documentations/performance.md` before touching any code (not guessed):**
14B needs 56.38GB VRAM (64GB recommended) vs. 2B's 32.54GB -- both fit
inside the already-provisioned H100 (80GB), no CPU-offload flags needed,
no new GPU tier. Generation itself is genuinely slower, though: 79.87-
87.32s (2B) vs. 286.46-377.67s (14B) per 480p/16fps call on H100 (~3.6-
4.7x) -- still comfortably under the shared deployed function's 1200s
server-side timeout, but with far less margin than 2B's own untested-at-
14B-scale 600s client default. Given Wan's own client/server-timeout
mismatch incident this exact session, `cosmos14b` was given its own
model-specific default (1260s, matching Wan's) up front rather than
risking the identical bug a second time.

**Wired in as its own backend name, `cosmos14b`, not a flag on the
existing `cosmos` entry** -- `vitals/adapters/__init__.py`'s
`REAL_MODEL_BACKENDS` gained a third name (both `render_survival_report.
py`/`render_score_report.py` already derive their categorization
generically from this set, confirmed by reading, not assumed -- zero
changes needed there) so a 14B result gets its own `results/l0_demo_
occlusion_corridor_cosmos14b.json`, comparable to but never silently
overwriting the 2B population. `remote/modal_app_cosmos.py`'s
`video2world`/`download_checkpoints` and `vitals/adapters/cosmos.py`'s
`make_cosmos_generate_fn_modal` all gained a `model_size` parameter
(default `"2B"`, so every existing call site is unaffected); `scripts/
run_model_population.py`'s `make_generate_fn` dispatches `"cosmos14b"` to
the SAME adapter with `model_size="14B"` bound. 4 new/updated unit tests;
full suite 104/104.

**Real prerequisite, confirmed rather than assumed away: 14B is gated
SEPARATELY from 2B on Hugging Face** (`nvidia/Cosmos-Predict2-14B-
Video2World`'s own page has its own "agree to share your contact
information" gate -- accepting 2B's terms does not cover it). Cleared
directly by the account owner; `download_checkpoints` now takes a
`model_sizes` tuple (default `("2B",)`, so a plain re-run never touches
anything already working) rather than a hardcoded single size.

**The actual experiment: n=5, `occlusion_corridor` only (a diagnostic
probe, not a full population -- confirmed directly before committing to
n=80's ~5x greater GPU cost).** Ran cleanly on the first attempt, zero
timeouts, zero failures -- `results/l0_demo_occlusion_corridor_
cosmos14b.json`: **VI_50 = 1.13s, 95% CI [1.00, 1.33], n=5, termination
profile 100% R5**, same failure signature as every other real-model
result on this scenario.

**The honest answer: no, the larger checkpoint does not rescue the
result.** `bootstrap_vi_delta` against the existing populations:

| comparison | delta | 95% CI | verdict |
|---|---|---|---|
| cosmos14b vs. cosmos-2B (n=80) | -0.267s | [-0.400, -0.067] | SIGNIFICANT -- 14B fails EARLIER |
| cosmos14b vs. constant_velocity | -1.200s | [-1.533, -1.000] | SIGNIFICANT -- 14B worse |
| cosmos14b vs. copy_last_state | -0.533s | [-1.000, -0.333] | SIGNIFICANT -- 14B worse |

14B's own VI_50 (1.13s) is not just short of both baselines (2.33s,
1.67s) -- it's statistically indistinguishable from "no better than 2B,"
and the point estimate is slightly WORSE, not better. **Read with the
appropriate caution: n=5 is a small, illustrative sample** (same
`MIN_N_FOR_STATS` caveat every other small-n population in this document
carries) -- this is not strong evidence 14B is truly worse than 2B, but
it is real evidence AGAINST "a 7x larger checkpoint meaningfully closes
the gap." A confident claim either way would need n=80, not attempted
here (diagnostic scope, confirmed directly before spending the ~5x
greater GPU cost).

**What this does and doesn't settle.** It rules out "the 2B-vs-14B
choice alone explains the result" as the dominant explanation -- a
genuinely bigger, more capable checkpoint from the SAME model family
shows the identical qualitative failure (100% R5, fails within ~1-1.3s
of a 6s clip). It does NOT rule out every other honest caveat raised
alongside this one (480p resolution chosen for cost/turnaround rather
than quality; Wan's own single-frame conditioning vs. Cosmos's 5-frame
window; plain, unsteered prompts by design) -- those remain untested,
not disproven. The finding stands as: two structurally different real
video models, at two different Cosmos sizes, all fail this specific
physics-plausibility test consistently -- narrower evidence than "all
video models everywhere fail," broader than "one small checkpoint had a
bad day."

**Second sub-experiment, same conversation, same discipline: does
resolution change the result?** `COSMOS_RESOLUTION` was chosen as "480"
"for cost/turnaround, not quality" (the literal comment where it's
defined) -- untested against Cosmos's own supported "720" until now.
Isolated as its OWN backend, `cosmos720p` (same 2B model_size as the
original published result, only resolution varied -- one axis at a
time, not conflated with the model-size question above), reusing
`REAL_MODEL_BACKENDS`'s now-established pattern. No new Modal deploy or
checkpoint download needed for this one specifically -- `resolution` was
already a real parameter on the deployed `video2world` function from
before this session, and `download_checkpoints` had already fetched
BOTH 480 and 720 assets for 2B (`--resolution 480 720`, confirmed by
reading the download call directly, not assumed).

Ran cleanly, n=5, `occlusion_corridor`, zero timeouts, zero failures:
**VI_50 = 1.0s, 95% CI [1.0, 1.4]** -- if anything the LOWEST of the
three Cosmos configurations tested so far (480p/2B: 1.4s; 14B: 1.13s;
720p/2B: 1.0s), though the comparison against 480p/2B doesn't clear
significance at this small n (delta -0.400s, CI [-0.400, 0.000] --
touches zero, not confidently different, unlike the 14B-vs-2B
comparison which did clear it). Still significantly worse than both
baselines: vs. constant_velocity, delta -1.333s CI[-1.633,-0.933]
SIGNIFICANT; vs. copy_last_state, delta -0.667s CI[-1.133,-0.267]
SIGNIFICANT.

**Running tally, all real-model configurations tested on
`occlusion_corridor` so far, every single one significantly worse than
both baselines (constant_velocity VI_50=2.33s, copy_last_state
VI_50=1.67s):**

| configuration | n | VI_50 | 95% CI |
|---|---|---|---|
| cosmos 2B, 480p (original) | 80 | 1.40s | [1.40, 1.40] |
| cosmos 14B, 480p | 5 | 1.13s | [1.00, 1.33] |
| cosmos 2B, 720p | 5 | 1.00s | [1.00, 1.40] |
| wan | 5 | 1.47s | [1.03, 3.53] |

Two independent axes (model capacity, resolution) both tested and both
came back "no improvement, possibly slightly worse" against the same
baseline. The remaining untested caveats (Wan's single-frame
conditioning, prompt richness) are the honest next candidates if this
line of investigation continues -- see the next-steps discussion in
conversation, not duplicated here.

### M6.5 — Every real-model result so far was capped at 3.0s; fixed, and a real batch-crash bug found along the way (2026-08)

Real sanity check raised directly ("are there any tasks/runs where it
succeeds because otherwise this is an invalid task?"). Pulled every
existing population's real numbers rather than assume: the GATE 1a/1b
ground-truth (`mujoco`) populations on `ramp_descent`/`ramp_descent_high_
friction` show 53%/62% censoring with VI_50=inf (the median rollout NEVER
fails within the clip) -- real, direct proof the detector is genuinely
discriminative, not just uniformly firing on everything. That answers the
sanity check.

But pulling those numbers surfaced a real, previously-undocumented
confound: every Cosmos/Wan population to date was scored over **t_max=3.0s**
(1s prefix + `--n-continuation-frames 60` = 2s), while every ground-
truth/baseline population runs over 5.97-7.97s. Every observed Cosmos/Wan
failure so far happens well inside 3s (1.0-1.5s), so this hasn't biased
the deltas already published -- but it means **no real model has ever had
a chance to demonstrate surviving longer**, a structural ceiling on what
this benchmark could say about model quality, independent of how good a
model actually is.

Root cause, checked directly rather than assumed: `resample_to_target`
was silently discarding real generated content. `wan.py`'s own module
docstring already stated this plainly (`num_frames=81` DEFAULT at 16fps,
~5.06s, generated unconditionally every call) -- confirmed by inspecting
`resample_to_target`'s own trim/pad boundary math (`vitals/adapters/
video_utils.py`): real (non-padded) frames exist up to `src_t/src_fps`;
for Wan that's 80/16=5.0s, i.e. up to 151 target frames at 30fps before
padding starts. Cosmos's own module docstring flagged this as genuinely
UNRESOLVED ("Cosmos's own docs do not specify how many frames one call
returns... do not trust the padding behavior") -- built `scripts/probe_
cosmos_native_length.py` (one real GPU call, reads `_read_video`'s raw
output BEFORE any resample/pad step) to check directly instead of
guessing: **93 frames @16fps = 5.81s**, even more than Wan, and NEVER
padding -- so no prior Cosmos result was silently scoring a frozen/held
frame as real model behavior, a real, worth-stating relief, not assumed.

Fixed by raising `--n-continuation-frames` to 150 (5.0s of continuation,
30 prefix + 150 = 180 frames = 6.0s total episode) -- comfortably inside
BOTH models' own real-content caps (151 for Wan, 172 for Cosmos), and
inside `ramp_descent_high_friction`'s own 180-frame reference horizon
exactly (no theta-array truncation mismatch). Re-ran Wan (n=5, cheap) on
both scenarios -- pre-existing short-horizon results backed up to
`results/archive/*.PRE_LONGHORIZON.json` first, per this session's own
established backup discipline.

**Real, asymmetric result, not fabricated:** `occlusion_corridor`'s VI_50
stayed 1.47s but the 95% CI upper bound jumped from 2.40s to **3.53s** --
one episode (seed=2) that would have been silently right-censored at the
old 3.0s cap turned out to have a real, later failure at t=3.53s, real
information the old horizon was hiding. `ramp_descent_high_friction`'s
numbers came back essentially IDENTICAL (VI_50=1.067s vs. 1.067s before)
-- not a bug, this scene's own real failures cluster very early (max
~1.07s across all 5 episodes) regardless of how much longer horizon is
available to fail within, so extending it had nothing new to reveal here.
Both are real, honest outcomes of the same fix -- one scenario had
hidden information, the other didn't, and the fix couldn't have known
which in advance.

**A real, separate crash-resilience bug found DURING this exact
re-run**, not hypothesized: `ramp_descent_high_friction`'s first Wan
long-horizon attempt hit a genuine `FunctionTimeoutError` (1200s) on ONE
episode (seed=1) -- and `run_model_population.py`'s own collection loop
had an unconditional `raise` on any episode failure, which (via
`ThreadPoolExecutor`'s own shutdown-then-propagate behavior) let the
OTHER 4 already-succeeded, already-real-GPU-cost episodes finish, then
threw the whole batch away anyway once the pool drained -- a 4/5-
successful run became a 0/5 result over one transient remote hiccup.
Fixed properly, not patched over: extracted `with_one_retry(fn, label)`
into `vitals/adapters/video_utils.py` (mirrors this project's own
established precedent for transient Modal failures -- M2.9's Wan retry,
which succeeded immediately on its second attempt) -- exactly one retry,
then the seed is excluded from the population (never silently retried
forever, never silently padded). `run_model_population.py`'s own JSON
output now carries `n_requested`/`failed_seeds` explicitly, so a
population's own `n` can never be silently mistaken for "as many as
`--seeds` asked for" when they differ. 3 new unit tests (`tests/test_
video_utils.py`, pure function, no GPU/network). Re-ran with the fix --
completed cleanly, 0 failures this time.

Full suite 83/83 passing. `scripts/run_validity_analysis.py` and
`score_report.pdf` regenerated with both updated Wan populations.

**Cosmos long-horizon re-run, n=80, both scenarios, approved and
attempted -- `ramp_descent_high_friction` succeeded cleanly on the first
try** (n=80, VI_50=1.0s, CI[1.0,1.0], identical to the short-horizon
result -- same "failures cluster too early to matter" story as Wan's own
result on this scenario). **`occlusion_corridor` needed FOUR attempts**,
three of which failed for three DIFFERENT real reasons, each root-caused
rather than blindly retried:

1. Killed at 57/80 by this session's OWN Bash tool timeout (`timeout:
   600000` caps a background process's lifetime at 10 minutes REGARDLESS
   of `run_in_background` -- a real tool-usage mistake, not an
   infra/Modal issue; the original short-clip n=80 runs happened to
   finish under 10 minutes, this longer-clip one didn't). Fixed
   procedurally: launch via `nohup ... & disown` (fully detached from the
   tool's own process tracking) plus a persistent `Monitor` watching the
   log for milestones/errors, decoupling real long-running work from any
   single tool call's own lifetime limit.
2. A genuine LOCAL machine DNS outage mid-run (confirmed directly:
   `[Errno 8] nodename nor servname provided` on ~50 consecutive seeds,
   then confirmed RESOLVED via `host modal.com`/`ping` before concluding
   anything) cascaded into every remaining episode failing both retry
   attempts, followed by the run hanging indefinitely on its very last
   episode for over a day of real elapsed time -- confirmed as a genuine
   hang (not just slow) via `modal app list`'s own Tasks column reading
   0 for `vitals-cosmos` while the local process still claimed to be
   waiting on it, the same live-activity check established earlier this
   session. No results were ever written (the script only writes after
   its FULL episode loop completes) -- nothing to salvage, required a
   clean restart.
3. **The real, generalizable finding**, surfaced by that SAME job hanging
   a SECOND time (75/80, confirmed stuck the same way -- 0 Modal tasks,
   flat CPU across a long elapsed span): `with_one_retry`'s own plain
   try/except is not sufficient protection on its own. It only catches
   calls that RAISE something -- a Modal `.remote()` call that hangs at
   the network/container level without ever raising just blocks the
   calling thread forever, and retry logic never gets a chance to run
   because the FIRST attempt never returns control at all. Fixed by
   adding `call_with_timeout(fn, timeout_s)` to `vitals/adapters/video_
   utils.py`: runs `fn` in a single-use worker thread and enforces a hard
   wall-clock deadline via `Future.result(timeout=...)`, raising
   `HungCallTimeout` (a real, catchable exception `with_one_retry` can
   then act on) instead of blocking indefinitely. **A real bug caught
   DURING testing, not shipped blind**: the first implementation wrapped
   the worker in `with ThreadPoolExecutor(...) as pool:`, whose own
   `__exit__` calls `shutdown(wait=True)` -- silently blocking for the
   FULL length of the hang anyway, one stack frame further down, entirely
   defeating the fix. Caught by the new tests themselves (`elapsed < 2.0`
   assertions that would have failed at ~5s), not by re-running the real
   n=80 job a fifth time to find out. Fixed by managing the pool
   manually and calling `shutdown(wait=False)` in both the success and
   timeout branches. `run_model_population.py` gained `--episode-timeout-s`
   (default 600s), wired through to every episode. 4 new unit tests
   (`tests/test_video_utils.py`), full suite 87/87.

**Fourth attempt completed successfully with the timeout fix in place.**
Two more transient hangs occurred mid-run (seeds 39 and 40, each "did not
return within 600.0s") -- but this time `call_with_timeout` did exactly
its job: each was caught as `HungCallTimeout`, retried once, and the
batch kept moving without blocking the other 78 episodes. One seed (35)
failed twice in a row and was honestly excluded (`failed_seeds=[35]`),
giving a final population of **n=79** rather than a silently-padded 80.

**Real, new result, not a repeat of the short-horizon number:**
`occlusion_corridor` Cosmos VI_50 = **1.4s**, 95% CI **[1.4, 1.4]**,
censoring=0.00, termination profile `{R5: 1.0}` -- every episode failed,
consistently, at the same real time, once the model was given its own
full native horizon (5.81s) instead of the old 3.0s cap. This is a
materially different, and materially more informative, number than the
pre-decal/short-horizon run: it shows Cosmos's occlusion-corridor failure
is a real, reproducible, deterministic-looking breakdown near t=1.4s,
not an artifact of truncating the clip before the model had a chance to
fail. Combined with `ramp_descent_high_friction`'s clean first-try n=80
(VI_50=1.0s, CI[1.0,1.0], identical to short-horizon), the long-horizon
re-run is now complete for both scenarios on both real models.

`scripts/run_validity_analysis.py`, `score_report.pdf`, and
`gate1_survival.html` regenerated against all four updated long-horizon
populations (`occlusion_corridor`/`ramp_descent_high_friction` x
`cosmos`/`wan`). `occlusion_corridor`'s Cosmos test-retest spread is
still 0.000s (VACUOUS MDD flag, expected -- CI width is already
[1.4,1.4], i.e. deterministic across seeds), same honest caveat as every
other zero-variance real-model population in this table.

### M7 — Validity analysis (week 5)

lambda sweep with rank correlation between levels; (alpha, pi) sensitivity
grid; bootstrap CIs; minimum detectable difference; seed test–retest;
calibration horizons and rank histograms for P6.

**Acceptance criterion:** model rank order is stable across the lambda sweep.
If it is not, that is a negative validity finding and must be published as
one — not resolved by picking a favourable lambda.

**Small fix, 2026-08: `run_l0_demo.py`'s survival analysis (Kaplan-Meier,
VI_50 + bootstrap CI, termination profile) now persists its FULL result,
including the raw per-episode event list, to
`results/l0_demo_<scenario>_<backend>.json`** -- previously this was
computed correctly but only ever printed to the console, downsampled even
there (~6 sampled points of the survival curve; the full grid/curve arrays,
and the raw per-episode population entirely, were computed then discarded
on process exit). Not a new computation, not a "scorecard" (this project's
own 3.11 rules out a single aggregate scalar; a survival curve +
termination profile IS the report that stands in for one) -- just making
an already-correct result actually accessible, and RE-ANALYZABLE without
re-simulating, after the process ends. `scripts/render_survival_report.py`
renders `results/gate1_survival.html` from every `l0_demo_*.json` found on
disk, auto-discovering scenarios (a new one shows up automatically after
its own `run_l0_demo.py` pass, no manual edits needed) -- step-function
survival curves, VI_50/CI/censoring/termination-profile stat cards, GATE
1a/1b tables, raw data table toggle. Both scripts are real, committed,
re-runnable tools now, not one-off artifacts.

**`stats/survival.py` gained `bootstrap_vi_delta(events_a, events_b, q,
n_boot, seed)`** -- percentile CI on VI_q(a) - VI_q(b), the direct
generalization of "compare a good rollout population against a bad one and
take the delta" (2026-08, requested directly) to any two Event
populations: two mutant conditions, two lambda levels, two scenarios
today, two real models' measured rollouts once M6 exists. Independent
two-sample bootstrap (each population resampled on its own) -- a PAIRED
bootstrap would be strictly more powerful if the two populations are
literally the same episode seeds measured under two conditions, and this
function does not do that; see its own docstring. Validated on real data,
not just synthetic: comparing two `vanish` populations that reliably
terminate at different, deterministic times (t_star=1.5s vs 4.0s) gives a
correctly SIGNIFICANT delta (+2.50s, CI [+2.50, +2.50] -- zero-width
because there is genuinely zero timing variance in either population, not
a bug). Comparing GATE 1a's own null (defect-free) population against
anything else at q=0.5 correctly returns NaN -- not a bug either: a
population deliberately designed to almost never terminate (~1% target)
has a structurally UNDEFINED median survival time (inf), the same reason
`bootstrap_vi` already drops non-finite draws rather than fudge them. The
practical lesson, worth stating plainly: **VI_50 comparisons are only
meaningful when both populations have a substantial, comparable
termination rate within the horizon** -- which is the expected regime once
comparing two real, imperfect models (M6/M7), not GATE 1a's own
deliberately-clean calibration population. For a heavily-censored
population, a LOWER quantile (VI_25, VI_10) may be the only one that's
well-defined at all -- pick q to match how much of the population you
actually expect to terminate, not always 0.5. `tests/test_core.py::
test_bootstrap_vi_delta_separates_clearly_different_populations` covers
both a clear-separation case and the population-vs-itself null case.

**This is real, usable statistical machinery today, but it is NOT the
full picture M7's own acceptance criterion asks for.** A significant delta
at ONE lambda level is suggestive, not conclusive -- M7's actual acceptance
criterion is RANK-ORDER STABILITY across the lambda sweep (this section's
own text, above), and a single-lambda comparison can flip sign at a
different lambda entirely. `bootstrap_vi_delta` is the tool for the
lambda-sweep's own per-level comparisons once that sweep exists; it is not
a substitute for running the sweep. Minimum detectable difference and seed
test-retest (this section's own list, above) -- i.e., how much VI_50 spread
exists from finite-sample noise ALONE, even between two runs of the
identical condition -- remain unbuilt and are what calibrates how large a
delta needs to be before it's even worth bootstrapping in the first place.

**Built (2026-08).** `stats/survival.py` gained `seed_test_retest(events,
q, n_splits, seed)` and `minimum_detectable_difference(events_reference,
t_max, q, n_per_arm, target_power, n_sim, n_boot, delta_grid, seed)`, plus
`scripts/run_validity_analysis.py` to run both over every existing
`results/l0_demo_*.json` -- zero new GPU/Modal cost, pure re-analysis of
data already collected (same "don't re-run what you already have"
discipline as everywhere else real cost is involved here).

`seed_test_retest` answers "if this experiment had drawn a different,
equally-sized batch of held-out seeds, would VI_50 come back the same?" by
repeatedly splitting ONE existing population into two DISJOINT halves
(sampling WITHOUT replacement) and measuring each half's own VI_50 --
closer to genuine seed-to-seed reproducibility than a same-sample
bootstrap CI is, since a bootstrap resample can overlap up to 100% with
any other resample and a disjoint half never does. `minimum_detectable_
difference` answers the calibration question directly: simulates two
n_per_arm populations drawn from a REAL reference population's own
empirical failure-time distribution (never an assumed parametric shape),
shifts one arm by a candidate delta (re-censored at `t_max` if the shift
pushes an event past the clip's own observed horizon -- an event can't be
observed happening later than the clip runs), and reports the smallest
delta at which `bootstrap_vi_delta` correctly flags the shift as
significant in >=80% of simulated replicates. 7 new unit tests (`tests/
test_validity_analysis.py`), full suite 80/80.

**Real numbers, all 16 existing populations, not just synthetic
validation (`python3 scripts/run_validity_analysis.py`):**

| scenario/backend | n | test-retest median\|Δ\| | MDD (power>=80%) |
|---|---|---|---|
| occlusion_corridor/cosmos | 80 | 0.000s | 0.038s **(VACUOUS)** |
| occlusion_corridor/wan | 5 (small) | 0.333s | not reached |
| occlusion_corridor/constant_velocity | 40 | 0.133s | 0.417s |
| occlusion_corridor/copy_last_state | 40 | 0.067s | not reached |
| ramp_descent_high_friction/cosmos | 80 | 0.000s | 0.038s **(VACUOUS)** |
| ramp_descent_high_friction/wan | 5 (small) | 0.033s | 0.050s |
| ramp_descent_high_friction/constant_velocity | 40 | 0.067s | not reached |
| ramp_descent_high_friction/copy_last_state | 40 | 0.067s | 0.267s |

**A real, easy-to-misread edge case, caught and labeled, not silently
reported:** Cosmos's own populations are PERFECTLY deterministic on both
scenarios (every episode fails at literally the same instant -- the
zero-variance finding already documented elsewhere in this file), so
`seed_test_retest` correctly returns exactly 0.000s. Naively self-scaling
`minimum_detectable_difference`'s own delta grid off that zero would
degenerate to testing delta=0 eight times over -- caught before it
shipped (`test_mdd_deterministic_reference_gets_a_nondegenerate_grid`),
fixed by flooring the grid's own scale against `t_max * 0.05` whenever the
reference population's natural spread is zero or undefined. Even with
that fix, the resulting MDD number (0.038s) is flagged `VACUOUS` in both
the script's own output and the score report's readout card: detecting
literally ANY consistent nonzero shift against a population with ZERO
natural noise is trivial, so this number reflects Monte Carlo/bootstrap
mechanics finding their own floor, not a meaningful real-world detection
threshold. The honest, useful numbers in the table above are the
BASELINE rows (real, non-degenerate spread) -- e.g. `constant_velocity`
on occlusion_corridor needs a true ~0.42s difference before it's reliably
detectable at n=40, a genuinely useful number for deciding whether a
future single-lambda `bootstrap_vi_delta` result is even in the
detectable range before treating it as conclusive.

`ramp_descent/mujoco` and `ramp_descent_high_friction/mujoco` (GATE 1a's
own near-never-terminates null populations) correctly report test-retest
as `undefined` -- consistent with `bootstrap_vi_delta`'s own documented
NaN-for-structurally-censored-population behavior, not a new bug: a
population deliberately designed to almost never terminate has almost no
random half with 2+ observed failures to compare.

Wired into `score_report.pdf`'s own COMPUTED READOUT card (`scripts/
render_score_report.py`, reading `results/validity_analysis.json` if
present, "not yet computed" otherwise -- never blocks the report from
rendering). Found and fixed a real layout bug while wiring this in: the
readout card's line spacing/height were hardcoded for exactly 5 lines
(`0.145` axes-fraction spacing, `read_h=1.35`); adding these 2 new lines
pushed a real page (one with 2 baseline deltas) to 7 lines, whose last
line landed at axes y=-0.25 -- entirely below the card, caught by
actually rendering and inspecting the PDF (not assumed from reading the
code), not by any test. Fixed by deriving both spacing and card height
from `len(lines)` (`line_spacing = 0.58 / (n_lines - 1)`, `read_h = 1.35 *
max(1, n_lines / 5)`) -- reproduces the original hardcoded values exactly
when n_lines==5, so every existing 5-line page is byte-identical, and
generalizes cleanly for any future line count.

**Honest scope note:** Wan's n=5 populations sit right at `seed_test_
retest`'s own minimum (needs >=4, >=2 per half) -- its numbers are
reported, not refused, but should be read as barely-informative, the same
caution `MIN_N_FOR_STATS` already applies to `bootstrap_vi_delta` in the
score report. Increasing Wan's n is the more useful next step than
trusting its current MDD/test-retest numbers as calibrated.

**Lambda sweep + rank-order stability -- built (2026-08), and it found a
real instability.** M7's own acceptance criterion (this section's own
text, above) had never actually been checked until now -- every threshold
in the project was calibrated at exactly ONE hand-tuned `lam` per
manifest, and nothing tested whether a comparative claim built on that
one calibration survives moving `lam` away from it.

**Scope, decided directly:** Truth + both baselines only for this first
pass, not the real video models (Cosmos/Wan). `lam` only reparameterizes
the reference ensemble and its calibrated thresholds -- it does not touch
how a candidate trajectory is reconstructed -- so Truth (the same mixed-
mutant population `run_l0_demo.py` already builds) and both baselines
(`ConstantVelocity`, `CopyLastState`) are pure local MuJoCo physics +
closed-form prediction, free to re-run at any number of `lam` levels. But
no real model's raw reconstructed trajectory was ever saved to disk --
only its final event time at the one `lam` it was scored under -- so
re-scoring Cosmos/Wan across a sweep would mean re-spending real GPU
money per level, not a free re-analysis. That's a separate, explicitly
deferred follow-up (cache raw trajectories once per model/scenario, then
sweep against the cache for free), not attempted here.

`scripts/run_lambda_sweep.py`: for `occlusion_corridor` and
`ramp_descent_high_friction` (the two scenarios with published real-model
comparisons), sweeps `lam` at 0.5x/1x/2x each manifest's own frozen value
(never the manifest file itself -- 3.12's "decide once" governs the
shipped calibration, not this diagnostic). At each level: rebuilds the
reference ensemble and calibrated thresholds, reruns GATE 1a/1b (directly
answering T5's own risk -- how does sensitivity/specificity respond to
moving the reference spread), and reruns the Truth population plus both
baseline populations. `vitals/stats/survival.py` gained
`spearman_rank_correlation(values_a, values_b)` (plain-numpy tie-aware
Spearman, no scipy dependency anywhere else in this project) to answer
"did the ranking stay the same" as a number rather than an eyeball. Whole
sweep (2 scenarios x 3 levels, ~250 local MuJoCo rollouts per level) runs
in ~13 seconds -- zero GPU/Modal cost, exactly as scoped. 5 new smoke
tests (`tests/test_lambda_sweep.py`, synthetic backend + tiny n) plus a
`spearman_rank_correlation` unit test (`tests/test_core.py`); full suite
93/93.

**`ramp_descent_high_friction`: perfectly stable across the whole grid.**
GATE 1a/1b pass at every level (5/5 GATE 1b cases OK throughout);
`ConstantVelocity` beats `CopyLastState` significantly at x0.5, x1.0, AND
x2.0 (delta +1.13s / +0.97s / +1.00s, every CI clear of zero); rank order
of {Truth, ConstantVelocity, CopyLastState} by VI_50 never changes
(Spearman rho=1.000 between every pair of adjacent levels, and end-to-end
x0.5-to-x2.0). This scenario's own comparative claim is genuinely robust
to the reference-spread choice, not an artifact of the one lambda it
happened to be tuned at.

**`occlusion_corridor`: NOT stable -- a real negative validity finding,
published as one, not tuned away.** GATE 1a/1b still pass at every level,
and the ConstantVelocity-vs-CopyLastState comparison is significant at
x0.5 and x1.0 (delta +0.73s, +0.67s) -- but at x2.0, `CopyLastState`'s
entire 40-episode population comes back **100% censored** (VI_50=inf,
bootstrap CI=[nan, nan]), because the reference ensemble at that much
wider spread calibrates an R5 threshold loose enough that "predict
nothing changes" never crosses it within the clip at all. The
ConstantVelocity-vs-CopyLastState delta at x2.0 is therefore itself
undefined (NaN, correctly -- the same "structurally undefined median"
representation `bootstrap_vi_delta` already uses elsewhere in this
project, not a bug), and the 3-way rank order flips: Spearman rho=1.000
from x0.5->x1.0, but **rho=-0.500 from x1.0->x2.0 and end-to-end**. This
is precisely the failure mode T5 names ("reference spread is a design
parameter, not a measurement... if invariance fails, comparative claims
fail") -- confirmed real on this project's own data, not hypothesized.
The manifest's own shipped `lam=6.0` (the x1.0 level) sits on the STABLE
side of this break, so nothing published so far is compromised by it --
but any future claim on this scenario built at a substantially wider
`lam` would need to check for exactly this degenerate-censoring failure
mode before trusting a rank order, not assume the x1.0 result generalizes.

`results/lambda_sweep_occlusion_corridor.json` and `results/lambda_sweep_
ramp_descent_high_friction.json` hold every level's full events/thresholds/
GATE results for re-analysis.

**The real-model follow-up -- done (2026-08, requested directly: "cache
trajectories and run the sweep on Cosmos and Wan").** `save_trajectory`/
`load_trajectory` added to `vitals/adapters/video_utils.py` (plain `.npz`,
JSON-encoded `names`/`meta`, not pickle -- readable without trusting
arbitrary code execution on load). `run_model_population.py` now caches
every successful episode's full (prefix+continuation) reconstructed
Trajectory to `results/trajectories/<scenario>_<model>/seed<N>.npz` as a
side effect, unconditionally, at zero extra GPU cost (it's the same
Trajectory already computed for scoring). `run_lambda_sweep.py` auto-
detects any `REAL_MODEL_BACKENDS` name with a populated cache dir for a
scenario and adds it as an extra condition at every lam level, re-scoring
the cached (already-paid-for) trajectories against that level's fresh
reference ensemble/thresholds -- no Modal/GPU call. Whole 2-scenario,
3-level, now-5-condition sweep still runs in ~13 seconds.

**A real, separate bug found and fixed populating the cache, not
hypothesized:** re-running Wan to populate its cache hit "Function call
was cancelled by user or a failure" on every seed, twice in a row (once
concurrent with the two Cosmos re-runs, once alone) -- looked at first
like the same resource-contention story as M6.5's Cosmos incidents, but
`modal app logs vitals-wan` showed something different: containers
legitimately still generating (some 70%+ through their 50 diffusion
steps, 12+ minutes in) when a cancellation signal arrived for MANY
in-flight inputs at once. Root cause, confirmed by reading `remote/
modal_app_wan.py` directly: its deployed `image2video` function has a
**server-side `timeout=1200`**, but this project's own
`--episode-timeout-s` default was `600` for every model -- fine for
Cosmos (both n=80 re-runs finished with zero client-side timeouts at that
value) but HALF of Wan's own real ceiling. The client was abandoning a
Wan call that was still legitimately running server-side, then
immediately firing an overlapping SECOND attempt for the same episode --
which is what actually produced the cancellation, not a genuine hang
(the thing `--episode-timeout-s` exists to catch). Fixed at the ROOT, not
patched around: `resolve_episode_timeout_s(model, explicit)` (`scripts/
run_model_population.py`) makes the default model-specific
(`{"cosmos": 600, "wan": 1260}`, Wan's own comfortably clearing its
server's 1200s ceiling) rather than one constant for every model; an
explicit `--episode-timeout-s` still overrides either default, including
`0` to disable. 4 new unit tests (`tests/test_run_model_population.py`).
Re-run with the fix: `occlusion_corridor` Wan landed n=3/5 (2 seeds
genuinely hit the server's real 1200s ceiling this time -- a legitimate
rare failure now, confirmed by the log's own "hit its timeout of 1200s"
message, not a false one); `ramp_descent_high_friction` Wan succeeded
cleanly, n=5/5. **A genuinely interesting side effect, worth stating
plainly:** the trajectory CACHE ended up with all 5/5 files for
`occlusion_corridor` Wan even though the SCORED population is n=3 --
the two "excluded" seeds' abandoned first attempts apparently completed
successfully in the background after the population loop had already
given up on them, and `save_trajectory` still ran when they did. This is
real, valid data (not corrupted -- checked directly, `T=180` for all 5,
matching the other 3), just late-arriving; the lambda sweep uses whatever
a cache directory actually contains, not the scored population's own
`n`, so it benefits from this for free. All 4 (scenario x model)
Cosmos/Wan re-runs matched or nearly matched their previously-published
numbers (`ramp_descent_high_friction`/Cosmos: VI_50=1.0 identical;
`occlusion_corridor`/Cosmos: VI_50=1.4 identical, n=80 this time with no
excluded seeds; `ramp_descent_high_friction`/Wan: VI_50=1.067 vs. 1.067
before) -- confirms this whole caching detour didn't silently change
what's being measured, only added the ability to re-score it for free.

**Real numbers, both scenarios, all 5 conditions, all 3 lam levels:**

`ramp_descent_high_friction` -- stable AND informative. Rank order barely
moves (Spearman rho=0.975, then 1.000, between adjacent levels). Both
real models are SIGNIFICANTLY worse than BOTH baselines at EVERY lam
level (`cosmos_vs_constant_velocity`, `cosmos_vs_copy_last_state`,
`wan_vs_constant_velocity`, `wan_vs_copy_last_state` all significant,
all negative, at x0.5/x1.0/x2.0) -- a real model underperforming even a
"do nothing" and a "constant velocity" baseline, robustly, regardless of
how the reference spread is chosen. This is the actually load-bearing
kind of finding VITALS is meant to produce, and the lambda sweep is what
makes it trustworthy rather than a single-lambda coincidence.

`occlusion_corridor` -- partially stable, and the SAME degenerate-
censoring failure from the baselines-only pass now visibly propagates
into the real-model comparisons too. `cosmos_vs_constant_velocity` is
significant (Cosmos worse) at all 3 levels -- robust. But
`copy_last_state` still goes 100% censored at x2.0 (the same finding as
before), which makes BOTH `cosmos_vs_copy_last_state` and
`wan_vs_copy_last_state` undefined (NaN) at that level, not just the
baseline-vs-baseline comparison -- a concrete demonstration of exactly
why T5's risk matters for REAL model comparisons, not just a baseline
sanity check: an unstable reference calibration doesn't just corrupt one
comparison, it corrupts every downstream comparison that depends on the
same threshold. `wan_vs_constant_velocity` is significant only at x0.5
(Wan's own small n=3-5 per level makes its own CIs wide enough to cross
zero at x1.0/x2.0) -- read as underpowered, not as evidence the effect
reverses.

Full suite 98/98. Extending the scenario list beyond these two remains
open, not attempted this pass.

**GATE 2 gained its own visual report, `scripts/render_gate2_report.py` ->
`results/gate2_instrument_tax.html` (2026-08, requested directly: "the
survival curve visualizer is great, can GATE 2 get one too").** Until now
GATE 2's own results (M5's published table above) existed only as prose in
this document and as scratch JSON that was never promoted into the
project's own `results/` directory -- real numbers from a real GPU/Modal
run, just not durably kept anywhere a script could re-read them. Copied the
three underlying raw-result files in (`gate2_instances_occlusion_corridor_
mujoco.json`, `gate2_artifact_sweep_occlusion_corridor_mujoco.json`,
`gate2_null_full_rerun_occlusion_corridor_mujoco.json`) so this report --
and any future one -- can be regenerated from them without re-running the
GPU pipeline.

Same auto-discovery convention as `render_survival_report.py` (a future
scenario's own GATE 2 pass shows up automatically, no manual edits), with
two chart types, chosen for the two different questions GATE 2 actually
answers:
- **A per-episode scatter of L0 event time vs. L1 event time**, one point
  per mutant instance, colored by mutant type, with a dashed y=x reference
  line -- this is the actual "instrument tax," made visible per-episode
  rather than only as an averaged bias number. Episodes where either side
  never fired (censored) are drawn as hollow squares at that channel's own
  clip-end time rather than silently dropped or faked as a real event --
  null's own 9/10 false-positive-through-Phi finding (M5's own writeup
  above) is directly visible this way: a cluster of hollow squares sitting
  far right (correctly never fired in ground truth) but low (fired early
  through video anyway).
- **A severity-sweep line chart** (mean mask IoU vs. artifact severity, one
  line per artifact type -- blur/noise/flicker) for the artifact-injection
  sweep, the other half of M5's own "done when" checklist.

The per-mutant summary table at the top of each scenario's block is
COMPUTED LIVE from the raw per-episode JSON (`summarize()` in the render
script), not hand-copied from this document's own prose table -- cross-
checked directly against it while building this (null 0/10 L0 / 9/10 L1,
vanish bias -0.079s +/- 0.165s, velocity_freeze correct-risk 7/8 / 5/8,
jitter bias +0.000s +/- 0.000s, all match to the published precision), so
the two can never silently drift apart the way a hand-copied second table
eventually would.

Only `occlusion_corridor` has a GATE 2 pass yet (M5's own scope), so
today's report shows one scenario -- ramp_descent's and projectile's own
real-GPU-pipeline passes are the still-open M5 follow-up item noted above,
not attempted this session (this machine has no GPU; running them is
remote/Modal work, not local report-building work).

---

## 8. How accuracy is actually achieved

"Flawless" is not available. **Auditable** is, and it is worth more. The
accuracy of this system comes from four disciplines, in order of importance.

### 8.1 Every measurement has a published error floor

Nothing enters the pipeline unvalidated. For each component: what is it
measured against, and what is its error? The answer goes in the paper, per
tier, never pooled. A measurable instrument knows when it has failed. If an
object is genuinely unrecoverable from the imagery, **exclude the episode and
publish the exclusion rate** — do not guess.

### 8.2 Defects are planted, so detection is verifiable

You never have to wonder whether a detector works. Corrupt a trajectory in a
known way at a known time, and check that the right detector fires at the
right moment. This gives sensitivity, specificity, event-time bias, ROC and
minimum detectable defect — with no world model involved.

One subtlety, discovered in the first run: **a defect is only detectable when
the true dynamics are actually changing at the injection point.** Freezing
velocity during steady motion is a no-op and correctly does not fire. So
injection times must be sampled where the reference ensemble spread is still
growing, not uniformly over the horizon.

### 8.3 The reference absorbs systematic error

Because thresholds come from reference-vs-reference deviation, anything that
affects both lanes equally cancels. This is why §3.1 and §3.3 are invariants
rather than preferences — they are the mechanism by which the instrument
tolerates its own imperfection.

### 8.4 Free parameters are swept, not chosen

`lambda`, `alpha`, `pi` are all free. The response is to report the response
surface and demonstrate ranking invariance, not to pick defensible-sounding
values. Where invariance fails, say so.

---

## 9. Known limitations — state, do not hide

| ID | Limitation | Mitigation | Residual |
|---|---|---|---|
| T1 | The six-fold decomposition may be wrong | factor analysis over per-property horizons; property-ablated oracle | publish a negative result if found |
| T2 | Input-domain shift — models may degrade on rendered input for non-physical reasons | render realism as a graded covariate, reported per model | **highest residual risk**; belongs in the abstract |
| T3 | External validity to real deployment unestablished. Concrete instance (§5): `phi/motion_prior.py`'s pixel-space kinematic re-id assumes near-constant object depth and a static camera, neither detected/corrected; real deployment needs camera intrinsics + monocular depth (the already-planned but unbuilt "3D positions" row, §7 M4) and, for a moving camera, visual odometry/SLAM | scoped as future work; build the monocular-depth path before any real-video claim | real |
| T4 | The simulator's physics is itself a model — contact solvers, restitution, friction become ground truth | cross-engine agreement on a subset | under-discussed; give it a paragraph |
| T5 | Reference spread is a design parameter, not a measurement | lambda sweep + ranking invariance | if invariance fails, comparative claims fail |
| T6 | Relation head has no external baseline | trained/validated on unlimited simulator ground truth | if precision inadequate, report P3 as a gap rather than ship it weak |
| T7 | Reference video is a clean render; candidate video is generated. Symmetry does **not** cancel this | artifact-injected validation set bounds it | bound it; never claim elimination |
| T8 | Causal-cone bound for P5 contamination | compute conservatively from simulator state | an over-tight cone manufactures failures |

---

## 10. Open research problems

These are not implementation tasks. Do not attempt them as if they were.

**The relation head (P3).** No off-the-shelf model extracts a contact/support
graph from video. The approach: train a small GNN or pairwise classifier on
geometric features from `Phi`'s other outputs (3D surface proximity, mask
adjacency, relative vertical displacement, relative velocity, contact
persistence) against unlimited ground-truth graphs from the simulator.
Deliberately low capacity, so it generalizes on geometry rather than
memorizing appearance. **If it does not reach acceptable precision, P3 is
reported as an unmeasured gap.** That is an acceptable outcome.

**The causal cone (P5 contamination).** Determining which regions an
intervention provably could not have reached requires a propagation bound
computed from simulator state. Too tight and you manufacture contamination
failures out of measurement slack. Report the bound's tightness.

---

## 11. Scope boundaries

**In scope for v1:** P2 (entity), P4 (dynamics), P6 (calibration) — one per
measurement modality, which preserves the structural claim that a complete
evaluation needs three experimental designs. 15 scenarios, M=20, 3 lambda
levels, 15s horizon, 3–4 open-weight models.

**Out of scope for v1:** P1, P3, P5; the relation head; multi-view; hazard
regression; deformables; fluids; neural rendering; real-world capture.

**Permanently out of scope:** any claim that passing implies understanding;
any aggregate leaderboard scalar; any language model in the measurement path;
web-scraped or found footage as scored data.

**Future scene ideas, recorded 2026-08 (not scoped or scheduled -- do not
build without a fresh scoping pass, same discipline as ramp_descent's own
non-planar work above):**

1. **Irreversible state change** (e.g. a glass pane shattering when struck).
   Engine mismatch: MuJoCo is a rigid-body engine with no native fracture --
   this overlaps directly with "deformables," already listed out-of-scope
   for v1 above. Buildable only as a crude approximation (pre-fragmented
   geometry held by constraints that break past a stress threshold), not a
   natural fit; a real implementation would likely want a different engine
   entirely.
2. **Rigid body dynamics** (e.g. a billiards break). The MOST feasible of
   the five -- this is MuJoCo's actual strength, and a direct extension of
   the K>1 multi-object work already built this session (M5.6) rather than
   a new capability. Best candidate to pick up first, if/when this list is
   revisited.
3. **Complex articulation** (e.g. an object passing through a revolving
   door). The SCENE is buildable in MuJoCo (native hinge/joint support).
   The MEASUREMENT side is the real gap: this needs actual 6-DoF pose
   estimation, already explicitly flagged as deferred in `reconstruct.py`'s
   own module docstring ("No 6-DoF pose... no point tracking at all") --
   not a small extension, a genuinely new measurement capability.
4. **Multiple subjects** (e.g. a busy street with many pedestrians). A
   SIMPLIFIED version (many independent simple rigid-body proxies, not
   realistic gait) is a further scaling of the K>1 work (M5.6) from K=2 to
   K=10+ and roughly as feasible; a REALISTIC version (actual walking
   humanoids) is a much bigger lift needing either a pretrained controller
   or hand-authored gait kinematics.
5. **Fluid dynamics.** LEAST feasible in MuJoCo -- it has a basic
   aerodynamic drag model for forces on rigid bodies, not an actual fluid
   solver (no free-surface water, no SPH/grid simulation). Already listed
   out-of-scope for v1 above ("fluids"), for the same reason. Would need a
   genuinely different tool, not a MuJoCo scene.

---

## 12. Honest framing for the writeup

Six weeks buys a **validated instrument and a first real measurement**. It does
not buy enough scenarios for statistically separable rankings — confidence
intervals will overlap between similar models.

Claim: *"we built the measuring device and proved it works."*
Not: *"here is the leaderboard."*

Write that into the contribution section on day one. It is the difference
between a methods paper that ages well and a ranking paper that gets attacked
on statistical power.