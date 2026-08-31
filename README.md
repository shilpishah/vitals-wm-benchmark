# VITALS

**V**alidity **I**ntervals and **T**ermination **A**ttribution for **L**ong-horizon **S**imulation

Measures how long a world model's rollout stays a valid prediction of the
physical world, and which necessary property fails first.

---

## The core loop

```
M reference rollouts from perturbed initial conditions
  -> threshold = 99th pct of how much references differ from EACH OTHER
  -> run candidate, find first sustained crossing
  -> (time, which detector) per rollout
  -> Kaplan-Meier survival curve;  median = the validity interval
```

Everything else is elaboration.

## Layout

**`AGENT.md` is the living design doc** -- the full narrative of every
milestone, decision, and finding, in the order it happened. This README is
just the map; AGENT.md is where the reasoning behind each piece lives.

```
vitals/                 the library -- everything importable
  types.py              Trajectory, Event, EpisodeSpec
  physics/
    perturb.py          Sigma: perturbation model, scalar lambda
    synthetic.py         analytic double -- no MuJoCo needed, for testing the core
    runner.py            MuJoCo rollout + fork + GATE 0 determinism check
  mutants/library.py    known defects at known times -- the validation engine
  detect/
    statistics.py       sigma_k(candidate, references) -> (T,)
    thresholds.py       leave-one-out quantile estimation
    events.py           first crossing + precedence rule
    physical_constants.py  known-physics-derived quantities (deceleration, etc.)
  phi/                  video -> Trajectory reconstruction (the "instrument")
    segmentation.py     mask extraction stand-in
    reidentify.py       existence-vs-visibility tracking across occlusion
    motion_prior.py     metric-anchored prediction from a partial track
    reconstruct.py       pixel track -> 3D Trajectory (flat-ground / known-plane / ballistic)
  render/mujoco_renderer.py   Trajectory -> video + ground-truth masks
  stats/survival.py     Kaplan-Meier, validity interval, bootstrap CI/delta
  adapters/base.py       WorldModel protocol + degenerate baselines (M6 lands here)
scenes/                 MJCF, one file per scenario (isolated per AGENT.md 3.6)
configs/manifests/      declarative EpisodeSpecs, one per scenario -- source of
                        truth for scenario parameters; the corpus regenerates
                        from these alone
scripts/                runnable entry points -- run_l0_demo.py (state-space
                        core), run_gate2.py (instrument tax, needs remote GPU),
                        run_cosmos_population.py (real-model population,
                        needs remote GPU), render_survival_report.py (GATE 1
                        HTML report), render_gate2_report.py (GATE 2 HTML
                        report), plot_survival_native.py (non-browser native
                        GUI alternative to the HTML report), render_
                        annotated_comparison_video.py (true-vs-reconstructed
                        position overlay video, needs remote GPU once per
                        episode), visualize_reference_ensemble.py
tests/                  one file per module, each runnable standalone
                        (python3 tests/test_X.py) as well as under pytest
remote/                 Modal harness for GPU-bound work -- modal_app.py
                        (SAM2/DINOv2, app "vitals-phi") and modal_app_cosmos.py
                        (Cosmos-Predict2 Video2World, app "vitals-cosmos"),
                        each with its own call_*.py invocation script
                        (never `modal run` directly -- see either file's own
                        docstring for why)

-- generated, not source; safe to delete and regenerate, never edit by hand --
results/                JSON + HTML output of scripts/run_l0_demo.py,
                        run_gate2.py, run_cosmos_population.py,
                        render_survival_report.py, and render_gate2_report.py
results/videos/         annotated comparison videos from render_annotated_
                        comparison_video.py -- real local mp4s, playable with
                        any native video player, no browser
viz/                    reference-ensemble videos/images from visualize_
                        reference_ensemble.py, one subfolder per scenario
```

## Quickstart

```bash
python3 tests/test_core.py        # no MuJoCo needed
python3 scripts/run_l0_demo.py    # full core, state-space only
```

Once MuJoCo is installed, swap `physics.synthetic` for `physics.runner` in the
demo. Nothing else changes -- that is the point of the Trajectory abstraction.

To regenerate the survival-curve report across every scenario:

```bash
for m in configs/manifests/*.yaml; do
  VITALS_MANIFEST=$m VITALS_BACKEND=mujoco python3 scripts/run_l0_demo.py
done
python3 scripts/render_survival_report.py   # -> results/gate1_survival.html
```

## Five design decisions

1. **Trajectory is the primary artifact, not video.** Everything upstream of
   rendering speaks state. Makes mutants trivial, makes observation
   interventions free re-renders, makes re-rendering at new realism cheap.

2. **Mutants live next to physics, not next to models.** A mutant is a
   corrupted trajectory, so once rendered it is photometrically identical to a
   real render except for the injected defect.

3. **Same detectors run on state (L0) and on video (L1).** The delta between
   them IS the instrument error floor. You get it for free.

4. **One adapter interface.** Mutants, baselines and real video models all
   implement `WorldModel`, so the eval loop never branches on model type.

5. **Cache Phi output by `(video_sha, phi_version)`.** You will re-run the
   statistics dozens of times while tuning alpha and the persistence window.
   Worth days.

## Gates

Run in order. Each is cheap and each can kill the design early.

**GATE 0 -- fork determinism.** `runner.gate_fork_determinism(scene)`.
Deep-copy `mjData`, apply a null intervention, step both branches, assert
bitwise identity. If this fails, causal-consistency is unmeasurable in this
engine. Run it before writing anything else.

**GATE 1a -- false-termination floor.** Held-out references must terminate at
rate ~alpha. Higher means the detector or the statistic is wrong.

**GATE 1b -- sensitivity and specificity.** Each mutant must fire on the
*right* detector at the *right* time. A mutant that fires on the wrong risk
means the detectors are not separable.

**GATE 2 -- instrument tax.** Same mutants, measured through video instead of
ground-truth state. The event-time difference is the number you publish.

**GATE 3 -- baseline sanity.** Constant-velocity should show a long validity
interval in free flight and a short one under contact. If it does not, the
metric is wrong, not the baseline.

## Note from the first run

`velocity_freeze` injected at t=2.0s did not fire, and this is correct: by
then the ball has settled into steady horizontal motion, so freezing velocity
is a no-op. Injected during the fall it fires every time. A defect is only
detectable when the true dynamics are actually changing at the injection
point -- which is the minimum-detectable-defect concept, and it means mutant
injection times must be sampled where the dynamics are live.

## Scope for the first six weeks

In: entity persistence (R2), dynamics (R5), calibration. 15 scenarios,
M=20, 3 lambda levels, 15s horizon, 3-4 open-weight models.

Out: relational/support detection and its learned relation head, frame
structure, causal consistency, multi-view, hazard regression.