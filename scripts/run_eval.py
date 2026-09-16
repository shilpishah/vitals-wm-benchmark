"""GATE 3: score an actual WorldModel candidate against the reference
ensemble -- not a hand-planted mutant. Same detect/ + stats/ pipeline
validated by run_l0_demo.py (M2); this is the first script in the repo that
evaluates a MODEL rather than the instrument.

Per AGENT.md 3.7, conditioning must come from a realization NOT in the
reference ensemble R -- otherwise the model is scored against a
distribution containing its own conditioning, biasing every validity
interval upward. Per 3.10, the bootstrap CI resamples EPISODES (each
episode = one independent held-out conditioning draw), never the N samples
within one episode -- those are correlated by construction (same
conditioning, same model), and only ONE of them (candidates[0]) feeds the
event/survival pipeline per episode. N>1 exists for the P6 distributional
design (stats/scoring.py, built 2026-09-11, `score_episode`) -- not wired
into this script; see adapters/base.py's docstring.

Manifest-driven like run_l0_demo.py, and deliberately runnable against
EITHER scenario without modification -- GATE 3 is itself a two-scenario
comparison: constant_velocity should show a LONG validity interval on
occlusion_corridor (already rolling at roughly constant decelerating
speed -- close to free flight) and a SHORT one on ramp_descent (incline
acceleration, then a sharp direction change at the ramp/floor transition --
contact-dominated). If it doesn't show that split, the metric is wrong, not
the baseline.

    python scripts/run_eval.py --adapter constant_velocity
    python scripts/run_eval.py --adapter constant_velocity \\
        --manifest configs/manifests/occlusion_corridor.yaml
    python scripts/run_eval.py --adapter copy_last_state --backend mujoco
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import functools
import json
import os
import numpy as np
import yaml
from vitals.types import EpisodeSpec
from vitals.physics import make_backend
from vitals.adapters.base import ADAPTERS, concat_trajectory, prefix_of
from vitals.detect.statistics import (sigma_existence, sigma_kinematic, sigma_interpenetration,
                                      kinematic_axes_for)
from vitals.detect.thresholds import estimate_threshold
from vitals.detect.events import extract_event
from vitals.detect.long_horizon import make_long_horizon_sigma
from vitals.stats.survival import (validity_interval, termination_profile,
                                   bootstrap_vi, kaplan_meier,
                                   restricted_mean_validity, bootstrap_rmvt, survival_at)
HORIZONS_S = [1.5, 2.0, 3.0]   # same fixed horizons as run_model_population.py

ROOT = pathlib.Path(__file__).resolve().parents[1]
# STATS is built in main() once --manifest is known, and MUST be the same
# construction run_model_population.py uses for real models (R1 restricted
# to obj=0, R3's own per-scenario axes, R2 gated on target_property == "P3",
# R4 built on top of whichever base channels are active). This script's
# whole purpose is a same-scenario, same-theta comparison against real-
# model results (GATE 3, and the score report's own "Δ vs. baseline"
# cards) -- a mismatched channel set or calibration would silently
# compare numbers computed under two different detectors.
#
# 2026-09: FOUND AS A REAL GAP while fixing the score report -- this
# script had stayed at R1+R3 (unrestricted R1, no R2, no R4, hardcoded
# thresholds_median, `track_all_objects` dropped by load_spec) after the
# population script gained all of those, which is exactly why billiards /
# occlusion_corridor_interpenetration had no baseline lines at all and why
# no baseline file carried a `long_horizon` block. Baselines are scored
# over the manifest's FULL horizon (real-model runs are truncated to
# prefix+continuation frames); every channel here is causal and the LOO
# thresholds are per-frame, so the report re-censors a baseline at each
# model page's own t_max rather than this script producing one file per
# episode length.


def load_spec(manifest_path):
    manifest = yaml.safe_load((ROOT / manifest_path).read_text())
    return EpisodeSpec(name=manifest["name"], scene=manifest["scene"],
                        target_property=manifest["target_property"], band=manifest["band"],
                        lam=manifest["lam"], n_reference=manifest["n_reference"],
                        horizon_s=manifest["horizon_s"], fps=manifest["fps"],
                        seed=manifest.get("seed", 0),
                        perturb_mode=manifest.get("perturb_mode", "full"),
                        track_all_objects=manifest.get("track_all_objects", False))


def main():
    global STATS
    ap = argparse.ArgumentParser()
    # "truth" (2026-09-12, user: "why is this not being compared against the
    # ideal?"): the PHYSICS IDEAL -- the candidate IS the held-out true
    # realization, scored through the identical thresholds. Its survival is
    # 1 minus the calibrated false-termination rate (~alpha), the ceiling
    # any predictor could reach in state space. Written as backend
    # "truth"; the score report draws it as "ideal (physics)". The
    # instrument ceiling (the same truth rendered -> codec -> Phi) is
    # run_model_population.py's own `--model truth_render`.
    ap.add_argument("--adapter", required=True, choices=sorted(ADAPTERS) + ["truth"])
    ap.add_argument("--manifest", default="configs/manifests/ramp_descent.yaml")
    ap.add_argument("--backend", default=os.environ.get("VITALS_BACKEND", "synthetic"))
    ap.add_argument("--n-episodes", type=int, default=40,
                     help="independent held-out conditioning draws -- the bootstrap "
                          "resamples over these (3.10), never over --n-samples")
    ap.add_argument("--n-samples", type=int, default=1,
                     help="candidates per episode from the model (only candidates[0] "
                          "is scored for now -- see module docstring)")
    ap.add_argument("--t-c", type=float, default=1.0, help="conditioning length, seconds")
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    spec = load_spec(args.manifest)
    STATS = {"R1": functools.partial(sigma_existence, obj=0),
             "R3": functools.partial(sigma_kinematic, axes=kinematic_axes_for(spec.name))}
    roll = make_backend(args.backend, scene=spec.scene)

    class _Truth:   # the ideal: predict() is never called; main() uses the held-out rollout itself
        name = "truth"
    model = _Truth() if args.adapter == "truth" else ADAPTERS[args.adapter]()

    refs = [roll(spec, 1000 + i) for i in range(spec.n_reference)]
    dt, t_max = refs[0].dt, float(refs[0].t[-1])
    # R2: same gating rule as run_model_population.py / run_l0_demo.py
    # (pre-registered P3, not K>=2 -- see their comments for the bug that
    # gating on K alone caused).
    # Soft-body scenarios (AGENT.md M9): state-space candidates (`truth`)
    # get R6 conservation / R7 shape against the state-space band, with
    # shape descriptors attached through the scenario's registered camera
    # -- the same wiring as run_l0_demo.py's SOFT branch.
    from vitals.phi import scene_geometry as sg
    SOFT = bool(sg.SCENES.get(spec.name, {}).get("soft_body", False))
    if SOFT:
        from vitals.detect.statistics import sigma_conservation, sigma_shape
        from vitals.physics import softbody as sb
        from vitals.render.mujoco_renderer import camera_pose
        _cp, _cm, _fv = camera_pose(spec.scene, sg.get(spec.name)["camera"][0])
        _base_roll = roll

        def roll(spec_, seed):   # noqa: F811 -- soft scenes attach shape to every state rollout
            return sb.attach_shape(_base_roll(spec_, seed), _cp, _cm, _fv, 640, 360)
        refs = [roll(spec, 1000 + i) for i in range(spec.n_reference)]
        STATS = dict(STATS, R6=sigma_conservation, R7=sigma_shape)
    if spec.target_property == "P3" and not SOFT:
        assert refs[0].K >= 2, f"{spec.name!r} is P3 but its scene has K={refs[0].K} objects"
        STATS = dict(STATS, R2=sigma_interpenetration)
    thetas = {k: estimate_threshold(refs, fn, alpha=args.alpha) for k, fn in STATS.items()}
    # R4 (long-horizon/cumulative consistency): built on the active base
    # channels, calibrated the same LOO way, scored as an INDEPENDENT
    # event -- never merged into STATS/thetas (events.py's own docstring).
    sigma_R4 = make_long_horizon_sigma(STATS, refs, thetas)
    theta_R4 = estimate_threshold(refs, sigma_R4, alpha=args.alpha)

    t_c_frames = max(2, int(round(args.t_c / dt)))    # >=2 frames: baselines need a velocity estimate
    remaining_s = spec.horizon_s - t_c_frames * dt

    print(f"scenario={spec.name}  target_property={spec.target_property}  "
          f"adapter={model.name}  backend={args.backend}")
    print(f"reference M={spec.n_reference}  conditioning={args.t_c:.2f}s "
          f"({t_c_frames} frames)  T_max={t_max:.1f}s\n")

    events, events_r4 = [], []
    for ep in range(args.n_episodes):
        # Held-out realization: a seed range disjoint from the reference
        # ensemble's (1000..1000+M-1), never scored as its own reference.
        full = roll(spec, seed=50_000 + ep)
        conditioning = prefix_of(full, t_c_frames)

        if args.adapter == "truth":
            cand = full                      # the realization itself -- a perfect prediction
        else:
            candidates = model.predict(conditioning, remaining_s, args.n_samples)
            cand = concat_trajectory(conditioning, candidates[0])

        sigmas = {k: fn(cand, refs) for k, fn in STATS.items()}
        events.append(extract_event(sigmas, thetas, dt, t_max))
        events_r4.append(extract_event({"R4": sigma_R4(cand, refs)}, {"R4": theta_R4}, dt, t_max))

    vi50 = validity_interval(events, 0.5)
    lo, hi = bootstrap_vi(events, 0.5, n_boot=1000)
    prof = termination_profile(events)
    cens = sum(e.censored for e in events) / len(events)

    vi50_r4 = validity_interval(events_r4, 0.5)
    lo_r4, hi_r4 = bootstrap_vi(events_r4, 0.5, n_boot=1000)
    cens_r4 = sum(e.censored for e in events_r4) / len(events_r4)
    grid_r4, surv_r4 = kaplan_meier(events_r4)

    print(f"n={len(events)} episodes   channels={sorted(STATS)} + R4")
    print(f"VI_50            = {vi50:.2f}s   95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"censoring        = {cens:.2f}")
    print("termination profile = " + ", ".join(f"{k}:{v:.2f}" for k, v in prof.items()))
    print(f"R4 (long-horizon): fired on {1 - cens_r4:.0%} of episodes, VI_50 = {vi50_r4}")

    grid, surv = kaplan_meier(events)
    print("\nS(t):  " + "  ".join(
        f"{t:.1f}s={s:.2f}" for t, s in zip(grid[::max(1, len(grid)//6)],
                                            surv[::max(1, len(surv)//6)])))

    # Persisted so this drops into the SAME visualizer (render_survival_
    # report.py) every other population already does -- previously this
    # only ever printed to console (2026-08, found while answering "can
    # you not compare it against all the ground truth stuff we did
    # before": ConstantVelocity/CopyLastState already run through the
    # IDENTICAL WorldModel.predict() pipeline Cosmos does, genuinely
    # comparable to it, just never saved anywhere the report could pick
    # up). `backend` is the ADAPTER's own name, not "mujoco" -- these are
    # real (if trivial) model predictions through the adapter interface,
    # not privileged ground truth, and must render with the same "REAL
    # MODEL" labeling Cosmos gets, not be silently treated as another
    # ground-truth validation curve.
    result = dict(
        scenario=spec.name, target_property=spec.target_property, backend=model.name,
        M=spec.n_reference, lam=spec.lam, t_max=t_max,
        # Derived from what was actually scored (same fix as run_model_
        # population.py: a hardcoded {R1, R3} silently hid R2 there).
        thresholds_median={k: float(np.median(v[np.isfinite(v)])) for k, v in thetas.items()},
        gate_1a=dict(false_termination_rate=None, target_alpha=args.alpha, passed=None),
        gate_1b=[],
        n_requested=args.n_episodes, failed_seeds=[],
        survival=dict(
            n=len(events), vi50=vi50, vi50_ci95=[lo, hi], censoring_rate=cens,
            termination_profile=prof,
            rmvt=restricted_mean_validity(events, t_max),
            rmvt_ci95=list(bootstrap_rmvt(events, t_max, n_boot=400)),
            S_at={str(h): s for h, s in zip(HORIZONS_S, survival_at(events, HORIZONS_S))},
            kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
            events=[dict(time=e.time, risk=e.risk, censored=e.censored) for e in events],
        ),
        # Same shape as run_model_population.py's own `long_horizon` block,
        # so the score report reads baseline and real-model R4 identically.
        long_horizon=dict(
            n=len(events_r4), vi50=vi50_r4, vi50_ci95=[lo_r4, hi_r4], censoring_rate=cens_r4,
            theta_R4_median=float(np.median(theta_R4[np.isfinite(theta_R4)]))
                             if np.any(np.isfinite(theta_R4)) else None,
            kaplan_meier=dict(t=grid_r4.tolist(), S=surv_r4.tolist()),
            events=[dict(time=e.time, censored=e.censored) for e in events_r4],
        ),
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"l0_demo_{spec.name}_{model.name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
