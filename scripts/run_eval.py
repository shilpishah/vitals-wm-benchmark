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
design (fair_crps, stats/scoring.py), not built yet -- see
adapters/base.py's docstring.

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
from vitals.detect.statistics import sigma_existence, sigma_kinematic, kinematic_axes_for
from vitals.detect.thresholds import estimate_threshold
from vitals.detect.events import extract_event
from vitals.stats.survival import (validity_interval, termination_profile,
                                   bootstrap_vi, kaplan_meier)

ROOT = pathlib.Path(__file__).resolve().parents[1]
# R5's own axes restriction is rebound in main() once --manifest is known
# (AGENT.md M2.7) -- MUST match run_l0_demo.py/run_gate2.py/run_model_
# population.py's own identical construction. This script's whole
# purpose is a same-scenario, same-theta comparison against those other
# scripts' own results (GATE 3, and now the score report's own "Δ vs.
# baseline" cards) -- a mismatched axes restriction would silently
# compare numbers computed under two different detector calibrations.
STATS = {"R2": sigma_existence, "R5": sigma_kinematic}


def load_spec(manifest_path):
    manifest = yaml.safe_load((ROOT / manifest_path).read_text())
    return EpisodeSpec(name=manifest["name"], scene=manifest["scene"],
                        target_property=manifest["target_property"], band=manifest["band"],
                        lam=manifest["lam"], n_reference=manifest["n_reference"],
                        horizon_s=manifest["horizon_s"], fps=manifest["fps"],
                        seed=manifest.get("seed", 0),
                        perturb_mode=manifest.get("perturb_mode", "full"))


def main():
    global STATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
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
    STATS = {"R2": sigma_existence,
             "R5": functools.partial(sigma_kinematic, axes=kinematic_axes_for(spec.name))}
    roll = make_backend(args.backend, scene=spec.scene)
    model = ADAPTERS[args.adapter]()

    refs = [roll(spec, 1000 + i) for i in range(spec.n_reference)]
    dt, t_max = refs[0].dt, float(refs[0].t[-1])
    thetas = {k: estimate_threshold(refs, fn, alpha=args.alpha) for k, fn in STATS.items()}

    t_c_frames = max(2, int(round(args.t_c / dt)))    # >=2 frames: baselines need a velocity estimate
    remaining_s = spec.horizon_s - t_c_frames * dt

    print(f"scenario={spec.name}  target_property={spec.target_property}  "
          f"adapter={model.name}  backend={args.backend}")
    print(f"reference M={spec.n_reference}  conditioning={args.t_c:.2f}s "
          f"({t_c_frames} frames)  T_max={t_max:.1f}s\n")

    events = []
    for ep in range(args.n_episodes):
        # Held-out realization: a seed range disjoint from the reference
        # ensemble's (1000..1000+M-1), never scored as its own reference.
        full = roll(spec, seed=50_000 + ep)
        conditioning = prefix_of(full, t_c_frames)

        candidates = model.predict(conditioning, remaining_s, args.n_samples)
        cand = concat_trajectory(conditioning, candidates[0])

        sigmas = {k: fn(cand, refs) for k, fn in STATS.items()}
        events.append(extract_event(sigmas, thetas, dt, t_max))

    vi50 = validity_interval(events, 0.5)
    lo, hi = bootstrap_vi(events, 0.5, n_boot=1000)
    prof = termination_profile(events)
    cens = sum(e.censored for e in events) / len(events)

    print(f"n={len(events)} episodes")
    print(f"VI_50            = {vi50:.2f}s   95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"censoring        = {cens:.2f}")
    print("termination profile = " + ", ".join(f"{k}:{v:.2f}" for k, v in prof.items()))

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
        thresholds_median=dict(R2=float(np.median(thetas["R2"][np.isfinite(thetas["R2"])])),
                                R5=float(np.median(thetas["R5"][np.isfinite(thetas["R5"])]))),
        gate_1a=dict(false_termination_rate=None, target_alpha=args.alpha, passed=None),
        gate_1b=[],
        survival=dict(
            n=len(events), vi50=vi50, vi50_ci95=[lo, hi], censoring_rate=cens,
            termination_profile=prof,
            kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
            events=[dict(time=e.time, risk=e.risk, censored=e.censored) for e in events],
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
