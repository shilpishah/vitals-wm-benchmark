"""M7: lambda sweep + rank-order stability across levels.

T5's own risk (AGENT.md): "reference spread is a design PARAMETER, not a
measurement" -- every threshold in this project is calibrated against a
reference ensemble built at one, hand-tuned `lam` per manifest. M7's stated
acceptance criterion is that a comparative claim (which of several
conditions has the longer validity interval) survives moving `lam` away
from that one tuned value -- if it doesn't, that's a real negative validity
finding to publish, not something to quietly re-tune away.

Truth + both baselines (ConstantVelocity, CopyLastState) are pure local
MuJoCo physics + closed-form prediction -- every level is free, rebuilt
fresh at each lam. Real video models (Cosmos/Wan) are DIFFERENT: `lam`
only reparameterizes the reference ensemble/thresholds, never how a
candidate trajectory is reconstructed, so once an episode's full
reconstructed Trajectory is cached to disk (`scripts/run_model_
population.py`'s own `save_trajectory` call, 2026-08 -- see that script's
docstring), it can be re-scored at ANY number of lam levels for free too.
This script auto-detects `results/trajectories/<scenario>_<model>/*.npz`
for every name in `REAL_MODEL_BACKENDS` and, if present, adds that model
as an extra condition at every level -- no code change needed when a new
model's cache shows up. If no cache exists for a scenario (yet), that
scenario's sweep silently falls back to Truth+baselines only, exactly
this script's ORIGINAL scope -- never an error, since real-model caching
is opt-in per population run, not guaranteed to exist.

For each of `--manifest` (repeatable; defaults to occlusion_corridor and
ramp_descent_high_friction, the two scenarios with published real-model
comparisons) and each multiplier in `--lam-multipliers` of that manifest's
OWN frozen lam (never the manifest file itself -- 3.12's "decide once"
applies to the shipped calibration, not to this diagnostic sweep):
rebuilds the reference ensemble and calibrated thresholds at that lam,
reruns GATE 1a/1b (documents how sensitivity/specificity respond, the
literal content of the T5 risk), and reruns Truth + both baseline
populations. Persists one `results/lambda_sweep_<scenario>.json` per
scenario with every level's data, and prints a summary: VI_50 rank order
per level, Spearman rank correlation between adjacent levels, and whether
bootstrap_vi_delta's sign/significance for ConstantVelocity vs
CopyLastState stays stable across the whole grid.

    python scripts/run_lambda_sweep.py
    python scripts/run_lambda_sweep.py --manifest configs/manifests/occlusion_corridor.yaml \\
        --lam-multipliers 0.5,1,2 --backend mujoco
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import functools
import json
from dataclasses import replace

import numpy as np
import yaml

from vitals.types import EpisodeSpec
from vitals.physics import make_backend
from vitals.mutants import library as mut
from vitals.adapters import REAL_MODEL_BACKENDS
from vitals.adapters.base import ADAPTERS, concat_trajectory, prefix_of
from vitals.adapters.video_utils import load_trajectory
from vitals.detect.statistics import sigma_existence, sigma_kinematic, sigma_interpenetration, kinematic_axes_for
from vitals.detect.thresholds import estimate_threshold
from vitals.detect.events import extract_event
from vitals.stats.survival import (validity_interval, bootstrap_vi, bootstrap_vi_delta,
                                   spearman_rank_correlation)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALPHA = 0.01
DEFAULT_MANIFESTS = ["configs/manifests/occlusion_corridor.yaml",
                     "configs/manifests/ramp_descent_high_friction.yaml"]
BASE_CONDITIONS = ["truth", "constant_velocity", "copy_last_state"]


def load_spec(manifest_path):
    manifest = yaml.safe_load((ROOT / manifest_path).read_text())
    return EpisodeSpec(name=manifest["name"], scene=manifest["scene"],
                        target_property=manifest["target_property"], band=manifest["band"],
                        lam=manifest["lam"], n_reference=manifest["n_reference"],
                        horizon_s=manifest["horizon_s"], fps=manifest["fps"],
                        seed=manifest.get("seed", 0),
                        perturb_mode=manifest.get("perturb_mode", "full"))


def build_reference(roll, spec, n, seed0):
    return [roll(spec, seed0 + i) for i in range(n)]


def calibrate(roll, spec):
    """Same STATS/theta construction as run_l0_demo.py/run_eval.py, R1+R3
    by default -- R2 (sigma_interpenetration) is added below, ONLY when
    `spec.target_property == "P3"` (2026-09: previously never applied
    since neither of this script's own DEFAULT_MANIFESTS is P3-registered
    -- generalized now that occlusion_corridor_interpenetration, the one
    P3 manifest, can also be swept via `--manifest`; a strict no-op for
    every manifest this script was already used against, same gating
    discipline run_l0_demo.py's own R2 wiring established, deliberately
    NOT on K>=2 -- see that script's own comment for the real bug found
    gating on K alone)."""
    refs = build_reference(roll, spec, spec.n_reference, seed0=1000)
    dt, t_max = refs[0].dt, float(refs[0].t[-1])
    stats = {"R1": sigma_existence,
             "R3": functools.partial(sigma_kinematic, axes=kinematic_axes_for(spec.name))}
    if spec.target_property == "P3":
        assert refs[0].K >= 2, (
            f"{spec.name!r} is registered target_property: P3 but its scene has K={refs[0].K} "
            f"objects -- sigma_interpenetration needs obj=0/target=1 both real.")
        stats["R2"] = sigma_interpenetration
    thetas = {k: estimate_threshold(refs, fn, alpha=ALPHA) for k, fn in stats.items()}
    # R1's own obj=0 restriction, for scoring cached real-model (video-
    # reconstructed) trajectories only -- those are always K=1 (Phi's
    # single-object-only reconstruction), same restriction run_gate2.py's
    # own STATS already applies to its own L1 candidates, for the same
    # documented reason (sigma_existence's own docstring: unrestricted
    # whole-count comparison against a K>1 reference ensemble fires R1
    # 100% of the time regardless of tracking quality). DELIBERATELY NOT
    # used for `stats` above (GATE1a/1b/Truth/baseline all stay full-scene
    # K, matching every already-published sweep's own behavior exactly --
    # a strict no-op for occlusion_corridor/ramp_descent_high_friction,
    # both native K=1; only changes behavior for a K>1 scenario's
    # real-model condition, found directly, 2026-08, collision's own K=2
    # scene being the first K>1 scenario swept).
    #
    # R2 in `stats_video` too, when registered above -- unlike R1's
    # obj=0 restriction (which only matters because real-model candidates
    # used to be structurally K=1), a P3 manifest's real-model candidate
    # now genuinely carries K=2 (2026-09, VideoWorldModel's own dual-
    # object reconstruction) via the SAME `merge_secondary` mechanism GATE
    # 2's own instrument check already validated -- so R2 must be scored
    # here exactly like R1/R3 already are, not skipped.
    stats_video = {"R1": functools.partial(sigma_existence, obj=0), "R3": stats["R3"]}
    if "R2" in stats:
        stats_video["R2"] = stats["R2"]
    thetas_video = {k: estimate_threshold(refs, fn, alpha=ALPHA) for k, fn in stats_video.items()}
    return refs, stats, thetas, stats_video, thetas_video, dt, t_max


def score(stats, thetas, dt, t_max, refs, traj):
    sigmas = {k: fn(traj, refs) for k, fn in stats.items()}
    return extract_event(sigmas, thetas, dt, t_max)


NEAR_FLAT_SCENARIOS = ("occlusion_corridor", "collision")


def build_demo_cases(spec, base):
    """GATE 1b cases -- identical construction to run_l0_demo.py's own
    build_demo_cases for a non-P3 manifest (NEAR_FLAT_SCENARIOS ->
    duplicate, everything else -> wrong_gravity; NEAR_FLAT_SCENARIOS' own
    lower R3 threshold needs a smaller jitter sigma to fire reliably,
    except collision, which needs a LARGER one once R3 is X-only
    restricted there -- see run_l0_demo.py's own build_demo_cases
    docstring for the full mechanism). Duplicated rather than imported:
    run_l0_demo.py builds its SPEC/roll from env vars at module import
    time, which would fire real (if cheap) side effects just to reach
    this one pure function. Keep in sync by hand if either changes --
    both are short and reviewed together.

    FIXED (2026-08, found directly running collision through this script
    for the first time): this copy had fallen out of sync with run_l0_
    demo.py/run_gate2.py's own already-established per-scenario tuning --
    it still branched on a bare `occlusion_corridor` prefix check (missing
    collision from the near-flat set entirely, so collision got
    wrong_gravity instead of duplicate) and only had a two-way jitter
    sigma split (0.05/0.15, missing collision's own 0.8). Both silently
    produced GATE1b 3/5 (wrong_gravity and jitter both censored, the same
    near-no-op failure modes already root-caused and fixed elsewhere for
    this exact scenario) instead of the correct 5/5 -- a real bug, not a
    genuine collision-specific GATE1b weakness."""
    t_mid = round(spec.horizon_s * 0.25, 2)
    cases = [
        ("null", mut.null(base)),
        ("vanish", mut.vanish(base, t_star=t_mid)),
        ("velocity_freeze", mut.velocity_freeze(base, t_star=t_mid)),
    ]
    if spec.name.startswith(NEAR_FLAT_SCENARIOS):
        cases.append(("duplicate", mut.duplicate(base, t_star=t_mid)))
    else:
        wg_factor = 0.4 if spec.name == "projectile" else 0.6
        cases.append(("wrong_gravity", mut.wrong_gravity(base, factor=wg_factor)))
    jitter_sigma = 0.05 if spec.name.startswith("occlusion_corridor") else (
        0.8 if spec.name == "collision" else 0.15)
    cases.append((f"jitter sig={jitter_sigma}", mut.jitter(base, sigma=jitter_sigma)))
    return cases


def run_gate1a(roll, spec, refs, stats, thetas, dt, t_max, n):
    held_out = build_reference(roll, spec, n, seed0=9000)
    events = [score(stats, thetas, dt, t_max, refs, h) for h in held_out]
    fp = sum(1 for e in events if not e.censored) / len(events)
    return dict(false_termination_rate=fp, target_alpha=ALPHA, passed=bool(fp <= 5 * ALPHA)), held_out


def run_gate1b(spec, refs, stats, thetas, dt, t_max, base):
    results = []
    for label, m in build_demo_cases(spec, base):
        e = score(stats, thetas, dt, t_max, refs, m.traj)
        ok = (e.risk or None) == (m.risk_expected or None)
        results.append(dict(label=label, risk_expected=m.risk_expected, risk_fired=e.risk,
                            censored=e.censored, ok=ok))
    return results


def truth_population(roll, spec, refs, stats, thetas, dt, t_max, n):
    """The same null/vanish/velocity_freeze/jitter 4-way mix run_l0_demo.py
    persists as `results/l0_demo_<scenario>_mujoco.json` -- the "Truth"
    category render_survival_report.py already plots (category_for:
    backend == "mujoco"). Duplicated for the same reason build_demo_cases
    is -- run_l0_demo.py is not import-safe."""
    mutant_kinds, mutant_probs = ["null", "vanish", "velocity_freeze", "jitter"], [0.25] * 4
    rng = np.random.default_rng(7)
    t_lo, t_hi = spec.horizon_s * 0.15, spec.horizon_s * 0.7
    events = []
    for i in range(n):
        b = roll(spec, 20000 + i)
        kind = rng.choice(mutant_kinds, p=mutant_probs)
        t_star = float(rng.uniform(t_lo, t_hi))
        m = (mut.null(b) if kind == "null" else
             mut.vanish(b, t_star=t_star) if kind == "vanish" else
             mut.velocity_freeze(b, t_star=t_star) if kind == "velocity_freeze" else
             mut.jitter(b, sigma=0.04))
        events.append(score(stats, thetas, dt, t_max, refs, m.traj))
    return events


def baseline_population(roll, spec, refs, stats, thetas, dt, t_max, adapter_name, n, t_c=1.0):
    """Identical to run_eval.py's own episode loop (held-out seeds
    50_000+, disjoint from the reference ensemble's 1000.. range)."""
    model = ADAPTERS[adapter_name]()
    t_c_frames = max(2, int(round(t_c / dt)))
    remaining_s = spec.horizon_s - t_c_frames * dt
    events = []
    for ep in range(n):
        full = roll(spec, 50_000 + ep)
        conditioning = prefix_of(full, t_c_frames)
        candidates = model.predict(conditioning, remaining_s, 1)
        cand = concat_trajectory(conditioning, candidates[0])
        events.append(score(stats, thetas, dt, t_max, refs, cand))
    return events


def load_cached_population(cache_dir):
    """Every `seed*.npz` `save_trajectory`d by a real-model population run
    (`scripts/run_model_population.py`), in sorted-seed order (deterministic,
    not filesystem-listing order). Returns [] if the directory doesn't
    exist -- "no cache for this scenario/model yet" is a normal, silent
    case (see this module's own docstring), not an error."""
    if not cache_dir.is_dir():
        return []
    paths = sorted(cache_dir.glob("seed*.npz"),
                   key=lambda p: int(p.stem.replace("seed", "")))
    return [load_trajectory(p) for p in paths]


def real_model_population(stats, thetas_full, dt, refs, trajs):
    """Re-scores ALREADY-RECONSTRUCTED real-model trajectories against
    THIS level's fresh reference ensemble/thetas -- no GPU/Modal call,
    reconstruction already happened once when the trajectory was cached
    and lam never affects it (this module's own docstring). Real-model
    episodes are shorter than the reference ensemble's own native horizon
    (`run_model_population.py`'s own `--n-continuation-frames` cap), so
    thetas/t_max are sliced to the cached episode's own frame count --
    the identical fix `run_model_population.py` itself applies, taken
    from each cached trajectory's own length rather than re-deriving it
    from CLI args this script never had."""
    if not trajs:
        return []
    n_episode_frames = trajs[0].T
    assert all(t.T == n_episode_frames for t in trajs), (
        "cached trajectories in one cache dir have inconsistent lengths -- "
        "were they generated with different --n-continuation-frames/--prefix-frames?")
    thetas_ep = {k: v[:n_episode_frames] for k, v in thetas_full.items()}
    t_max_ep = n_episode_frames * dt
    return [score(stats, thetas_ep, dt, t_max_ep, refs, t) for t in trajs]


def run_one_level(roll, base_spec, lam_multiplier, n_gate1a, n_truth, n_baseline,
                  real_model_cache_dirs=None):
    spec = replace(base_spec, lam=base_spec.lam * lam_multiplier)
    refs, stats, thetas, stats_video, thetas_video, dt, t_max = calibrate(roll, spec)
    gate_1a, held_out = run_gate1a(roll, spec, refs, stats, thetas, dt, t_max, n_gate1a)
    gate_1b = run_gate1b(spec, refs, stats, thetas, dt, t_max, held_out[0])

    conditions = {
        "truth": truth_population(roll, spec, refs, stats, thetas, dt, t_max, n_truth),
        "constant_velocity": baseline_population(roll, spec, refs, stats, thetas, dt, t_max,
                                                  "constant_velocity", n_baseline),
        "copy_last_state": baseline_population(roll, spec, refs, stats, thetas, dt, t_max,
                                                "copy_last_state", n_baseline),
    }
    for name, cache_dir in (real_model_cache_dirs or {}).items():
        trajs = load_cached_population(cache_dir)
        if trajs:
            conditions[name] = real_model_population(stats_video, thetas_video, dt, refs, trajs)

    vi50 = {name: validity_interval(ev, 0.5) for name, ev in conditions.items()}
    ci = {name: bootstrap_vi(ev, 0.5, n_boot=400) for name, ev in conditions.items()}

    # Pairwise deltas: always constant_velocity-vs-copy_last_state (the
    # original scope), PLUS -- whenever a real model's own cache was found
    # -- that model against each baseline, the actually decision-relevant
    # comparison ("does this real model beat a naive predictor, at THIS
    # lam level").
    pairs = [("constant_velocity", "copy_last_state")]
    for name in conditions:
        if name not in BASE_CONDITIONS:
            pairs += [(name, "constant_velocity"), (name, "copy_last_state")]
    deltas = {}
    for a, b in pairs:
        point, lo, hi = bootstrap_vi_delta(conditions[a], conditions[b], q=0.5, n_boot=400)
        deltas[f"{a}_vs_{b}"] = dict(point=point, ci95=[lo, hi])

    return dict(
        lam_multiplier=lam_multiplier, lam=spec.lam, base_lam=base_spec.lam,
        gate_1a=gate_1a, gate_1b=gate_1b, condition_names=sorted(conditions),
        vi50=vi50, vi50_ci95={k: list(v) for k, v in ci.items()},
        deltas=deltas,
        events={name: [dict(time=e.time, risk=e.risk, censored=e.censored) for e in ev]
                for name, ev in conditions.items()},
    )


def summarize(scenario_name, levels):
    # Same condition set at every level of one scenario's sweep (real-model
    # caches either exist for this scenario or don't -- not level-dependent),
    # so level 0's own condition_names is representative of all of them.
    # Fixed BASE_CONDITIONS order first (stable across scenarios/runs), any
    # real models appended alphabetically after.
    conditions = [c for c in BASE_CONDITIONS if c in levels[0]["vi50"]] + \
        sorted(c for c in levels[0]["vi50"] if c not in BASE_CONDITIONS)

    print(f"\n{'='*70}\n{scenario_name}\n{'='*70}")
    print(f"{'lam (xbase)':<14}{'GATE1a':<10}{'GATE1b':<10}" +
          "".join(f"{c+' VI50':<24}" for c in conditions))
    for lv in levels:
        g1a = "PASS" if lv["gate_1a"]["passed"] else "FAIL"
        g1b_ok = sum(1 for c in lv["gate_1b"] if c["ok"])
        g1b = f"{g1b_ok}/{len(lv['gate_1b'])}"
        row = f"{lv['lam_multiplier']:<14.2f}{g1a:<10}{g1b:<10}"
        for c in conditions:
            lo, hi = lv["vi50_ci95"][c]
            row += f"{lv['vi50'][c]:.2f} [{lo:.2f},{hi:.2f}]".ljust(24)
        print(row)

    print(f"\npairwise deltas, by level:")
    for lv in levels:
        print(f"  lam x{lv['lam_multiplier']:.2f}:")
        for pair, d in lv["deltas"].items():
            lo, hi = d["ci95"]
            sig = "SIGNIFICANT" if (hi < 0 or lo > 0) else "not significant"
            print(f"    {pair}: delta={d['point']:+.2f} s  CI[{lo:+.2f},{hi:+.2f}]  {sig}")

    print("\nSpearman rank correlation of VI_50 order, adjacent levels:")
    vi50_series = {c: [lv["vi50"][c] for lv in levels] for c in conditions}
    for i in range(len(levels) - 1):
        a = [vi50_series[c][i] for c in conditions]
        b = [vi50_series[c][i + 1] for c in conditions]
        rho = spearman_rank_correlation(a, b)
        print(f"  x{levels[i]['lam_multiplier']:.2f} -> x{levels[i+1]['lam_multiplier']:.2f}: rho={rho:.3f}")
    a_first = [vi50_series[c][0] for c in conditions]
    b_last = [vi50_series[c][-1] for c in conditions]
    rho_end_to_end = spearman_rank_correlation(a_first, b_last)
    print(f"  x{levels[0]['lam_multiplier']:.2f} -> x{levels[-1]['lam_multiplier']:.2f} "
          f"(full sweep): rho={rho_end_to_end:.3f}")
    return rho_end_to_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", default=None,
                    help="repeatable; defaults to occlusion_corridor + ramp_descent_high_friction")
    ap.add_argument("--lam-multipliers", default="0.5,1.0,2.0")
    ap.add_argument("--backend", default="mujoco")
    ap.add_argument("--n-gate1a", type=int, default=60)
    ap.add_argument("--n-truth", type=int, default=80)
    ap.add_argument("--n-baseline", type=int, default=40)
    args = ap.parse_args()

    manifests = args.manifest or DEFAULT_MANIFESTS
    multipliers = [float(x) for x in args.lam_multipliers.split(",")]

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    for manifest_path in manifests:
        base_spec = load_spec(manifest_path)
        roll = make_backend(args.backend, scene=base_spec.scene)

        real_model_cache_dirs = {
            model: (ROOT / "results" / "trajectories" / f"{base_spec.name}_{model}")
            for model in sorted(REAL_MODEL_BACKENDS)
        }
        found = {name: d for name, d in real_model_cache_dirs.items() if d.is_dir()}

        print(f"\nsweeping lam for {base_spec.name} (base lam={base_spec.lam}), "
              f"multipliers={multipliers}, backend={args.backend}, "
              f"real-model caches found: {sorted(found) or 'none (Truth+baselines only)'}...")
        levels = []
        for mult in multipliers:
            print(f"  lam x{mult:.2f} (={base_spec.lam * mult:.3f})...")
            levels.append(run_one_level(roll, base_spec, mult, args.n_gate1a,
                                        args.n_truth, args.n_baseline, real_model_cache_dirs))

        rho_end_to_end = summarize(base_spec.name, levels)

        out_path = out_dir / f"lambda_sweep_{base_spec.name}.json"
        with open(out_path, "w") as f:
            json.dump(dict(scenario=base_spec.name, backend=args.backend,
                           lam_multipliers=multipliers, levels=levels,
                           rank_correlation_full_sweep=rho_end_to_end), f, indent=2)
        print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
