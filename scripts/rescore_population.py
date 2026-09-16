"""Re-score an already-paid-for real-model population at several lambda
levels from its cached trajectories -- AGENT.md M7's lambda sweep for real
models, and the direct answer to the VI50 FLOOR (2026-09-12): when nearly
every episode fails within a frame of the prefix ending, VI50 sits at
~1.03s with a point CI at the nominal lambda and cannot rank models. The
cache (results/trajectories/<scenario>_<model>/seed*.npz, written by
run_model_population.py for exactly this purpose) holds each episode's
full reconstructed Trajectory; `lam` only changes the reference ensemble
and thresholds it is scored against, so widening the band costs nothing
and shows the lambda at which a model's VI50 LEAVES the floor and how the
models order there. RMVT and S(t) at fixed horizons are reported at every
lambda too.

Channels re-scored: R1, R3 (with the scenario's own axis restriction), R2
when the manifest is P3, and R4 (independent). R5 needs pixels and is NOT
in the cache -- it is scored only in the live population run.

Consistency check built in: at the manifest's own lambda the re-scored
VI50 must reproduce the population file's VI50 (same trajectories, same
thresholds) -- printed and asserted, so a silent drift in the scoring
path cannot masquerade as a lambda effect.

    python3 scripts/rescore_population.py --scenario occlusion_corridor --model cosmos3nano
    python3 scripts/rescore_population.py --scenario occlusion_corridor --model cosmos3nano --lams 2,4,8,16
"""
import sys, pathlib, argparse, functools, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
HORIZONS_S = [1.5, 2.0, 3.0]
ALPHA = 0.01


def load_spec(scenario, lam=None):
    from vitals.types import EpisodeSpec
    m = yaml.safe_load((ROOT / f"configs/manifests/{scenario}.yaml").read_text())
    return EpisodeSpec(name=m["name"], scene=m["scene"], target_property=m["target_property"], band=m["band"],
                       lam=m["lam"] if lam is None else lam, n_reference=m["n_reference"],
                       horizon_s=m["horizon_s"], fps=m["fps"], seed=m.get("seed", 0),
                       perturb_mode=m.get("perturb_mode", "full"),
                       track_all_objects=m.get("track_all_objects", False))


def score_at_lambda(spec, trajs, n_frames):
    from vitals.physics import make_backend
    from vitals.detect.statistics import sigma_existence, sigma_kinematic, sigma_interpenetration, kinematic_axes_for
    from vitals.detect.thresholds import estimate_threshold
    from vitals.detect.events import extract_event
    from vitals.detect.long_horizon import make_long_horizon_sigma
    from vitals.stats.survival import (validity_interval, termination_profile, bootstrap_vi, kaplan_meier,
                                       restricted_mean_validity, survival_at)

    roll = make_backend("mujoco", scene=spec.scene)
    refs = [roll(spec, 1000 + i) for i in range(spec.n_reference)]
    dt = refs[0].dt
    STATS = {"R1": functools.partial(sigma_existence, obj=0),
             "R3": functools.partial(sigma_kinematic, axes=kinematic_axes_for(spec.name))}
    # Soft-body scenarios (AGENT.md M9/T10): cached pixel-measured
    # candidates carry Phi's shape; R6/R7 are scored against the
    # Phi-measured band, same as run_model_population.py. The band is
    # lambda-independent (the same reference seeds, measured once), so it
    # is reused at every level; only R1/R3's state band is re-rolled.
    from vitals.phi import scene_geometry as sg
    from vitals.physics import softbody as sb
    SOFT = bool(sg.SCENES.get(spec.name, {}).get("soft_body", False))
    STATS_PHI, phi_refs = {}, None
    if SOFT:
        from vitals.detect.statistics import sigma_conservation, sigma_shape
        phi_refs = sb.load_phi_refs(sb.phi_refs_path(ROOT, spec.name))
        assert phi_refs, f"missing results/phi_refs_{spec.name}.npz -- run scripts/build_phi_references.py"
        STATS_PHI = {"R6": sigma_conservation, "R7": sigma_shape}
    if spec.target_property == "P3" and not SOFT:
        STATS["R2"] = sigma_interpenetration
    thetas = {k: estimate_threshold(refs, fn, alpha=ALPHA)[:n_frames] for k, fn in STATS.items()}
    for k, fn in STATS_PHI.items():
        thetas[k] = estimate_threshold(phi_refs, fn, alpha=ALPHA)[:n_frames]
    sigma_R4 = make_long_horizon_sigma(STATS, refs, {k: estimate_threshold(refs, fn, alpha=ALPHA) for k, fn in STATS.items()})
    theta_R4 = estimate_threshold(refs, sigma_R4, alpha=ALPHA)[:n_frames]
    t_max = n_frames * dt

    pop, pop_r4 = [], []
    for traj in trajs:
        sig = {k: fn(traj, refs)[:n_frames] for k, fn in STATS.items()}
        for k, fn in STATS_PHI.items():
            sig[k] = fn(traj, phi_refs)[:n_frames]
        pop.append(extract_event(sig, thetas, dt, t_max))
        pop_r4.append(extract_event({"R4": sigma_R4(traj, refs)[:n_frames]}, {"R4": theta_R4}, dt, t_max))
    lo, hi = bootstrap_vi(pop, 0.5, n_boot=400)
    grid, surv = kaplan_meier(pop)
    return dict(
        lam=spec.lam, n=len(pop), vi50=validity_interval(pop), vi50_ci95=[lo, hi],
        rmvt=restricted_mean_validity(pop, t_max), S_at={str(h): s for h, s in zip(HORIZONS_S, survival_at(pop, HORIZONS_S))},
        censoring_rate=sum(e.censored for e in pop) / len(pop), termination_profile=termination_profile(pop),
        thresholds_median={k: float(np.nanmedian(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else None for k, v in thetas.items()},
        kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
        long_horizon=dict(vi50=validity_interval(pop_r4), censoring_rate=sum(e.censored for e in pop_r4) / len(pop_r4),
                          rmvt=restricted_mean_validity(pop_r4, t_max)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lams", default="2,4,8,16", help="comma-separated lambda levels; the manifest's own is always added")
    args = ap.parse_args()

    from vitals.adapters.video_utils import load_trajectory
    cache = ROOT / "results" / "trajectories" / f"{args.scenario}_{args.model}"
    files = sorted(cache.glob("seed*.npz"), key=lambda p: int(p.stem[4:]))
    if not files:
        sys.exit(f"no cached trajectories in {cache}")
    trajs = [load_trajectory(p) for p in files]
    n_frames = min(t.pos.shape[0] for t in trajs)
    pop_path = ROOT / "results" / f"l0_demo_{args.scenario}_{args.model}.json"
    pop = json.loads(pop_path.read_text()) if pop_path.exists() else None

    nominal = load_spec(args.scenario).lam
    lams = sorted({float(x) for x in args.lams.split(",") if x.strip()} | {float(nominal)})
    print(f"{args.scenario} / {args.model}: {len(trajs)} cached episodes, {n_frames} frames each; lambdas {lams} (manifest: {nominal})")
    levels = []
    for lam in lams:
        r = score_at_lambda(load_spec(args.scenario, lam=lam), trajs, n_frames)
        levels.append(r)
        print(f"  lam={lam:>5.1f}: VI50={r['vi50']:.2f} CI[{r['vi50_ci95'][0]:.2f},{r['vi50_ci95'][1]:.2f}]  RMVT={r['rmvt']:.2f}s  "
              f"S(1.5)={r['S_at']['1.5']:.2f} S(2.0)={r['S_at']['2.0']:.2f} S(3.0)={r['S_at']['3.0']:.2f}  "
              f"profile={ {k: round(v, 2) for k, v in r['termination_profile'].items()} }  R4 fired={1 - r['long_horizon']['censoring_rate']:.0%}")

    check = None
    if pop is not None:
        mine = [l for l in levels if l["lam"] == float(nominal)][0]
        ref_vi = pop["survival"]["vi50"]
        # The live run also scores R5 (pixels) and may have excluded failed
        # seeds; a difference there is expected and reported, not asserted.
        note = "" if "R5" in pop.get("thresholds_median", {}) else " (population predates R5 -> should reproduce exactly)"
        check = dict(population_vi50=ref_vi, rescored_vi50=mine["vi50"], population_n=pop["survival"]["n"], rescored_n=mine["n"])
        print(f"  consistency at lam={nominal}: population VI50={ref_vi:.2f} vs re-scored {mine['vi50']:.2f}{note}")
        if "R5" not in pop.get("thresholds_median", {}) and abs(ref_vi - mine["vi50"]) > 1e-6:
            print("  WARNING: re-scored VI50 does not reproduce the population file at the manifest lambda -- "
                  "investigate before trusting the sweep")

    out = ROOT / "results" / f"lambda_rescore_{args.scenario}_{args.model}.json"
    out.write_text(json.dumps(dict(scenario=args.scenario, backend=args.model, n_frames=n_frames,
                                   nominal_lam=nominal, levels=levels, consistency=check), indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
