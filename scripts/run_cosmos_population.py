"""Builds a REAL, small population of Cosmos-Predict2 episodes on
occlusion_corridor, scored through the EXACT SAME calibrated detector
machinery (sigma_existence/sigma_kinematic + estimate_threshold +
extract_event) already used for every other survival curve in this
project (AGENT.md M6, 2026-08: "run a few more episodes to build a
population"). Writes `results/l0_demo_occlusion_corridor_cosmos.json` in
the SAME schema `run_l0_demo.py` produces, so it drops straight into the
existing visualizer (`render_survival_report.py`) alongside the
ground-truth-validation curves already there -- the first entry in that
report that isn't ground-truth-only.

Per episode (REAL GPU cost, two Modal calls -- treat this as expensive,
not free): render the real prefix locally -> call Cosmos's `video2world`
(vitals-cosmos app) -> resample its output to this project's convention ->
run the WHOLE stitched sequence through the real Phi pipeline
(`run_phi_reconstruct_from_frames`, vitals-phi app) -> stitch prefix
(real physics) + reconstructed continuation into one Trajectory
(`adapters.base.concat_trajectory`, the SAME function `CopyLastState`/
`ConstantVelocity` use) -> score it with the SAME calibrated thresholds
used for every occlusion_corridor population so far.

Small population size (5 episodes) is a REAL cost tradeoff, stated
plainly: each episode costs 2 real GPU calls. This is illustrative,
first-real-data output, NOT a statistically powered result -- compare to
the 80-episode synthetic-mutant populations already in
results/l0_demo_occlusion_corridor_mujoco.json. Treat VI50/CI here as
preliminary.

    python3 scripts/run_cosmos_population.py --seeds 1,2,3,4,5
"""
import argparse
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np


def build_episode_trajectory(scenario, spec, seed, prefix_frames, n_continuation_frames, cfg, renderer):
    from vitals.adapters.base import prefix_of, concat_trajectory
    from vitals.adapters.cosmos import SCENARIO_PROMPTS, _write_video, _read_video, _resample_to_target
    from vitals.physics import make_backend
    from vitals.types import Trajectory
    import modal
    import tempfile

    rollout = make_backend("mujoco", scene=spec.scene)
    full = rollout(spec, seed=seed)
    conditioning = prefix_of(full, prefix_frames)

    prefix_frames_obj, prefix_gt = renderer.render(conditioning, cameras=cfg["camera"])
    frame0_mask = (prefix_gt.segmentation[0] == 0)

    with tempfile.TemporaryDirectory() as tmp:
        prefix_path = pathlib.Path(tmp) / "prefix.mp4"
        _write_video(prefix_frames_obj.rgb, prefix_path, fps=spec.fps)
        prefix_bytes = prefix_path.read_bytes()

    print(f"  [seed={seed}] calling Cosmos video2world...")
    cosmos_fn = modal.Function.from_name("vitals-cosmos", "video2world")
    out_bytes = cosmos_fn.remote(prefix_video_bytes=prefix_bytes, prompt=SCENARIO_PROMPTS[scenario],
                                 num_conditional_frames=5, fps=16, resolution="480", seed=0)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = pathlib.Path(tmp) / "continuation.mp4"
        out_path.write_bytes(out_bytes)
        raw_continuation, raw_fps = _read_video(out_path)

    continuation_frames = _resample_to_target(
        raw_continuation, raw_fps, spec.fps, n_continuation_frames,
        (prefix_frames_obj.rgb.shape[1], prefix_frames_obj.rgb.shape[2]))
    all_frames = np.concatenate([prefix_frames_obj.rgb, continuation_frames], axis=0)

    print(f"  [seed={seed}] calling real Phi (SAM2/DINOv2) reconstruction...")
    phi_fn = modal.Function.from_name("vitals-phi", "run_phi_reconstruct_from_frames")
    full_traj = phi_fn.remote(all_frames=all_frames, prefix_len=prefix_frames, scenario_name=scenario,
                              frame0_mask=frame0_mask, cam_pos=prefix_gt.cam_pos, cam_mat=prefix_gt.cam_mat,
                              fovy_deg=prefix_gt.fovy_deg, fps=spec.fps)

    n = full_traj.T - prefix_frames
    continuation_traj = Trajectory(
        t=np.arange(n) * (1.0 / spec.fps), pos=full_traj.pos[prefix_frames:].copy(),
        quat=full_traj.quat[prefix_frames:].copy(), present=full_traj.present[prefix_frames:].copy(),
        names=list(full_traj.names), meta=dict(full_traj.meta))
    return concat_trajectory(conditioning, continuation_traj)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="occlusion_corridor")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--prefix-frames", type=int, default=30)
    parser.add_argument("--n-continuation-frames", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=8,
                        help="episodes run concurrently -- each is 2 real Modal GPU calls (network-bound, "
                             "safe to run in parallel via threads); kept modest by default (not e.g. "
                             "n_episodes) since this is a shared team Modal workspace, not to overwhelm it "
                             "with every episode's containers spinning up at once")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    import yaml
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.detect.statistics import sigma_existence, sigma_kinematic
    from vitals.detect.thresholds import estimate_threshold
    from vitals.detect.events import extract_event
    from vitals.stats.survival import validity_interval, termination_profile, bootstrap_vi, kaplan_meier

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((ROOT / f"configs/manifests/{args.scenario}.yaml").read_text())
    spec = EpisodeSpec(name=manifest["name"], scene=manifest["scene"], target_property=manifest["target_property"],
                        band=manifest["band"], lam=manifest["lam"], n_reference=manifest["n_reference"],
                        horizon_s=manifest["horizon_s"], fps=manifest["fps"], seed=manifest.get("seed", 0),
                        perturb_mode=manifest.get("perturb_mode", "full"))
    M, LAM, ALPHA = spec.n_reference, spec.lam, 0.01
    STATS = {"R2": sigma_existence, "R5": sigma_kinematic}

    print(f"building the SAME M={M} reference ensemble + calibrated thresholds already used for "
          f"every other {args.scenario} population (cheap, local, state-space only)...")
    roll = make_backend("mujoco", scene=spec.scene)
    refs = [roll(spec, 1000 + i) for i in range(M)]
    dt = refs[0].dt
    thetas = {k: estimate_threshold(refs, fn, alpha=ALPHA) for k, fn in STATS.items()}

    # Every Cosmos episode here is shorter than the reference ensemble's own
    # native length (prefix+continuation << the manifest's full horizon_s) --
    # sigma_existence/sigma_kinematic return an array sized to the
    # CANDIDATE's own T (detect/statistics.py), but theta is sized to the
    # references' full T, so the elementwise comparison in extract_event
    # doesn't broadcast unless theta is sliced to match. Same fix
    # scripts/run_gate2.py already established for its own truncated
    # (T_RENDER) candidates -- threshold VALUES for t < episode length are
    # identical either way, this only trims the unused tail. Found the hard
    # way: seed=1's first attempt spent both real GPU calls, then crashed
    # in this LOCAL scoring step on a shape mismatch (90,) vs (240,) --
    # re-run, not lost data, but a real, avoidable cost if this had been
    # checked before running any seeds at all.
    n_episode_frames = args.prefix_frames + args.n_continuation_frames
    thetas_episode = {k: v[:n_episode_frames] for k, v in thetas.items()}
    t_max_episode = n_episode_frames * dt

    cfg = sg.get(args.scenario)
    # NOT shared across threads -- each worker gets its OWN MujocoRenderer.
    # render() itself already constructs a fresh mjModel/mjData/Renderer
    # per call (mujoco_renderer.py's own render() body), so sharing one
    # instance's render() calls across threads would likely be fine too,
    # but a renderer per worker removes any doubt rather than relying on
    # that -- cheap to construct, not worth the risk for a multi-hour run.

    def run_one(seed):
        renderer = MujocoRenderer(spec.scene, height=240, width=320)
        print(f"episode seed={seed}: starting...")
        traj = build_episode_trajectory(args.scenario, spec, seed, args.prefix_frames,
                                        args.n_continuation_frames, cfg, renderer)
        sigmas = {k: fn(traj, refs) for k, fn in STATS.items()}
        e = extract_event(sigmas, thetas_episode, dt, t_max_episode)
        print(f"episode seed={seed}: -> risk={e.risk} time={e.time if not e.censored else 'censored'}")
        return seed, e

    import concurrent.futures
    results_by_seed = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_one, seed): seed for seed in seeds}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            seed = futures[fut]
            try:
                seed, e = fut.result()
                results_by_seed[seed] = e
            except Exception as exc:
                print(f"episode seed={seed}: FAILED -- {exc}")
                raise
            done += 1
            print(f"  [{done}/{len(seeds)} episodes complete]")

    # Preserve the requested seed ORDER in the output population, not
    # completion order (as_completed's own order is nondeterministic under
    # concurrency) -- keeps a re-run with the same --seeds reproducible in
    # its own recorded event order, even though wall-clock completion order
    # varies run to run.
    pop = [results_by_seed[seed] for seed in seeds]

    vi50 = validity_interval(pop, 0.5)
    lo, hi = bootstrap_vi(pop, 0.5, n_boot=400) if len(pop) > 1 else (float("nan"), float("nan"))
    prof = termination_profile(pop)
    cens = sum(ev.censored for ev in pop) / len(pop)
    grid, surv = kaplan_meier(pop)

    print(f"\nCosmos population (n={len(pop)}, real GPU-generated episodes, ILLUSTRATIVE not powered)")
    print(f"  VI_50 = {vi50}   95% CI [{lo}, {hi}]   censoring={cens:.2f}")
    print(f"  termination profile: {prof}")

    result = dict(
        scenario=args.scenario, target_property=spec.target_property, backend="cosmos",
        M=M, lam=LAM, t_max=t_max_episode,
        thresholds_median=dict(R2=float(np.median(thetas_episode["R2"][np.isfinite(thetas_episode["R2"])])),
                                R5=float(np.median(thetas_episode["R5"][np.isfinite(thetas_episode["R5"])]))),
        # No GATE 1a/1b for a real-model population -- those concepts (null
        # false-positive rate, planted-defect demonstration mutants) are
        # specific to the synthetic-mutant validation populations; a real
        # model has no "planted defect" to check sensitivity against. Left
        # present but empty/null so the existing report template (which
        # reads these fields) renders a blank section instead of erroring,
        # not because they're meaningfully "0" or "failed."
        gate_1a=dict(false_termination_rate=None, target_alpha=ALPHA, passed=None),
        gate_1b=[],
        survival=dict(
            n=len(pop), vi50=vi50, vi50_ci95=[lo, hi], censoring_rate=cens,
            termination_profile=prof,
            kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
            events=[dict(time=e.time, risk=e.risk, censored=e.censored) for e in pop],
        ),
    )
    out_path = ROOT / "results" / f"l0_demo_{args.scenario}_cosmos.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
