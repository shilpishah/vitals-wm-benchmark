"""Builds a REAL population of real-model episodes on any scenario,
scored through the EXACT SAME calibrated detector machinery
(sigma_existence/sigma_kinematic + estimate_threshold + extract_event)
already used for every other survival curve in this project. Writes
`results/l0_demo_<scenario>_<model>.json` in the SAME schema `run_l0_
demo.py` produces, so it drops straight into the existing visualizer.

Model-agnostic by construction (AGENT.md M6, 2026-08 -- generalized from
`run_cosmos_population.py` once a second real model, Wan2.1, needed the
identical population-building logic): properly reuses `VideoWorldModel`
(`vitals/adapters/video_model.py`) instead of re-implementing its own
render -> generate -> reconstruct -> slice sequence per model, which is
what the ORIGINAL Cosmos-only version of this script did (a real, now-
fixed duplication -- that version hand-rolled Cosmos's own video-file
calling convention inline rather than going through the `generate_fn`
abstraction that already existed for exactly this). Adding a THIRD model
later means adding one `--model` case below, not writing a new
population script.

Per episode (REAL GPU cost, two Modal calls regardless of which model --
treat this as expensive, not free): render the real prefix locally ->
`model.predict()` (which internally calls the model's own `generate_fn`,
then a `phi_fn` that runs the REAL Phi pipeline on the remote `vitals-phi`
app) -> stitch prefix (real physics) + reconstructed continuation via
`adapters.base.concat_trajectory` (the SAME function `CopyLastState`/
`ConstantVelocity` use) -> score with the SAME calibrated thresholds used
for every population on this scenario.

    python3 scripts/run_model_population.py --model cosmos --seeds 1,2,3,4,5
    python3 scripts/run_model_population.py --model wan --seeds 1,2,3,4,5

--n-video-episodes (2026-08, requested directly: "all of the videos should
be being generated possibly in one single video that shows all of the
rollouts to save space") -- for the FIRST N seeds (in --seeds order),
captures the frames `VideoWorldModel.predict` already generates while
scoring them (via its own `capture_frames` flag -- zero extra GPU cost,
this is not a second pass) and tiles them, annotated (green/red position
markers + prefix/continuation border), into one grid-mosaic video at
`results/videos/<scenario>_<model>_grid.mp4`. Fewer than N episodes in the
whole population (e.g. a 5-episode Wan run asked for a 9-tile grid) fills
however many tiles actually exist -- never pads by re-running extra
episodes just to fill a requested grid shape.

Every successful episode's full reconstructed Trajectory (prefix +
continuation) is ALSO cached to `results/trajectories/<scenario>_<model>/
seed<N>.npz` (2026-08, `vitals.adapters.video_utils.save_trajectory`) --
not used by this script itself, but what lets `scripts/run_lambda_sweep.py`
re-score this population against any number of lam-calibrated threshold
sets afterward for free, since lam never affects reconstruction, only the
reference ensemble/thresholds a trajectory gets scored against.
"""
import argparse
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from vitals.adapters import REAL_MODEL_BACKENDS

ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_generate_fn(model_name, scenario):
    """The ONLY genuinely model-specific piece -- everything else in this
    script is shared. Adding a model means TWO one-line additions, not
    one: its name in `vitals/adapters/__init__.py`'s own
    `REAL_MODEL_BACKENDS` (so `--model` accepts it AND
    `render_survival_report.py` categorizes its results correctly -- see
    that module's own docstring for the real bug this two-place
    requirement was found fixing), and a dispatch line here matching
    whatever `make_<model>_generate_fn_modal` that model's own adapter
    file exports (vitals/adapters/cosmos.py, wan.py, ...)."""
    if model_name == "cosmos":
        from vitals.adapters.cosmos import make_cosmos_generate_fn_modal
        return make_cosmos_generate_fn_modal(scenario)
    if model_name == "cosmos14b":
        # Same adapter, same deployed app -- only model_size differs
        # (vitals/adapters/__init__.py's own docstring: a separate
        # backend NAME, not a flag on "cosmos", so results never
        # silently overwrite the 2B population under the same file name).
        from vitals.adapters.cosmos import make_cosmos_generate_fn_modal
        return make_cosmos_generate_fn_modal(scenario, model_size="14B")
    if model_name == "cosmos720p":
        # Same adapter, same deployed app, same model_size ("2B") --
        # only resolution differs, isolating that ONE axis against the
        # same baseline the 14B test used (vitals/adapters/__init__.py's
        # own docstring).
        from vitals.adapters.cosmos import make_cosmos_generate_fn_modal
        return make_cosmos_generate_fn_modal(scenario, resolution="720")
    if model_name == "wan":
        from vitals.adapters.wan import make_wan_generate_fn_modal
        return make_wan_generate_fn_modal(scenario)
    raise ValueError(f"{model_name!r} is in REAL_MODEL_BACKENDS but has no dispatch case here -- "
                      f"add one to make_generate_fn() above (the two-place drift this docstring warns about).")


def make_modal_phi_fn(scenario):
    """phi_fn matching VideoWorldModel's own contract, calling the REAL
    Phi pipeline on the remote `vitals-phi` app (`run_phi_reconstruct_
    from_frames`) instead of the local-GPU default -- this dev machine has
    no GPU, same reason every other real call in this project goes
    through Modal."""
    def phi_fn(all_frames, prefix_len, frame0_mask, cam_pos, cam_mat, fovy_deg):
        import modal
        f = modal.Function.from_name("vitals-phi", "run_phi_reconstruct_from_frames")
        return f.remote(all_frames=all_frames, prefix_len=prefix_len, scenario_name=scenario,
                        frame0_mask=frame0_mask, cam_pos=cam_pos, cam_mat=cam_mat, fovy_deg=fovy_deg)
    return phi_fn


# Cosmos-2B's own observed per-episode latency comfortably clears 600s
# (both n=80 long-horizon re-runs this session finished with zero
# client-side timeouts at that value); Wan needs to clear its OWN
# deployed function's 1200s server-side ceiling (remote/modal_app_wan.py's
# own `image2video`) with margin, or a real bug recurs -- see
# resolve_episode_timeout_s's own docstring.
#
# cosmos14b (2026-08): the SAME deployed `video2world` function backs
# both sizes (remote/modal_app_cosmos.py), so it has the SAME 1200s
# server-side ceiling as 2B -- but 14B itself is NOT proportionally
# faster underneath. Real published benchmarks (Cosmos-Predict2's own
# `documentations/performance.md`, fetched directly, not guessed):
# 480p/16fps generation takes 79.87-87.32s for 2B vs. 286.46-377.67s for
# 14B on H100 (~3.6-4.7x slower) -- still comfortably under the 1200s
# server ceiling on pure compute alone, but with much less margin than
# 2B's own 600s default was ever tested against. Given Wan's own
# incident this session (a client timeout SHORTER than the server's own
# ceiling caused every episode to fail by abandoning-then-duplicating a
# still-running call), cosmos14b gets the same safety margin already
# proven for Wan -- comfortably past the shared 1200s server ceiling --
# rather than reusing 2B's untested-at-this-scale 600s.
#
# cosmos720p (2026-08): same reasoning applied to the OTHER un-isolated
# axis (resolution, not model size). Real published, non-NATTEN 720p/2B
# timing (this project's own deployed call never passes `--natten`):
# 228.8-378.5s on H100, depending on variant -- the SAME order of
# magnitude as 14B's own 480p latency above, so it gets the identical
# safety margin, not 2B/480p's own narrower-tested 600s.
EPISODE_TIMEOUT_S_DEFAULT = {"cosmos": 600, "cosmos14b": 1260, "cosmos720p": 1260, "wan": 1260}


def resolve_episode_timeout_s(model, explicit):
    """`explicit` is `args.episode_timeout_s` as parsed (None if the user
    didn't pass --episode-timeout-s). A flat single default across every
    model was a real bug, found directly (2026-08): 600s is comfortably
    above Cosmos's own observed latency but SHORTER than Wan's own
    deployed function's server-side timeout (1200s) -- so the client was
    ABANDONING (not cancelling) a Wan call that was still legitimately
    running server-side, then immediately firing an OVERLAPPING second
    attempt for the same episode, which is what actually caused 'Function
    call was cancelled' failures on every seed (NOT a genuine hang, which
    is what this timeout was built to catch). Pulled out as its own
    function so this policy is unit-testable without running main()."""
    if explicit is not None:
        return explicit
    return EPISODE_TIMEOUT_S_DEFAULT.get(model, 600)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(REAL_MODEL_BACKENDS))
    parser.add_argument("--scenario", default="occlusion_corridor")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--prefix-frames", type=int, default=30)
    parser.add_argument("--n-continuation-frames", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=8,
                        help="episodes run concurrently -- each is 2 real Modal GPU calls (network-bound, "
                             "safe to run in parallel via threads); kept modest by default (not e.g. "
                             "n_episodes) since this is a shared team Modal workspace, not to overwhelm it "
                             "with every episode's containers spinning up at once")
    parser.add_argument("--n-video-episodes", type=int, default=0,
                        help="capture+tile the first N seeds (in --seeds order) into one grid video -- "
                             "zero extra GPU cost, reuses frames these episodes generate anyway")
    parser.add_argument("--grid-video-path", default=None,
                        help="default: results/videos/<scenario>_<model>_grid.mp4")
    parser.add_argument("--episode-timeout-s", type=float, default=None,
                        help="hard wall-clock deadline per attempt (vitals.adapters.video_utils."
                             "call_with_timeout) -- protects against a call that hangs WITHOUT ever "
                             "raising (2026-08, found directly: a plain try/except retry alone let a "
                             "single hung episode block an entire n=80 population indefinitely, three "
                             "separate times on the same real re-run). Default is MODEL-SPECIFIC (see "
                             "EPISODE_TIMEOUT_S_DEFAULT below), not one constant for every model -- a real "
                             "bug, found directly (2026-08): a flat 600s default was comfortably above "
                             "Cosmos's own observed latency but SHORTER than Wan's own deployed function's "
                             "server-side timeout (1200s, remote/modal_app_wan.py's own `image2video`), so "
                             "the client was abandoning (not cancelling) a Wan call that was still "
                             "legitimately running server-side, then immediately firing an OVERLAPPING "
                             "second attempt for the same episode -- which is what actually caused "
                             "'Function call was cancelled' failures on every seed, not a hang. Pass this "
                             "flag explicitly to override either model's default; 0 disables the timeout "
                             "entirely.")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    video_seeds = set(seeds[:args.n_video_episodes])
    args.episode_timeout_s = resolve_episode_timeout_s(args.model, args.episode_timeout_s)

    import functools
    import yaml
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.detect.statistics import sigma_existence, sigma_kinematic, kinematic_axes_for
    from vitals.detect.thresholds import estimate_threshold
    from vitals.detect.events import extract_event
    from vitals.stats.survival import validity_interval, termination_profile, bootstrap_vi, kaplan_meier
    from vitals.adapters.base import prefix_of, concat_trajectory
    from vitals.adapters.video_model import VideoWorldModel
    from vitals.render.annotate import annotate_episode_frames, tile_grid_video
    from vitals.adapters.video_utils import _write_video, with_one_retry, save_trajectory

    manifest = yaml.safe_load((ROOT / f"configs/manifests/{args.scenario}.yaml").read_text())
    spec = EpisodeSpec(name=manifest["name"], scene=manifest["scene"], target_property=manifest["target_property"],
                        band=manifest["band"], lam=manifest["lam"], n_reference=manifest["n_reference"],
                        horizon_s=manifest["horizon_s"], fps=manifest["fps"], seed=manifest.get("seed", 0),
                        perturb_mode=manifest.get("perturb_mode", "full"))
    M, LAM, ALPHA = spec.n_reference, spec.lam, 0.01
    # R5's own axes restriction (AGENT.md M2.6) -- MUST match run_l0_demo.py/
    # run_gate2.py's own identical construction; see run_gate2.py's own
    # comment on STATS for why this has to be the SAME statistic across
    # all three scripts, not independently chosen per script.
    STATS = {"R2": sigma_existence,
             "R5": functools.partial(sigma_kinematic, axes=kinematic_axes_for(spec.name))}

    print(f"building the SAME M={M} reference ensemble + calibrated thresholds already used for "
          f"every other {args.scenario} population (cheap, local, state-space only)...")
    roll = make_backend("mujoco", scene=spec.scene)
    refs = [roll(spec, 1000 + i) for i in range(M)]
    dt = refs[0].dt
    thetas = {k: estimate_threshold(refs, fn, alpha=ALPHA) for k, fn in STATS.items()}

    # Every real-model episode here is shorter than the reference
    # ensemble's own native length -- theta must be sliced to match or
    # extract_event's comparison doesn't broadcast. Same fix scripts/
    # run_gate2.py established for its own truncated candidates, reused
    # (not rediscovered) here -- see the original Cosmos-only version of
    # this script for the full incident writeup (a real bug that cost one
    # episode's worth of GPU calls before being caught).
    n_episode_frames = args.prefix_frames + args.n_continuation_frames
    thetas_episode = {k: v[:n_episode_frames] for k, v in thetas.items()}
    t_max_episode = n_episode_frames * dt
    horizon_s = args.n_continuation_frames * dt

    cfg = sg.get(args.scenario)
    generate_fn = make_generate_fn(args.model, args.scenario)
    phi_fn = make_modal_phi_fn(args.scenario)

    # Every episode's full (prefix+continuation) reconstructed Trajectory is
    # cached to disk -- 2026-08, added specifically so the lambda sweep
    # (AGENT.md M7) can re-score a real model's ALREADY-PAID-FOR episodes at
    # any number of lam-calibrated threshold sets for free afterward,
    # instead of needing a fresh (real GPU cost) population per lam level.
    # `lam` never affects reconstruction, only the reference ensemble/
    # thresholds it gets scored against -- so this cache is reusable
    # forever, not just for the lam levels checked today. Cheap: each
    # episode is a few hundred KB at most (T~180 frames, K<=2 objects).
    traj_cache_dir = ROOT / "results" / "trajectories" / f"{args.scenario}_{args.model}"
    traj_cache_dir.mkdir(parents=True, exist_ok=True)

    def run_one_attempt(seed):
        # A fresh VideoWorldModel (and its own renderer) per worker -- not
        # shared across threads, same "don't rely on unverified thread-
        # safety for a multi-hour run" discipline as the renderer-per-
        # worker choice already made for the Cosmos-only version of this
        # script. generate_fn/phi_fn themselves are just closures making
        # network calls -- safe to share.
        model = VideoWorldModel(scenario_name=args.scenario, scene=spec.scene, generate_fn=generate_fn,
                                phi_fn=phi_fn, fps=spec.fps, height=240, width=320)
        rollout = make_backend("mujoco", scene=spec.scene)
        full = rollout(spec, seed=seed)
        conditioning = prefix_of(full, args.prefix_frames)
        capture = seed in video_seeds
        continuation_traj = model.predict(conditioning, horizon_s, n_samples=1, capture_frames=capture)[0]
        traj = concat_trajectory(conditioning, continuation_traj)
        save_trajectory(traj, traj_cache_dir / f"seed{seed}.npz")

        sigmas = {k: fn(traj, refs) for k, fn in STATS.items()}
        e = extract_event(sigmas, thetas_episode, dt, t_max_episode)

        annotated = None
        if capture:
            cap = model.last_capture
            true_pos = full.pos[:n_episode_frames, 0]
            annotated = annotate_episode_frames(cap["all_frames"], true_pos, cap["recon_pos"],
                                                 cap["cam_pos"], cap["cam_mat"], cap["fovy_deg"],
                                                 args.prefix_frames, border_thickness=8)
        return e, annotated

    def run_one(seed):
        # ONE retry on failure, PLUS a hard per-attempt timeout -- see
        # with_one_retry's/call_with_timeout's own docstrings for why both
        # are needed (found directly, three separate times on the same
        # real re-run: a plain try/except retry alone still blocks
        # forever on a call that hangs WITHOUT ever raising an exception).
        print(f"episode seed={seed}: starting...")
        timeout = args.episode_timeout_s if args.episode_timeout_s > 0 else None
        result, exc = with_one_retry(lambda: run_one_attempt(seed), label=f"episode seed={seed}",
                                     timeout_s=timeout)
        if result is None:
            print(f"episode seed={seed}: excluding from the population (n will be reported as "
                  f"smaller than --seeds requested, not silently padded)")
            return seed, None, None
        e, annotated = result
        print(f"episode seed={seed}: -> risk={e.risk} time={e.time if not e.censored else 'censored'}")
        return seed, e, annotated

    import concurrent.futures
    results_by_seed = {}
    video_by_seed = {}
    failed_seeds = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_one, seed): seed for seed in seeds}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            seed = futures[fut]
            try:
                seed, e, annotated = fut.result()
                if e is None:
                    failed_seeds.append(seed)
                else:
                    results_by_seed[seed] = e
                    if annotated is not None:
                        video_by_seed[seed] = annotated
            except Exception as exc:
                # Defensive only -- run_one() itself already catches and
                # retries every real failure mode internally (its own
                # sentinel `e is None` return above). Reaching here means
                # something OUTSIDE that (e.g. a bug in this collection
                # loop itself) -- still excluded, not fatal to the rest of
                # the batch, same "one bad seed doesn't destroy N-1 good,
                # already-paid-for episodes" principle as everywhere else
                # in this function.
                print(f"episode seed={seed}: FAILED in collection loop (unexpected) -- {exc}")
                failed_seeds.append(seed)
            done += 1
            print(f"  [{done}/{len(seeds)} episodes complete]")

    if failed_seeds:
        print(f"\n{len(failed_seeds)}/{len(seeds)} episode(s) permanently failed after retry, "
              f"excluded from this population: seeds {sorted(failed_seeds)}")
    pop = [results_by_seed[seed] for seed in seeds if seed in results_by_seed]
    if not pop:
        print("\nEVERY episode failed -- nothing to score. Not writing a results file "
              "(an empty/fabricated population would be worse than no file at all).")
        return

    if video_by_seed:
        # seeds order (not dict/completion order) -- deterministic tile
        # placement regardless of which episode's Modal call happened to
        # finish first.
        ordered_frames = [video_by_seed[s] for s in seeds if s in video_by_seed]
        grid = tile_grid_video(ordered_frames)
        grid_path = pathlib.Path(args.grid_video_path) if args.grid_video_path else (
            ROOT / "results" / "videos" / f"{args.scenario}_{args.model}_grid.mp4")
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        _write_video(grid, grid_path, fps=spec.fps)
        print(f"\ngrid video ({len(ordered_frames)} episodes, seeds "
              f"{[s for s in seeds if s in video_by_seed]}) -> {grid_path}")

    vi50 = validity_interval(pop, 0.5)
    lo, hi = bootstrap_vi(pop, 0.5, n_boot=400) if len(pop) > 1 else (float("nan"), float("nan"))
    prof = termination_profile(pop)
    cens = sum(ev.censored for ev in pop) / len(pop)
    grid, surv = kaplan_meier(pop)

    print(f"\n{args.model} population (n={len(pop)}, real GPU-generated episodes"
          f"{f', {len(failed_seeds)} requested seed(s) excluded after failing twice' if failed_seeds else ''})")
    print(f"  VI_50 = {vi50}   95% CI [{lo}, {hi}]   censoring={cens:.2f}")
    print(f"  termination profile: {prof}")

    result = dict(
        scenario=args.scenario, target_property=spec.target_property, backend=args.model,
        M=M, lam=LAM, t_max=t_max_episode,
        thresholds_median=dict(R2=float(np.median(thetas_episode["R2"][np.isfinite(thetas_episode["R2"])])),
                                R5=float(np.median(thetas_episode["R5"][np.isfinite(thetas_episode["R5"])]))),
        # No GATE 1a/1b for a real-model population -- see AGENT.md's own
        # writeup on this (Truth vs. Baseline/Cosmos distinction) for why.
        gate_1a=dict(false_termination_rate=None, target_alpha=ALPHA, passed=None),
        gate_1b=[],
        # n_requested/failed_seeds (2026-08): a real-model population's own
        # n can now be SMALLER than --seeds asked for (one seed failing
        # twice no longer destroys every other seed's already-real-GPU-
        # cost result -- see run_one's own docstring) -- recorded
        # explicitly so n=X in this file is never silently mistaken for
        # "X requested, X succeeded" when they differ.
        n_requested=len(seeds), failed_seeds=sorted(failed_seeds),
        survival=dict(
            n=len(pop), vi50=vi50, vi50_ci95=[lo, hi], censoring_rate=cens,
            termination_profile=prof,
            kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
            events=[dict(time=e.time, risk=e.risk, censored=e.censored) for e in pop],
        ),
    )
    out_path = ROOT / "results" / f"l0_demo_{args.scenario}_{args.model}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
