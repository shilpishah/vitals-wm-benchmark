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

from vitals.adapters import REAL_MODEL_BACKENDS, MODEL_REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_generate_fn(model_name, scenario):
    """The ONLY genuinely model-specific piece -- everything else in this
    script is shared. Adding a model means TWO one-line additions, not
    one: an entry in `vitals/adapters/__init__.py`'s own `MODEL_REGISTRY`
    (so `--model` accepts it AND `render_survival_report.py` categorizes
    its results correctly -- see that module's own docstring for the real
    bug this two-place requirement was found fixing; mark it `restricted`
    there too if its license needs an explicit opt-in), and a dispatch
    line here matching whatever `make_<model>_generate_fn_modal` that
    model's own adapter file exports (vitals/adapters/cosmos.py, wan.py,
    ...)."""
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
    if model_name == "hunyuan":
        from vitals.adapters.hunyuan import make_hunyuan_generate_fn_modal
        return make_hunyuan_generate_fn_modal(scenario)
    if model_name == "cosmos3nano":
        from vitals.adapters.cosmos3 import make_cosmos3_generate_fn_modal
        return make_cosmos3_generate_fn_modal(scenario)
    if model_name == "cosmos3super":   # same adapter/contract, different app (TP=4 on 4x H100)
        from vitals.adapters.cosmos3 import make_cosmos3_generate_fn_modal
        return make_cosmos3_generate_fn_modal(scenario, app_name="vitals-cosmos3super")
    if model_name == "cogvideox15":
        from vitals.adapters.cogvideox import make_cogvideox_generate_fn_modal
        return make_cogvideox_generate_fn_modal(scenario)
    if model_name == "wan22":
        from vitals.adapters.wan22 import make_wan22_generate_fn_modal
        return make_wan22_generate_fn_modal(scenario)
    if model_name == "ltx23":
        from vitals.adapters.ltx import make_ltx_generate_fn_modal
        return make_ltx_generate_fn_modal(scenario)
    # API-access models (2026-09): one adapter parameterized by the gateway
    # model id (vitals/adapters/runway.py's own docstring) -- the id comes
    # from the registry entry, so adding another Runway-gateway model is
    # a registry line only; this branch already covers it.
    if MODEL_REGISTRY.get(model_name, {}).get("provider") == "runway":
        from vitals.adapters.runway import make_runway_generate_fn
        meta = MODEL_REGISTRY[model_name]
        return make_runway_generate_fn(scenario, model=meta["model_id"], ratio=meta.get("ratio", "1104:832"),
                                       allowed_durations=meta.get("durations"),
                                       supports_seed=meta.get("supports_seed", True))
    raise ValueError(f"{model_name!r} is in REAL_MODEL_BACKENDS but has no dispatch case here -- "
                      f"add one to make_generate_fn() above (the two-place drift this docstring warns about).")


def make_modal_phi_fn(scenario):
    """phi_fn matching VideoWorldModel's own contract, calling the REAL
    Phi pipeline on the remote `vitals-phi` app (`run_phi_reconstruct_
    from_frames`) instead of the local-GPU default -- this dev machine has
    no GPU, same reason every other real call in this project goes
    through Modal.

    frame0_mask_secondary/frame0_mask_tertiary: OPTIONAL keywords,
    default None -- accepted here (2026-09, R2/sigma_interpenetration and
    then billiards/multi-collision wired into real-model scoring) purely
    to match VideoWorldModel.predict()'s own conditional call (only ever
    passed for target_property == "P3" or the explicit track_all_objects
    opt-in); passed straight through to the remote function. None for
    every other scenario, a strict no-op identical to before either
    parameter existed."""
    def phi_fn(all_frames, prefix_len, frame0_mask, cam_pos, cam_mat, fovy_deg,
               frame0_mask_secondary=None, frame0_mask_tertiary=None):
        import modal
        f = modal.Function.from_name("vitals-phi", "run_phi_reconstruct_from_frames")
        return f.remote(all_frames=all_frames, prefix_len=prefix_len, scenario_name=scenario,
                        frame0_mask=frame0_mask, cam_pos=cam_pos, cam_mat=cam_mat, fovy_deg=fovy_deg,
                        frame0_mask_secondary=frame0_mask_secondary,
                        frame0_mask_tertiary=frame0_mask_tertiary)
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
# wan (2026-09): a DEEPER problem than the client/server-race incident
# above -- a real, ISOLATED (zero concurrency, zero contention) call was
# measured directly (not assumed) to take 27:46 (1666s) wall-clock,
# exceeding even the OLD 1200s SERVER ceiling itself, not just racing a
# client timeout against a server call that would have succeeded. Found
# by reproducing a real n=50 population's own hang directly: Modal's own
# container logs showed genuine `FunctionTimeoutError`s at the 1200s
# mark, and a single call run completely alone hit the identical wall.
# `remote/modal_app_wan.py`'s own `image2video` timeout raised 1200 ->
# 2400 (real margin above the measured 1666s, not a guess) -- this
# client default raised to stay above THAT, same "client >= server"
# invariant the original incident fix already established.
# runway_* (2026-09): hosted API, no container cold start -- the SDK's own
# wait_for_task_output default is 600s and the adapter passes that through;
# 900s here keeps the "client >= server" invariant with margin. NOT a
# measured number yet (no real call has been made from this project) --
# revisit with real task timings, same discipline as wan's own entry.
# cosmos3nano (2026-09): UNMEASURED placeholder -- 16B on H100 at 480p for
# ~49 frames plus a cold ~32GB load; server timeout is 1500s, this stays
# above it. Replace with a real timed call, same discipline as wan.
# Fixed horizons at which S(t) is reported alongside VI50 (2026-09-12) --
# the same three on every population so pages are comparable; horizons past
# a population's own t_max are simply S(t_max) (censored plateau).
HORIZONS_S = [1.5, 2.0, 3.0]

EPISODE_TIMEOUT_S_DEFAULT = {
    # truth_render (2026-09-12): no generative model, a local render + one Phi call.
    # (An earlier edit put this entry's comment on the brace line and swallowed
    # every other default into it -- caught by tests/test_run_model_population.py.)
    "truth_render": 300,
    "cosmos": 600, "cosmos14b": 1260, "cosmos720p": 1260, "wan": 2460,
                             "hunyuan": 1260, "runway_gen4.5": 900, "runway_veo3.1": 1200,
                             "runway_seedance2.5": 1200, "runway_gemini_omni": 1200, "runway_h3_max": 1200,
                             "cosmos3nano": 1560,
                             "cosmos3super": 3660,   # UNMEASURED; server 3600s (TP=4 on 4x H100); placeholder until timed
                             # cogvideox15 (2026-09): first real call exceeded 1800s on A10G with
                             # the card's sequential-offload recipe (measured, not assumed); the
                             # app now uses model-level offload and a 3600s server timeout -- this
                             # stays above it. Still not a measured completion time.
                             "cogvideox15": 3660,
                             # wan22 (2026-09): MEASURED 14:48 (888s) on the first isolated call on
                             # A100-80GB -- faster than wan2.1's 27:46, well inside the 2400s server
                             # timeout; client kept above server. Not tightened: cold starts and
                             # 8-way population load add variance a single warm-ish call doesn't show.
                             "wan22": 2460,
                             # ltx23 (2026-09): UNMEASURED; server 1500s.
                             "ltx23": 1560}


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
    # "truth_render" (2026-09-12): the INSTRUMENT IDEAL -- not a model. Each
    # episode's continuation is the TRUE rollout rendered by the same
    # renderer/camera, passed through the same mp4 codec a model's output
    # goes through, then reconstructed by Phi and scored on every channel
    # (R5 included). Its survival is what a perfect video model would score
    # after the perception tax -- the honest ceiling for a real model on
    # this scenario (GATE 2's null instances, at population scale). Never in
    # MODEL_REGISTRY: it must not get a report page of its own; the report
    # draws it as the "ideal (instrument)" series on every model's page.
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY) + ["truth_render"],
                        help="a name registered but not in REAL_MODEL_BACKENDS is license-restricted "
                             "and not currently enabled -- see the error this raises for how to opt in")
    parser.add_argument("--scenario", default="occlusion_corridor")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--prefix-frames", type=int, default=30)
    parser.add_argument("--n-continuation-frames", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=8,
                        help="episodes run concurrently -- each is 2 real Modal GPU calls (network-bound, "
                             "safe to run in parallel via threads); kept modest by default (not e.g. "
                             "n_episodes) since this is a shared team Modal workspace, not to overwhelm it "
                             "with every episode's containers spinning up at once")
    parser.add_argument("--merge-existing", action="store_true",
                        help="merge this run's seeds INTO the existing results file for the same scenario/model "
                             "instead of overwriting it (2026-09-12: needed for top-ups and for API models whose "
                             "daily generation cap is below a full population). Seeds already present are kept, "
                             "not re-run; thresholds must match the existing file's.")
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
    if args.model != "truth_render" and args.model not in REAL_MODEL_BACKENDS:   # truth_render: not a model, no license
        meta = MODEL_REGISTRY.get(args.model, {})
        raise SystemExit(
            f"--model {args.model!r} is license-restricted ({meta['license']}) and not currently "
            f"enabled: {meta['restriction_note']}\n"
            f"Set VITALS_ENABLE_RESTRICTED_MODELS={args.model} to opt in for this run.")
    # Blank entries ignored: a trailing comma (macOS `seq -s,` emits one)
    # killed a real orchestrator submission at parse time (2026-09-11).
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    video_seeds = set(seeds[:args.n_video_episodes])
    args.episode_timeout_s = resolve_episode_timeout_s(args.model, args.episode_timeout_s)

    import functools
    import yaml
    from vitals.types import EpisodeSpec, Trajectory
    from vitals.physics import make_backend
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.detect.statistics import sigma_existence, sigma_kinematic, sigma_interpenetration, kinematic_axes_for
    from vitals.detect.thresholds import estimate_threshold
    from vitals.detect.events import extract_event
    from vitals.detect.long_horizon import make_long_horizon_sigma
    from vitals.stats.survival import (validity_interval, termination_profile, bootstrap_vi, kaplan_meier,
                                       restricted_mean_validity, bootstrap_rmvt, survival_at)
    from vitals.phi.frame_consistency import make_frame_invariance_channel, R5_MIN_PX
    from vitals.adapters.base import prefix_of, concat_trajectory
    from vitals.adapters.video_model import VideoWorldModel
    from vitals.render.annotate import annotate_episode_frames, tile_grid_video
    from vitals.adapters.video_utils import _write_video, with_one_retry, save_trajectory

    manifest = yaml.safe_load((ROOT / f"configs/manifests/{args.scenario}.yaml").read_text())
    spec = EpisodeSpec(name=manifest["name"], scene=manifest["scene"], target_property=manifest["target_property"],
                        band=manifest["band"], lam=manifest["lam"], n_reference=manifest["n_reference"],
                        horizon_s=manifest["horizon_s"], fps=manifest["fps"], seed=manifest.get("seed", 0),
                        perturb_mode=manifest.get("perturb_mode", "full"),
                        track_all_objects=manifest.get("track_all_objects", False))
    M, LAM, ALPHA = spec.n_reference, spec.lam, 0.01
    # R3's own axes restriction (AGENT.md M2.6) -- MUST match run_l0_demo.py/
    # run_gate2.py's own identical construction; see run_gate2.py's own
    # comment on STATS for why this has to be the SAME statistic across
    # all three scripts, not independently chosen per script.
    #
    # R1's own obj=0 restriction -- for the SAME reason run_gate2.py
    # already binds it (see sigma_existence's own docstring): this
    # script's candidate is always K=1 (Phi's single-object-only real-
    # video reconstruction, sliced to match below), but a K>1 scenario's
    # own reference ensemble (collision, K=2: cue + target) is whole-
    # scene K=2 -- unrestricted whole-count sigma_existence would compare
    # candidate_count=1 against reference_mean~2 at every frame regardless
    # of tracking quality, firing R1 100% of the time on perfect tracking.
    # Found directly (2026-08, first K>1 scenario run through this
    # script, before it could silently corrupt a real population) rather
    # than waiting to see it in the numbers. Strict no-op for every
    # existing K=1 scenario.
    STATS = {"R1": functools.partial(sigma_existence, obj=0),
             "R3": functools.partial(sigma_kinematic, axes=kinematic_axes_for(spec.name))}

    print(f"building the SAME M={M} reference ensemble + calibrated thresholds already used for "
          f"every other {args.scenario} population (cheap, local, state-space only)...")
    roll = make_backend("mujoco", scene=spec.scene)
    refs = [roll(spec, 1000 + i) for i in range(M)]
    dt = refs[0].dt

    # R2 (sigma_interpenetration), wired into real-model (L1) scoring for
    # the first time (2026-09) -- SAME gating rule as run_l0_demo.py's own
    # STATS (target_property == "P3", NOT K>=2; see that script's own
    # comment for the real bug found gating on K alone instead). A strict
    # no-op for every manifest not pre-registered P3, collision (P4, K=2)
    # included. VideoWorldModel itself only threads a second object's
    # frame-0 mask through phi_fn under this SAME condition (its own
    # `target_property` docstring) -- STATS and the adapter's own
    # reconstruction behavior must agree, or R2 would be scored against a
    # candidate that was never given a second object to track at all.
    # Soft-body scenarios (AGENT.md M9/T10, 2026-09-14): pixel-measured
    # candidates get R6 conservation / R7 shape scored against the
    # Phi-MEASURED reference band (results/phi_refs_<scenario>.npz,
    # scripts/build_phi_references.py -- the same tracker on both sides;
    # SAM2's mask of a resting soft body is ~13% larger than the rendered
    # silhouette, a systematic error the state band cannot absorb).
    # R1/R3/R4 keep the state-space band as everywhere. A soft scenario
    # without its Phi band is refused: scoring R6/R7 against state
    # references fired on every GATE 2 null.
    from vitals.physics import softbody as sb
    SOFT = bool(sg.SCENES.get(spec.name, {}).get("soft_body", False))
    phi_refs, STATS_PHI = None, {}
    if SOFT:
        from vitals.detect.statistics import sigma_conservation, sigma_shape
        phi_refs = sb.load_phi_refs(sb.phi_refs_path(ROOT, spec.name))
        assert phi_refs, (f"{spec.name!r} is a soft-body scenario but results/phi_refs_{spec.name}.npz is missing -- "
                          f"run scripts/build_phi_references.py first (AGENT.md T10).")
        STATS_PHI = {"R6": sigma_conservation, "R7": sigma_shape}
    if spec.target_property == "P3" and not SOFT:
        assert refs[0].K >= 2, (
            f"{spec.name!r} is registered target_property: P3 but its scene has K={refs[0].K} "
            f"objects -- sigma_interpenetration needs obj=0/target=1 both real. Fix the manifest "
            f"or the scene, not this check.")
        STATS = dict(STATS, R2=sigma_interpenetration)

    thetas = {k: estimate_threshold(refs, fn, alpha=ALPHA) for k, fn in STATS.items()}
    for k, fn in STATS_PHI.items():
        thetas[k] = estimate_threshold(phi_refs, fn, alpha=ALPHA)
    if SOFT:
        print(f"soft-body channels: Phi-measured band of {len(phi_refs)} refs; "
              + "  ".join(f"theta_{k} median={float(np.nanmedian(thetas[k])):.2f}" for k in STATS_PHI))

    # R4 (long-horizon/cumulative consistency, 2026-09) -- built on
    # whichever base channels are ACTUALLY active in STATS above (so it
    # automatically tracks R2's own P3-only gating with no separate
    # condition here), calibrated the SAME LOO/estimate_threshold way,
    # but scored and reported as an INDEPENDENT event below, never merged
    # into `STATS`/`thetas` -- see AGENT.md's own writeup + events.py's
    # own module docstring for why: extract_event returns exactly one
    # winning Event, and merging R4 in would let an early R1/R2/R3 hit
    # silently suppress a real long-horizon finding.
    sigma_R4 = make_long_horizon_sigma(STATS, refs, thetas)
    theta_R4 = estimate_threshold(refs, sigma_R4, alpha=ALPHA)

    # Every real-model episode here is shorter than the reference
    # ensemble's own native length -- theta must be sliced to match or
    # extract_event's comparison doesn't broadcast. Same fix scripts/
    # run_gate2.py established for its own truncated candidates, reused
    # (not rediscovered) here -- see the original Cosmos-only version of
    # this script for the full incident writeup (a real bug that cost one
    # episode's worth of GPU calls before being caught).
    n_episode_frames = args.prefix_frames + args.n_continuation_frames
    thetas_episode = {k: v[:n_episode_frames] for k, v in thetas.items()}
    theta_R4_episode = theta_R4[:n_episode_frames]
    t_max_episode = n_episode_frames * dt
    horizon_s = args.n_continuation_frames * dt

    cfg = sg.get(args.scenario)

    # R5 (frame invariance, WIRED 2026-09-12) -- the pixel-scoped channel,
    # scored for real video models only (state-space baselines have no
    # pixels). Background-change statistic anchored at the last real
    # prefix frame, null-calibrated on the SAME reference ensemble
    # rendered + codec-roundtripped, thresholded by the same LOO
    # machinery; see frame_consistency.py's own "R5 as WIRED" section for
    # why it is background change rather than patch drift. Enters the
    # precedence race FIRST (events.PRECEDENCE): a re-framed scene is
    # diagnosed as R5, not as the apparent object motion Phi would
    # otherwise read into it -- the exact mislabel the Cosmos 3
    # occlusion_corridor population showed (AGENT.md, that entry).
    print(f"R5 calibration: rendering the M={M} reference ensemble ({n_episode_frames} frames each) ...")
    theta_R5, sigma_R5, diag_R5 = make_frame_invariance_channel(
        refs, spec.scene, cfg["camera"], cfg["object_radius"], n_episode_frames, args.prefix_frames, spec.fps,
        height=240, width=320, alpha=ALPHA)
    thetas_episode["R5"] = theta_R5[:n_episode_frames]
    print(f"R5 calibration: theta_R5 median={float(np.nanmedian(theta_R5[args.prefix_frames:])):.2f} px "
          f"(floor {R5_MIN_PX} px)")
    r5_diag_by_seed = {}
    generate_fn = None if args.model == "truth_render" else make_generate_fn(args.model, args.scenario)
    phi_fn = make_modal_phi_fn(args.scenario)

    def make_truth_render_fn(full_traj):
        """Per-episode 'generator' for --model truth_render: the true
        continuation frames, rendered and codec-roundtripped. The prefix
        frames VideoWorldModel renders itself are byte-identical to ours
        (same renderer, camera, size), so the stitched clip is one
        continuous real render -- plus the codec noise a model's decoded
        output would carry."""
        from vitals.phi.frame_consistency import codec_roundtrip
        renderer = MujocoRenderer(spec.scene, height=240, width=320)
        fr, _ = renderer.render(full_traj, cameras=cfg["camera"])
        frames = codec_roundtrip(fr.rgb, spec.fps)

        def generate(prefix_frames, n_frames):
            p = len(prefix_frames)
            out = frames[p:p + n_frames]
            if out.shape[0] < n_frames:   # pad by holding the last frame (rollout shorter than asked)
                out = np.concatenate([out, np.repeat(out[-1:], n_frames - out.shape[0], axis=0)])
            return out
        return generate

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
        rollout = make_backend("mujoco", scene=spec.scene)
        full = rollout(spec, seed=seed)
        gen = make_truth_render_fn(full) if args.model == "truth_render" else generate_fn
        model = VideoWorldModel(scenario_name=args.scenario, scene=spec.scene, generate_fn=gen,
                                phi_fn=phi_fn, fps=spec.fps, height=240, width=320,
                                target_property=spec.target_property, track_all_objects=spec.track_all_objects)
        conditioning = prefix_of(full, args.prefix_frames)
        # capture_frames=True for EVERY episode now (2026-09-12): R5 needs
        # the generated pixels, which existed in-process all along (Phi
        # runs on them) and were discarded after reconstruction. ~21MB per
        # episode, released when this function returns.
        capture = True
        continuation_traj = model.predict(conditioning, horizon_s, n_samples=1, capture_frames=capture)[0]
        # Phi reconstruction only ever returns the single scored object
        # (obj=0 convention, same restriction GATE 2's own STATS already
        # applies) -- for a K>1 scene (collision, K=2: cue + target),
        # `conditioning` still carries every object MuJoCo simulated
        # (needed by model.predict()'s own render step, which must show
        # the full scene), so it has to be sliced down to match before
        # concatenation. Found directly (2026-08, first K>1 scenario run
        # through this script): concat_trajectory's np.concatenate
        # rejected K=2 conditioning against K=1 continuation. A no-op for
        # every existing K=1 scenario.
        if conditioning.pos.shape[1] != continuation_traj.pos.shape[1]:
            K = continuation_traj.pos.shape[1]
            conditioning = Trajectory(conditioning.t.copy(), conditioning.pos[:, :K].copy(),
                                       conditioning.quat[:, :K].copy(), conditioning.present[:, :K].copy(),
                                       list(conditioning.names[:K]), dict(conditioning.meta))
        traj = concat_trajectory(conditioning, continuation_traj)
        cap = model.last_capture
        if SOFT and cap.get("recon_shape") is not None:
            # Phi's shape over the whole clip (prefix included): R6's ratio
            # is to the clip's own frame 0, exactly how the Phi band was built.
            traj.shape = cap["recon_shape"][:traj.T]
        save_trajectory(traj, traj_cache_dir / f"seed{seed}.npz")

        sigmas = {k: fn(traj, refs) for k, fn in STATS.items()}
        for k, fn in STATS_PHI.items():
            sigmas[k] = fn(traj, phi_refs)
        r5_args = (cap["all_frames"], cap["recon_pos"], cap["recon_present"], cap["cam_pos"], cap["cam_mat"], cap["fovy_deg"])
        sigmas["R5"] = sigma_R5(*r5_args)[:n_episode_frames]
        r5_diag_by_seed[seed] = diag_R5(*r5_args)
        e = extract_event(sigmas, thetas_episode, dt, t_max_episode)
        e_r4 = extract_event({"R4": sigma_R4(traj, refs)}, {"R4": theta_R4_episode}, dt, t_max_episode)

        annotated = None
        if seed in video_seeds:
            # ALL objects (T,K,3) -- was object 0 only; see video_model.py's
            # own last_capture comment for why this now matches recon_pos.
            # Sliced to cap["recon_pos"]'s own K (the RECONSTRUCTED count,
            # not the scene's full simulated count -- e.g. billiards' own
            # K=3 scene reconstructs only K=1 unless track_all_objects
            # opted in): true_pos must never carry MORE objects than
            # recon_pos has markers for, same K-mismatch principle the
            # conditioning/continuation_traj slicing above already
            # enforces, just for annotation instead of scoring.
            n_annotated_objects = cap["recon_pos"].shape[1]
            true_pos = full.pos[:n_episode_frames, :n_annotated_objects]
            annotated = annotate_episode_frames(cap["all_frames"], true_pos, cap["recon_pos"],
                                                 cap["cam_pos"], cap["cam_mat"], cap["fovy_deg"],
                                                 args.prefix_frames, border_thickness=8)
        return e, e_r4, annotated

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
            return seed, None, None, None
        e, e_r4, annotated = result
        r3_note = f" R4={'censored' if e_r4.censored else f'{e_r4.time:.2f}s'}"
        print(f"episode seed={seed}: -> risk={e.risk} time={e.time if not e.censored else 'censored'}{r3_note}")
        return seed, e, e_r4, annotated

    import concurrent.futures
    results_by_seed = {}
    results_r4_by_seed = {}
    video_by_seed = {}
    failed_seeds = []
    # Runway's tier is read from organization.retrieve(): on 2026-09-11 it
    # allowed 1 concurrent generation and 50/day per model (so the pool was
    # capped at 1 and populations paced one scenario per day); re-read
    # 2026-09-14 it allows 5 concurrent and 1000/day for every video
    # model. The pool is capped at the tier's concurrency for API models
    # regardless of --max-workers; the daily cap is NOT enforced here (a
    # population that trips it fails visibly per seed).
    max_workers = args.max_workers
    # Runway tier re-read 2026-09-14 (organization.retrieve()): every
    # video model now allows 5 concurrent generations and 1000/day (was
    # 1 and 50 on 2026-09-11), so the cap is 5, not 1.
    RUNWAY_MAX_CONCURRENT = 5
    if MODEL_REGISTRY.get(args.model, {}).get("provider") == "runway" and max_workers > RUNWAY_MAX_CONCURRENT:
        print(f"{args.model}: provider 'runway' allows {RUNWAY_MAX_CONCURRENT} concurrent generations -- "
              f"max_workers {max_workers} -> {RUNWAY_MAX_CONCURRENT}")
        max_workers = RUNWAY_MAX_CONCURRENT
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, seed): seed for seed in seeds}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            seed = futures[fut]
            try:
                seed, e, e_r4, annotated = fut.result()
                if e is None:
                    failed_seeds.append(seed)
                else:
                    results_by_seed[seed] = e
                    results_r4_by_seed[seed] = e_r4
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
    # --merge-existing (2026-09-12): fold the previous file's per-seed
    # events into this run's before any statistic is computed. Files
    # written before this flag existed carry no per-event seed; they are
    # only merged when their seed list is unambiguous (no failed seeds and
    # n == n_requested, i.e. seeds 1..n in order), otherwise refused.
    out_path = ROOT / "results" / f"l0_demo_{args.scenario}_{args.model}.json"
    all_seeds = list(seeds)
    if args.merge_existing and out_path.exists():
        from vitals.types import Event as _Event
        old = json.loads(out_path.read_text())
        old_thr = old.get("thresholds_median", {})
        new_thr = {k: float(np.median(v[np.isfinite(v)])) for k, v in thetas_episode.items() if np.any(np.isfinite(v))}
        for k in set(old_thr) & set(new_thr):
            if abs(old_thr[k] - new_thr[k]) > 0.05 * max(abs(old_thr[k]), 1e-9):
                raise SystemExit(f"--merge-existing refused: threshold {k} differs ({old_thr[k]:.4g} vs {new_thr[k]:.4g}); "
                                 f"the two populations were not scored under the same calibration")
        old_ev = old["survival"]["events"]
        if all("seed" in e for e in old_ev):
            old_seed_of = [e["seed"] for e in old_ev]
        elif not old.get("failed_seeds") and old.get("n_requested") == len(old_ev):
            old_seed_of = list(range(1, len(old_ev) + 1))
        else:
            raise SystemExit("--merge-existing refused: the existing file predates per-seed events and its seed list is ambiguous")
        old_r4 = old.get("long_horizon", {}).get("events", [])
        old_fi = old.get("frame_invariance", {}).get("per_seed", {})
        kept = 0
        for s, e, e4 in zip(old_seed_of, old_ev, old_r4 if len(old_r4) == len(old_ev) else [None] * len(old_ev)):
            if s in results_by_seed:
                continue
            results_by_seed[s] = _Event(time=e["time"], risk=e["risk"], censored=e["censored"])
            if e4 is not None:
                results_r4_by_seed[s] = _Event(time=e4["time"], risk="R4" if not e4["censored"] else None, censored=e4["censored"])
            if str(s) in old_fi:
                r5_diag_by_seed[s] = old_fi[str(s)]
            kept += 1
        failed_seeds = [s for s in failed_seeds if s not in results_by_seed] + [s for s in old.get("failed_seeds", []) if s not in results_by_seed and s not in seeds]
        all_seeds = sorted(set(seeds) | set(old_seed_of))
        print(f"\n--merge-existing: kept {kept} episode(s) from the existing file; population is now {len(results_by_seed)} seeds")
    pop = [results_by_seed[seed] for seed in all_seeds if seed in results_by_seed]
    pop_seeds = [seed for seed in all_seeds if seed in results_by_seed]
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

    # R4 -- same aggregation machinery (validity_interval/termination_
    # profile/kaplan_meier all just operate on a list of Event objects,
    # which is exactly what R4 produces too), computed over its OWN
    # independent pop_r4, reported as its own parallel `long_horizon`
    # block below -- never merged into `survival` above.
    pop_r4 = [results_r4_by_seed[seed] for seed in all_seeds if seed in results_r4_by_seed]
    pop_r4_seeds = [seed for seed in all_seeds if seed in results_r4_by_seed]
    vi50_r4 = validity_interval(pop_r4, 0.5)
    lo_r4, hi_r4 = bootstrap_vi(pop_r4, 0.5, n_boot=400) if len(pop_r4) > 1 else (float("nan"), float("nan"))
    cens_r4 = sum(ev.censored for ev in pop_r4) / len(pop_r4)
    grid_r4, surv_r4 = kaplan_meier(pop_r4)

    print(f"\n{args.model} population (n={len(pop)}, real GPU-generated episodes"
          f"{f', {len(failed_seeds)} requested seed(s) excluded after failing twice' if failed_seeds else ''})")
    print(f"  VI_50 = {vi50}   95% CI [{lo}, {hi}]   censoring={cens:.2f}")
    print(f"  termination profile: {prof}")
    print(f"  R4 (long-horizon): fired on {1 - cens_r4:.0%} of episodes, VI_50 = {vi50_r4}")

    result = dict(
        scenario=args.scenario, target_property=spec.target_property, backend=args.model,
        M=M, lam=LAM, t_max=t_max_episode,
        # Derived from whatever's actually in STATS/thetas_episode, not a
        # hardcoded {R1, R3} -- FOUND AS A REAL BUG (2026-09), wiring R2
        # into this script: STATS/thetas correctly included R2 for a P3
        # manifest, but this dict silently dropped it from the SAVED json
        # anyway (the hardcoded key list here never got the memo), so
        # `thresholds_median` under-reported what was actually scored --
        # not a "R2 never fires" finding, an "R2 was invisible in the
        # output" bug. np.isfinite guards the same NaN-median case R1/R3
        # already guarded (R2 is NaN whenever either object is absent).
        thresholds_median={k: float(np.median(v[np.isfinite(v)])) for k, v in thetas_episode.items()},
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
        n_requested=len(all_seeds), failed_seeds=sorted(failed_seeds), seeds=pop_seeds,
        survival=dict(
            n=len(pop), vi50=vi50, vi50_ci95=[lo, hi], censoring_rate=cens,
            termination_profile=prof,
            # RMVT + S(t) at fixed horizons (2026-09-12, the VI50 floor):
            # rankable summaries of the whole curve; VI50 stays primary.
            rmvt=restricted_mean_validity(pop, t_max_episode),
            rmvt_ci95=list(bootstrap_rmvt(pop, t_max_episode, n_boot=400)) if len(pop) > 1 else [float("nan")] * 2,
            S_at={str(h): s for h, s in zip(HORIZONS_S, survival_at(pop, HORIZONS_S))},
            kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
            events=[dict(seed=s, time=e.time, risk=e.risk, censored=e.censored) for s, e in zip(pop_seeds, pop)],
        ),
        # R4 (long-horizon/cumulative consistency) -- an INDEPENDENT
        # diagnostic axis, deliberately not part of `survival` above (see
        # events.py's own module docstring for why: it doesn't compete in
        # the same first-sustained-crossing precedence race, so it can't
        # share that block's own single-winner-per-episode shape without
        # either losing information or misrepresenting what "censored"
        # means there). theta_R4_median guards the same NaN case
        # thresholds_median already does.
        # R5 photometric DIAGNOSTICS (2026-09-12): not a channel -- the
        # real->generated boundary jump (how unlike the render the model's
        # first frame is) and the model's own background drift after it,
        # in intensity levels. Reported so a pan/zoom (R5 proper) can be
        # told apart from a recolor the geometric channel cannot see.
        frame_invariance=dict(
            theta_R5_px_median=float(np.nanmedian(theta_R5[args.prefix_frames:])),
            photometric_boundary_jump_median=float(np.nanmedian([d["photometric_boundary_jump"] for d in r5_diag_by_seed.values()])) if r5_diag_by_seed else None,
            photometric_drift_median=float(np.nanmedian([d["photometric_drift_median"] for d in r5_diag_by_seed.values()])) if r5_diag_by_seed else None,
            per_seed={str(k): v for k, v in sorted(r5_diag_by_seed.items())},
        ),
        long_horizon=dict(
            n=len(pop_r4), vi50=vi50_r4, vi50_ci95=[lo_r4, hi_r4], censoring_rate=cens_r4,
            theta_R4_median=float(np.median(theta_R4_episode[np.isfinite(theta_R4_episode)]))
                             if np.any(np.isfinite(theta_R4_episode)) else None,
            kaplan_meier=dict(t=grid_r4.tolist(), S=surv_r4.tolist()),
            events=[dict(seed=s, time=e.time, censored=e.censored) for s, e in zip(pop_r4_seeds, pop_r4)],
        ),
    )
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
