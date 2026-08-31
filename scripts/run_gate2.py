"""GATE 2 (AGENT.md M5, "the instrument tax"): the M2 mutant sweep, run a
second time, measured through the real Phi pipeline (SAM2 tracking +
DINOv2 re-identification on Modal, from rendered video) instead of
ground-truth state. Same mutants, same detector math (sigma_existence,
sigma_kinematic, estimate_threshold, extract_event -- all imported
unchanged from detect/), same reference ensemble and thresholds as GATE 1
(scripts/run_l0_demo.py) computed once in state space. The ONLY thing that
differs between a mutant instance's L0 (state) event and its L1 (Phi)
event is the measurement channel -- that delta IS the instrument error
floor AGENT.md's M5 "done when" asks to publish.

Scope (2026-08: manifest is now selectable via VITALS_MANIFEST, matching
run_l0_demo.py's own convention -- was hardcoded to occlusion_corridor.
yaml only; generalized once ramp_descent_high_friction needed a GATE 2
pass too. See AGENT.md M2.5 for the real drift bug found and fixed doing
this -- `remote/modal_app.py::run_gate2_episode` had its own separate,
occlusion_corridor-hardcoded reconstruction logic, now unified with
`vitals.adapters.video_model.track_and_reconstruct`, the same function
real-model scoring already uses):
- null, vanish, velocity_freeze, jitter, and wrong_gravity for scenarios
  where it means something (see build_instances()) -- 5 of the 7 library
  mutants. `duplicate` and `teleport` both need K>1 object reconstruction,
  which the shared track/reconstruct path is still single-object only for
  (AGENT.md M2.5's own documented gap for occlusion_corridor_
  interpenetration) -- not attempted here either, same reasoning.
- Each rendered clip is truncated to T_RENDER frames, not the manifest's
  full horizon -- render + real GPU tracking cost scales with clip length
  and this is a validation pass, not a production sweep. For
  occlusion_corridor, that's ~6.7s (200 frames), widened from an initial
  3.3s after measuring that velocity_freeze needs longer than that to
  cross threshold on real held-out bases -- see T_RENDER's own comment.
  For a scenario with a shorter native horizon (e.g.
  ramp_descent_high_friction's own 6.0s), T_RENDER is capped to that
  scenario's own total frame count minus 1, not the fixed 200 -- a fixed
  value tuned for one scenario's own clip length must never silently ask
  to render MORE frames than a shorter scenario actually has.
  t_star=25% of the full horizon (matching GATE 1b's own demo cases). The
  reference ensemble and thresholds are NOT truncated -- estimate_
  threshold needs the full M=30 references at their own native length,
  and detect/statistics.py's `_common_T` already tolerates a shorter
  candidate by design (existing code, not something added for this
  script).

    python3 scripts/run_gate2.py
    VITALS_MANIFEST=configs/manifests/ramp_descent_high_friction.yaml python3 scripts/run_gate2.py
"""
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import functools
import json
import numpy as np
import yaml
import modal
from PIL import Image

from vitals.types import EpisodeSpec, Trajectory
from vitals.physics import make_backend
from vitals.mutants import library as mut
from vitals.detect.statistics import sigma_existence, sigma_kinematic, kinematic_axes_for
from vitals.detect.thresholds import estimate_threshold
from vitals.detect.events import extract_event
from vitals.phi import scene_geometry as sg
from vitals.render.mujoco_renderer import MujocoRenderer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-shilpishah-Downloads-important-code-research-midcentury"
                        "/4b7c4a29-48cf-41cf-bb25-20c1250e0978/scratchpad/gate2_episodes")
SCRATCH.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = os.environ.get("VITALS_MANIFEST", "configs/manifests/occlusion_corridor.yaml")
manifest = yaml.safe_load((ROOT / MANIFEST_PATH).read_text())
SPEC = EpisodeSpec(name=manifest["name"], scene=manifest["scene"], target_property=manifest["target_property"],
                    band=manifest["band"], lam=manifest["lam"], n_reference=manifest["n_reference"],
                    horizon_s=manifest["horizon_s"], fps=manifest["fps"], seed=manifest.get("seed", 0),
                    perturb_mode=manifest.get("perturb_mode", "full"))
M, ALPHA = SPEC.n_reference, 0.01
# 200 (~6.67s @ 30fps) was tuned specifically for occlusion_corridor's own
# 8s horizon -- widened from an initial 100 (~3.3s) after measuring that
# velocity_freeze needs ~5.4s to cross threshold on a real held-out base
# there. A scenario with a SHORTER native horizon (ramp_descent_high_
# friction: 6.0s = 180 frames) must never be asked to render more frames
# than it actually has -- capped to that scenario's own total minus 1
# instead. min(200, ...) leaves occlusion_corridor byte-identical to
# before this existed (200 < 240-1).
TOTAL_FRAMES = int(SPEC.horizon_s * SPEC.fps)
T_RENDER = min(200, TOTAL_FRAMES - 1)
T_STAR = round(SPEC.horizon_s * 0.25, 2)   # matches GATE 1b's own demo cases
N_NULL = 10
N_PER_MUTANT = 8

# Same set, same reasoning, as run_l0_demo.py's own NEAR_FLAT_SCENARIOS --
# duplicated (not imported) since these are two independent scripts kept
# in sync by hand, the same convention this file's own R5-axes comment
# already documents for why that's acceptable here. Keep the two
# constants identical if either ever changes.
NEAR_FLAT_SCENARIOS = ("occlusion_corridor", "collision")

CAM = sg.get(SPEC.name)["camera"]
PARK = np.array([0.0, 0.0, -50.0])   # off-camera park position, render-only -- see module docstring
POSITION_ERROR_ID_SWITCH_M = 0.5     # ~3x the ball's own diameter -- clearly a different object

# R5's own axes restriction (AGENT.md M2.6) -- same
# functools.partial(sigma_kinematic, axes=kinematic_axes_for(SPEC.name))
# pattern as run_l0_demo.py's own STATS, and MUST match it exactly: this
# script's whole point is comparing an L1 candidate against GATE 1's own
# state-space-calibrated theta, computed with the SAME sigma_kinematic. A
# mismatched axes restriction between the two scripts would silently
# compare incompatible statistics.
#
# R2's own obj=0 restriction (2026-08, a real bug found running GATE 2 on
# occlusion_corridor_distractor for the first time -- see sigma_
# existence's own docstring for the full mechanism) is DELIBERATELY NOT
# shared with run_l0_demo.py's own STATS: GATE 2's own L1 candidates
# (real video, `track_and_reconstruct`'s single-object return, M5.6) are
# always K=1, so whole-scene object-count existence spuriously fires
# against a K>1 reference ensemble (occlusion_corridor_distractor/
# _moving) regardless of tracking quality. run_l0_demo.py's own state-
# space mutant candidates DO need whole-count (unrestricted) existence --
# that is what makes `duplicate` (an EXTRA object, at a K the reference
# never had) detectable at all, and GATE 2 never tests `duplicate` in the
# first place (this module's own docstring: K>1 reconstruction is still
# single-object only). obj=0 is a strict no-op for every K=1 scenario
# (occlusion_corridor, ramp_descent_high_friction, projectile) -- count
# and single-object presence are identical there.
STATS = {"R2": functools.partial(sigma_existence, obj=0),
         "R5": functools.partial(sigma_kinematic, axes=kinematic_axes_for(SPEC.name))}
roll = make_backend("mujoco", scene=SPEC.scene)
renderer = MujocoRenderer(SPEC.scene, height=240, width=320)

APP_NAME = "vitals-phi"
VOLUME_NAME = "vitals-model-cache"


def build_reference(spec, M, seed0=1000):
    return [roll(spec, seed0 + i) for i in range(M)]


def truncate(traj, T):
    return Trajectory(t=traj.t[:T].copy(), pos=traj.pos[:T].copy(), quat=traj.quat[:T].copy(),
                       present=traj.present[:T].copy(), names=list(traj.names), meta=dict(traj.meta))


def evaluate(cand, refs, thetas, dt, t_max):
    sigmas = {k: fn(cand, refs) for k, fn in STATS.items()}
    return extract_event(sigmas, thetas, dt, t_max)


def build_instances():
    """(mutant_type, Mutant, episode_name) for the whole GATE 2 pool. Each
    instance gets its own fresh held-out base rollout -- seed range
    (50000+) deliberately disjoint from GATE 1's own reference (1000+) and
    held-out (9000+) ranges, so this experiment never silently reuses a
    GATE 1 trajectory.

    wrong_gravity is scenario-conditional (2026-08, generalized once
    ramp_descent_high_friction needed its own GATE 2 pass -- was
    unconditionally excluded here before, occlusion_corridor-only): still
    excluded for NEAR_FLAT_SCENARIOS (occlusion_corridor* AND `collision`
    as of 2026-08's Phase 1 scenario-diversity pass -- same symptom,
    independently confirmed: collision's own wrong_gravity fired 0/8 at
    L0, matching occlusion_corridor's own known near-no-op, not assumed
    to generalize without checking) -- NOT removed from mutants/library.py,
    which is untouched -- confirmed directly, not assumed: tested
    factor=0.2 AND factor=2.0 (well past the demo's own 0.6) against 3
    different held-out bases at the FULL 8s horizon, 18 checks total,
    zero fired -- occlusion_corridor's ball has almost no vertical
    excursion to begin with, so scaling it by any factor still produces a
    perturbation the reference ensemble's own z-spread absorbs as noise.
    Unlike run_l0_demo.py's own build_demo_cases (which swaps in
    `duplicate` for these scenarios instead), GATE 2 has NO substitute
    mutant to offer here -- `duplicate` needs K>1 object reconstruction,
    which the shared track/reconstruct path is still single-object only
    for (this module's own docstring) -- so NEAR_FLAT_SCENARIOS simply
    runs one fewer mutant type here (4, not 5), the same "no correct
    thing to test" honesty as everywhere else in this project, not a
    fabricated substitute. Included for every other scenario (ramp_descent
    family, projectile), at the SAME factor=0.6 run_l0_demo.py's own
    build_demo_cases() already uses as its default there -- not yet
    separately verified to fire reliably through the REAL video pipeline
    at this scenario's own GATE 2 truncation length; check the printed
    GATE 2 summary and retune factor here if it stays censored, the same
    way jitter's own sigma below was tuned. jitter's sigma matches GATE 1's
    own per-scenario value (0.05 occlusion_corridor*, 0.8 collision, 0.15
    everything else) rather than staying uniform -- collision's own 0.15
    stopped firing once SCENARIO_KINEMATIC_AXES restricted it to X-only
    (2026-08, fixing collision's GATE 2 null false-positive): per-frame
    IID jitter noise on a single axis needs a larger sigma to reliably
    produce a SUSTAINED crossing (events.py's pi=0.3s persistence), the
    same finding GATE 1's own run_l0_demo.py made and already documents in
    full. 0.8 (not re-derived independently here) confirmed to still fire
    reliably through the REAL video pipeline too, not assumed to transfer
    automatically from the state-space result -- see the printed GATE 2
    summary from the run that added this."""
    # Episode names are scenario-qualified (2026-08, a real bug found
    # directly, not theorized): two run_gate2.py invocations for DIFFERENT
    # scenarios, launched concurrently, both used the SAME bare names
    # (gate2_null_0, ...) for their own SCRATCH subdirectory AND remote
    # Modal volume path (`/episodes/{name}`) -- a race where one run's
    # own frames/meta.json silently overwrote the other's, producing a
    # T-shape mismatch (occlusion_corridor's own T_RENDER=200 candidate
    # scored against a T=179 trajectory that was actually ramp_descent_
    # high_friction's). Prefixing with SPEC.name makes two concurrent
    # scenarios' own episode names disjoint by construction.
    prefix = f"gate2_{SPEC.name}"
    out = []
    seed = 50000
    for i in range(N_NULL):
        base = roll(SPEC, seed); seed += 1
        out.append(("null", mut.null(base), f"{prefix}_null_{i}"))
    for i in range(N_PER_MUTANT):
        base = roll(SPEC, seed); seed += 1
        out.append(("vanish", mut.vanish(base, t_star=T_STAR), f"{prefix}_vanish_{i}"))
    for i in range(N_PER_MUTANT):
        base = roll(SPEC, seed); seed += 1
        out.append(("velocity_freeze", mut.velocity_freeze(base, t_star=T_STAR), f"{prefix}_velocity_freeze_{i}"))
    if not SPEC.name.startswith(NEAR_FLAT_SCENARIOS):
        for i in range(N_PER_MUTANT):
            base = roll(SPEC, seed); seed += 1
            out.append(("wrong_gravity", mut.wrong_gravity(base, factor=0.6), f"{prefix}_wrong_gravity_{i}"))
    jitter_sigma = 0.05 if SPEC.name.startswith("occlusion_corridor") else (
        0.8 if SPEC.name == "collision" else 0.15)
    for i in range(N_PER_MUTANT):
        base = roll(SPEC, seed); seed += 1
        out.append(("jitter", mut.jitter(base, sigma=jitter_sigma), f"{prefix}_jitter_{i}"))
    return out


def render_episode(traj, name):
    """traj: already truncated to T_RENDER. Renders frames + gt_masks +
    meta.json into SCRATCH/name -- same format run_reidentification /
    run_gate2_episode already consume (gen_full_episode.py's pattern).

    A render-only COPY substitutes a fixed off-camera position for any
    frame the object doesn't truly exist at (present=False) or has no
    defined position for (pos NaN, e.g. `vanish` after t*) -- MuJoCo can't
    render a NaN qpos, and this project has no scene-editing mechanism to
    actually remove a geom mid-clip. The ORIGINAL traj (with real NaN /
    present=False) is never touched -- it's what GATE 2 scores against,
    completely separate from what gets rendered."""
    render_traj = traj.copy()
    hidden = ~traj.present[:, 0] | np.isnan(traj.pos[:, 0]).any(axis=-1)
    render_traj.pos[hidden, 0] = PARK

    frames, gt = renderer.render(render_traj, cameras=CAM)

    out_dir = SCRATCH / name
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    for i in range(traj.T):
        Image.fromarray(frames.rgb[i]).save(out_dir / "frames" / f"{i:05d}.jpg")

    gt_masks = (gt.segmentation == 0)
    np.save(out_dir / "gt_masks.npy", gt_masks)

    # secondary_gt_masks.npy (2026-08, the occlusion_corridor_moving
    # wiring fix): object index 1 (the occluder, declared second in
    # occlusion_corridor_moving.xml, same body-order convention as index
    # 0 -> the ball everywhere else in this project) -- ONLY saved for a
    # scenario that actually registers `occluder_geometry` (today: just
    # this one), not written as dead weight for every other scenario.
    # Mutants only ever act on obj=0 (the ball); the occluder's own real
    # motion is untouched, so this file lets GATE 2 exercise the SAME
    # dynamic-occluder tracking `track_and_reconstruct` now supports,
    # instead of silently running occluder-unaware.
    if sg.get(SPEC.name).get("occluder_geometry") is not None:
        secondary_gt_masks = (gt.segmentation == 1)
        np.save(out_dir / "secondary_gt_masks.npy", secondary_gt_masks)

    # scenario_name (not plane_z directly) is what run_gate2_episode now
    # needs -- it looks up its own scene_geometry config generically via
    # track_and_reconstruct (AGENT.md M2.5), the same way every other Phi
    # caller already does, rather than being handed occlusion_corridor-
    # shaped kwargs (plane_z) directly.
    meta = dict(T=int(traj.T), fps=SPEC.fps, width=renderer.width, height=renderer.height,
                cam_pos=gt.cam_pos.tolist(), cam_mat=gt.cam_mat.tolist(),
                fovy_deg=float(gt.fovy_deg), scenario_name=SPEC.name)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    return out_dir


def upload_all(names):
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for name in names:
            batch.put_directory(str(SCRATCH / name), f"/episodes/{name}")
    print(f"uploaded {len(names)} episodes -> volume:{VOLUME_NAME}")


def traj_from_modal_result(res):
    t = np.array(res["traj_t"])
    T = len(t)
    pos = np.full((T, 1, 3), np.nan)
    for i, row in enumerate(res["traj_pos"]):
        if row[0][0] is not None:
            pos[i, 0] = row[0]
    present = np.array(res["traj_present"], dtype=bool)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1, 1))
    return Trajectory(t=t, pos=pos, quat=quat, present=present, names=["ball"])


def position_error(l0_cand, l1_traj):
    """Mean metric distance between L1's reconstructed position and L0
    ground truth, restricted to frames where BOTH are defined -- L1's pos
    is NaN during any gap (closed or not, reconstruct.py's own design);
    L0's can be NaN too (`vanish` after t*, genuinely doesn't exist)."""
    Tc = min(l0_cand.T, l1_traj.T)
    gt = l0_cand.pos[:Tc, 0]
    l1 = l1_traj.pos[:Tc, 0]
    both = np.isfinite(gt).all(axis=-1) & np.isfinite(l1).all(axis=-1)
    if not both.any():
        return None, 0
    err = np.linalg.norm(gt[both] - l1[both], axis=-1)
    return float(err.mean()), int(both.sum())


def id_switch_count(l0_cand, l1_traj, reid_events):
    """A successful reid match (matched=True) is an ID-switch if the
    position error at that exact frame, once L1 has a defined position
    there, exceeds POSITION_ERROR_ID_SWITCH_M -- a re-identification that
    landed on something far from where the real object actually was."""
    switches, checked = 0, 0
    Tc = min(l0_cand.T, l1_traj.T)
    for frame_idx, _score, matched, _method in reid_events:
        if not matched or frame_idx >= Tc:
            continue
        gt_p, l1_p = l0_cand.pos[frame_idx, 0], l1_traj.pos[frame_idx, 0]
        if not (np.isfinite(gt_p).all() and np.isfinite(l1_p).all()):
            continue
        checked += 1
        if np.linalg.norm(gt_p - l1_p) > POSITION_ERROR_ID_SWITCH_M:
            switches += 1
    return switches, checked


def main():
    print(f"GATE 2  scenario={SPEC.name}  T_RENDER={T_RENDER} ({T_RENDER / SPEC.fps:.2f}s)  "
          f"t_star={T_STAR}s  N_NULL={N_NULL}  N_PER_MUTANT={N_PER_MUTANT}")

    refs = build_reference(SPEC, M)
    dt = refs[0].dt
    t_max_render = T_RENDER * dt
    thetas = {k: estimate_threshold(refs, fn, alpha=ALPHA) for k, fn in STATS.items()}
    # extract_event compares sigma (sized to the CANDIDATE's own T, per
    # detect/statistics.py) elementwise against theta (sized to refs' full
    # T=240) -- fine when candidate and refs share a length (GATE 1, always
    # full-length), but every GATE 2 candidate is truncated to T_RENDER, so
    # theta must be sliced to match or the comparison's shapes don't
    # broadcast. Threshold VALUES are unaffected by slicing -- theta(t) for
    # t < T_RENDER is identical either way, this only trims the unused tail.
    thetas_render = {k: v[:T_RENDER] for k, v in thetas.items()}
    print(f"reference ensemble M={M} built, thresholds calibrated")

    instances = build_instances()
    print(f"{len(instances)} mutant instances built")

    print("rendering locally...")
    for mtype, m, name in instances:
        m.traj = truncate(m.traj, T_RENDER)
        render_episode(m.traj, name)
    print("render done")

    upload_all([name for _, _, name in instances])

    print("spawning Modal GATE 2 runs (concurrent)...")
    fn = modal.Function.from_name(APP_NAME, "run_gate2_episode")
    calls = [(mtype, m, name, fn.spawn(name=name)) for mtype, m, name in instances]

    rows = []
    for mtype, m, name, call in calls:
        try:
            res = call.get()
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            continue

        l0_event = evaluate(m.traj, refs, thetas_render, dt, t_max_render)
        l1_traj = traj_from_modal_result(res)
        l1_event = evaluate(l1_traj, refs, thetas_render, dt, t_max_render)
        pos_err, pos_err_n = position_error(m.traj, l1_traj)
        id_switches, id_checked = id_switch_count(m.traj, l1_traj, res["reid_events"])

        row = dict(mtype=mtype, name=name, risk_expected=m.risk_expected,
                   l0_risk=l0_event.risk, l0_time=l0_event.time, l0_censored=l0_event.censored,
                   l1_risk=l1_event.risk, l1_time=l1_event.time, l1_censored=l1_event.censored,
                   mean_iou=res["mean_iou_all"], occlusion_flag_accuracy=res["occlusion_flag_accuracy"],
                   position_error=pos_err, position_error_n=pos_err_n,
                   id_switches=id_switches, id_checked=id_checked, n_reid_events=res["n_reid_events"])
        rows.append(row)
        bias = (f"{l1_event.time - l0_event.time:+.2f}"
                if not l0_event.censored and not l1_event.censored else "   -")
        print(f"  {name:<28} L0={l0_event.risk or '-':<5}{'cens' if l0_event.censored else f'{l0_event.time:.2f}':>6}"
              f"  L1={l1_event.risk or '-':<5}{'cens' if l1_event.censored else f'{l1_event.time:.2f}':>6}"
              f"  bias={bias}  iou={res['mean_iou_all']:.2f}")

    out_path = SCRATCH / f"gate2_results_{SPEC.name}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nraw results -> {out_path}")

    print("\n=== GATE 2 per-mutant summary ===")
    for mtype in ["null", "vanish", "velocity_freeze", "wrong_gravity", "jitter"]:
        group = [r for r in rows if r["mtype"] == mtype]
        if not group:
            continue
        n = len(group)
        l0_fire = sum(1 for r in group if not r["l0_censored"])
        l1_fire = sum(1 for r in group if not r["l1_censored"])
        l0_correct = sum(1 for r in group if r["l0_risk"] == r["risk_expected"])
        l1_correct = sum(1 for r in group if r["l1_risk"] == r["risk_expected"])
        joint_fired = [r for r in group if not r["l0_censored"] and not r["l1_censored"]]
        biases = [r["l1_time"] - r["l0_time"] for r in joint_fired]
        ious = [r["mean_iou"] for r in group]
        occ_acc = [r["occlusion_flag_accuracy"] for r in group]
        pos_errs = [r["position_error"] for r in group if r["position_error"] is not None]
        total_switches = sum(r["id_switches"] for r in group)
        total_checked = sum(r["id_checked"] for r in group)

        print(f"\n{mtype}  (n={n}, risk_expected={group[0]['risk_expected']})")
        print(f"  L0 fired={l0_fire}/{n}  correct_risk={l0_correct}/{n}")
        print(f"  L1 fired={l1_fire}/{n}  correct_risk={l1_correct}/{n}")
        if biases:
            print(f"  event-time bias (L1-L0), n={len(biases)}: mean={np.mean(biases):+.3f}s  "
                  f"std={np.std(biases):.3f}s")
        print(f"  mean mask IoU={np.mean(ious):.3f}  mean occlusion-flag accuracy={np.mean(occ_acc):.3f}")
        if pos_errs:
            print(f"  mean track position error={np.mean(pos_errs):.3f}m (n={len(pos_errs)} instances with "
                  f"overlapping defined frames)")
        print(f"  ID-switches={total_switches}/{total_checked} successful reid matches checked")


if __name__ == "__main__":
    main()
