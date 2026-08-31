"""Assembles Phi's raw outputs (tracked masks, re-identification events) into
an actual vitals.types.Trajectory -- the literal M4 done-criterion (AGENT.md
§7): "Phi produces a Trajectory from video, and the same sigma_k functions
run on it unmodified." Nothing upstream of this module (segmentation.py,
reidentify.py, motion_prior.py) produces a Trajectory; this is the missing
glue.

What this module does NOT do, on purpose, for v1:

- No 6-DoF pose (Kabsch fit over tracked points), no point tracking at all
  -- both are real M4 component-table entries (AGENT.md §7) and both are
  DEFERRED, not silently skipped. Checked directly against the detectors
  that matter for v1: `detect/statistics.py::sigma_existence` reads only
  `Trajectory.present`; `sigma_kinematic` reads only `Trajectory.pos`.
  Neither reads `.quat`. `detect/events.py`'s own comment confirms v1 "only
  ever registers R2 and R5" (P2 existence, P4 kinematic) -- exactly the
  proposal's stated minimum-viable scope (VITALS_proposal.pdf §15), which
  explicitly defers the relation head and its pose/tracking complexity.
  `Trajectory.quat` is a required dataclass field regardless, so it's
  filled with a placeholder identity quaternion (IDENTITY_QUAT below) --
  inert for v1's active detectors, NOT a claim that orientation is
  measured. Do not read anything from `.quat` on a Phi-reconstructed
  Trajectory until real pose fitting exists.

- No monocular depth model, no multi-view triangulation. Position is
  recovered by ray/plane intersection: given the scene's own known camera
  calibration (same synthetic-only convenience already used throughout
  phi/motion_prior.py) and a KNOWN, FIXED resting height for the tracked
  object, unproject each frame's mask centroid to the 3D point where the
  camera ray hits that height plane. This is the exact geometric inverse
  of render/mujoco_renderer.py's own pinhole projection, so it's only ever
  as good as (a) the plane assumption holding and (b) mask-centroid
  position being unbiased -- see the next paragraph for exactly where each
  breaks.

Two honest scope limits this method has, not silently:

1. Valid ONLY for flat-ground (single-plane) motion. occlusion_corridor
   satisfies this (ball height is ~constant while resting/rolling on flat
   ground); ramp_descent does NOT (the ball's true z varies continuously
   down the incline) -- reconstruct_trajectory raises rather than silently
   returning a wrong position for a scenario whose motion isn't planar.
   Real 3D lifting (monocular depth or multi-view triangulation, the M4
   component table's actual planned path) is required before this
   generalizes past flat-ground scenarios -- same gap already named in
   motion_prior.py and AGENT.md's T3.
2. Mask centroid, not a tracked point. The proposal is explicit that a
   centroid is biased the instant an object is partially occluded. This
   matters less here than it otherwise would, because occlusion_corridor's
   occlusion is binary (the wall is large enough that the ball is either
   fully visible or fully hidden, not routinely partially clipped) and
   because reidentify.py's own reappearance-frame fix (AGENT.md defect #12)
   already requires >=50% of the object's own max area before treating a
   frame as "visible" -- but it is still a centroid, not a tracked point,
   and should be replaced once real point tracking exists.

A bug found and fixed while validating this module, worth recording so it
doesn't get reintroduced: `present` must NOT be "mask non-empty at this
instant." Trajectory's own docstring is explicit -- present means "object
exists in the world (NOT 'is visible')" -- and mask-non-empty measures
visibility, not existence, which conflates "occluded" with "gone" and
silently defeats the entire point of the re-identification machinery
(segmentation.py's cliff + reidentify.py's search-and-reprompt exist
specifically to distinguish those two states). Caught empirically: feeding
raw mask-presence into a reconstructed reference ensemble produced NaN
thresholds downstream (occlusion gaps meant some references had
simultaneously-missing position data at some time bins). The fix:
`present` follows reidentify.py's own reacquisition verdict -- True for
directly-visible frames AND for frames inside a gap that a later
`reid_event` confirms was successfully closed (the object was there the
whole time; re-identification is what proves it), False only for a gap
that never closes by the end of the clip. `pos` stays NaN for any
non-directly-visible frame regardless -- position genuinely isn't measured
during a gap, closed or not, and sigma_kinematic already has documented,
correct handling for that (NaN distance -> inf, "vanished object:
undefined kinematics").
"""
from __future__ import annotations
import os
import numpy as np
from ..types import Trajectory

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz -- placeholder, see module docstring


def centroid(mask):
    """(px, py) of a boolean mask's True pixels, or None if empty. Same
    definition as motion_prior.centroid -- kept independent (not imported)
    because the two modules are allowed to diverge once one of them moves
    to real point tracking."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _gap_explained_by_occluder(gap_start, gap_end, known_occluded_frames):
    """True iff EVERY recorded search attempt within [gap_start, gap_end)
    found the predicted position inside a KNOWN occluder -- i.e. physics
    never once looked for the object anywhere unexplained during this gap.
    Requires at least one recorded attempt; an empty window (nothing
    recorded, e.g. a gap shorter than forgiveness_frames) is NOT considered
    explained by this function -- that case is already handled separately
    by the forgiveness check in `_existence_mask`."""
    attempts = [v for k, v in known_occluded_frames.items() if gap_start <= k < gap_end]
    return len(attempts) > 0 and all(attempts)


def _existence_mask(masks, reid_events, T, forgiveness_frames=0, known_occluded_frames=None):
    """True at frame i if the object is directly visible there, OR the gap
    containing i is short enough that reidentify.py's own
    `forgiveness_frames` tolerance never treated it as a loss worth
    searching for at all, OR a later reid_event confirms a LONGER gap
    containing i was successfully closed, OR EVERY search attempt made
    during the gap found the predicted position inside a KNOWN occluder
    (see `_gap_explained_by_occluder`). False only for a gap that exceeded
    forgiveness, was never explained by a known occluder throughout, and
    never closed (search failed, or the clip ended before it could) -- see
    module docstring.

    forgiveness_frames must match the value actually passed to
    track_with_reidentification for these masks/reid_events (AGENT.md
    defect #20). Found the hard way, on a completely UNDISTURBED
    (`null` mutant) GATE 2 episode: with forgiveness_frames left at its
    old implicit 0, a single one-frame SAM2 flicker -- exactly the kind of
    benign hiccup forgiveness_frames=1 exists to absorb, and reidentify.py
    correctly never even triggered a search for it, so no reid_event was
    ever generated to "close" it -- still got recorded as an unresolved
    gap (present=False) by this function's old logic, which only trusted
    RECOVERY evidenced by a reid_event. sigma_existence, calibrated against
    state-space references where `present` structurally never dips, then
    reads that single manufactured False as a genuine existence violation:
    measured directly, this was the dominant cause of a 4/10 spurious R2
    firing rate on defect-free episodes that never needed reidentification
    at all (n_reid_events=0). A gap reidentify.py itself never considered
    worth searching for is not evidence the object stopped existing --
    it's ordinary segmentation noise, the same status pixel jitter already
    has elsewhere in this pipeline.

    known_occluded_frames: reidentify.py's ReidentifyResult.
    known_occluded_frames, or None (AGENT.md M5, existence-statistic
    hardening, 2026-08). This is a SEPARATE, later fix from the
    forgiveness one above -- found via GATE 2's own full sweep, not the
    single-episode organic validation: forgiveness/reid_event closure
    together still left a ~90% R2 false-positive rate on completely
    UNDISTURBED episodes, because a genuinely, correctly occluded object
    (still behind a KNOWN wall, physics correctly never even attempting a
    search -- `n_reid_events=0`) has NO reid_event to close its gap, and
    a gap that long is never within forgiveness either. Two threshold-side
    fixes were tried and both failed identically (fixed the false-positive
    rate perfectly, destroyed sensitivity to genuine `vanish` events
    almost completely) -- because reusing/recalibrating a THRESHOLD can't
    repair a STATISTIC that conflates "confirmed gone" and "honestly
    unresolved" into the same value. This is the statistic-level fix
    instead: a gap that's explained by a known occluder throughout is
    treated as present, exactly like state-space treats ordinary occlusion
    (invisible, not a violation) -- reusing the SAME occluder_bounds
    machinery defect #17 already built, just finally recorded and consulted
    here instead of only inside the search loop itself. A pure gap-DURATION
    statistic was considered and rejected: it can't be independently
    checked, either, since `vanish`'s t_star fires independent of the
    object's position, so a real vanish event that happens to occur near a
    real occluder is philosophically, not just practically, ambiguous from
    pixels alone -- this fix targets the common, spatially-unexplained
    case, not a claim of resolving that harder one."""
    known_occluded_frames = known_occluded_frames or {}
    visible = np.array([bool(masks.get(i) is not None and masks[i].sum() > 0) for i in range(T)])
    present = visible.copy()
    gap_start = None
    for i in range(T):
        if not visible[i] and gap_start is None:
            gap_start = i
        elif visible[i] and gap_start is not None:
            gap_len = i - gap_start
            closed = (gap_len <= forgiveness_frames
                      or _gap_explained_by_occluder(gap_start, i, known_occluded_frames)
                      or any(matched and gap_start <= frame_idx <= i
                             for frame_idx, _score, matched, _method in reid_events))
            if closed:
                present[gap_start:i] = True
            gap_start = None
    if gap_start is not None:
        explained = _gap_explained_by_occluder(gap_start, T, known_occluded_frames)
        if T - gap_start <= forgiveness_frames or explained:
            present[gap_start:T] = True
    return present


def unproject_to_known_plane(px, py, cam_pos, cam_mat, fovy_deg, width, height,
                              plane_point, plane_normal):
    """General ray/plane intersection -- the same geometric method
    `unproject_to_plane` uses, generalized to ANY known, fixed plane
    (point + normal), not just a horizontal one at a fixed z.
    `unproject_to_plane(..., plane_z)` is the special case
    `plane_point=(0,0,plane_z), plane_normal=(0,0,1)`, kept as its own
    function since every existing flat-ground call site already uses the
    simpler scalar convention and there's no reason to touch working code
    to adopt the general form.

    AGENT.md M4.5's "known fixed plane" case (e.g. ramp_descent's incline)
    -- valid ONLY while the object's true motion is confined to THIS
    SINGLE plane; same scope limit as `unproject_to_plane` itself (module
    docstring limit 1), just for an arbitrary plane instead of z=const. A
    scenario whose motion crosses between multiple known planes (e.g.
    ramp_descent's incline transitioning onto flat ground) needs the
    caller to pick the right plane per-frame -- see
    `reconstruct_trajectory`'s own `plane_pieces` parameter, which does
    exactly that for ramp_descent specifically.

    Returns None if the ray is parallel to the plane or the intersection
    is behind the camera (both indicate a degenerate/invalid call, not a
    position to silently fabricate) -- same contract as
    `unproject_to_plane`."""
    f = height / (2 * np.tan(np.radians(fovy_deg) / 2))
    dir_cam = np.array([(px - width / 2) / f, (height / 2 - py) / f, -1.0])
    dir_world = cam_mat @ dir_cam
    n = np.asarray(plane_normal, dtype=float)
    n = n / np.linalg.norm(n)
    p0 = np.asarray(plane_point, dtype=float)
    denom = n @ dir_world
    if abs(denom) < 1e-8:
        return None
    t = (n @ (p0 - cam_pos)) / denom
    if t <= 0:
        return None
    return cam_pos + t * dir_world


def unproject_to_plane(px, py, cam_pos, cam_mat, fovy_deg, width, height, plane_z):
    """Exact geometric inverse of mujoco_renderer.py's pinhole projection
    (validated against it directly, see tests/test_reconstruct.py): given a
    pixel and the camera's own known calibration, returns the 3D world
    point where the camera ray through that pixel intersects the horizontal
    plane z=plane_z. None if the ray is parallel to the plane or the
    intersection is behind the camera (both indicate a degenerate/invalid
    call, not a position to silently fabricate).

    Thin wrapper over `unproject_to_known_plane`'s general form -- kept as
    its own function (not just callers passing plane_point/plane_normal
    directly) because it's the common case and every existing flat-ground
    call site already uses this exact scalar signature."""
    return unproject_to_known_plane(px, py, cam_pos, cam_mat, fovy_deg, width, height,
                                     plane_point=(0.0, 0.0, plane_z), plane_normal=(0.0, 0.0, 1.0))


def _project_raw(pos3d, cam_pos, cam_mat, fovy_deg, width, height):
    """Perspective projection WITHOUT the depth<=0 -> None guard --
    returns a (px, py) pair regardless, even a nonsensical one for a point
    behind the camera. Exists purely so `fit_ballistic_trajectory`'s
    optimizer has a well-defined, differentiable residual at every
    intermediate parameter guess during fitting (an iterative solver can
    easily pass through a behind-camera guess on its way to a good one);
    `project_to_pixel` below is the public function and keeps the None
    contract unchanged for every other caller."""
    rel = np.asarray(pos3d) - cam_pos
    cam_frame = cam_mat.T @ rel
    x_r, y_u, z_negf = cam_frame
    depth = -z_negf
    f = height / (2 * np.tan(np.radians(fovy_deg) / 2))
    px = width / 2 + f * (x_r / depth)
    py = height / 2 - f * (y_u / depth)
    return depth, float(px), float(py)


def project_to_pixel(pos3d, cam_pos, cam_mat, fovy_deg, width, height):
    """Exact geometric inverse of `unproject_to_plane` -- given a 3D world
    point and the camera's own known calibration, returns the (px, py)
    pixel it projects to, or None if the point is behind the camera.
    Validated as an exact round-trip pair (project then unproject_to_plane,
    or vice versa) to ~1e-7 -- see tests/test_reconstruct.py. Used by
    motion_prior.py's metric-anchored prediction (AGENT.md defect #16) to
    convert an extrapolated metric position back to a pixel location for
    candidate gating."""
    depth, px, py = _project_raw(pos3d, cam_pos, cam_mat, fovy_deg, width, height)
    if depth <= 0:
        return None
    return px, py


def _ballistic_position(p, t, gravity_z):
    x0, y0, z0, vx0, vy0, vz0 = p
    return np.array([x0 + vx0 * t, y0 + vy0 * t, z0 + vz0 * t + 0.5 * gravity_z * t * t])


def _ballistic_residuals(p, times, pixel_obs, cam_pos, cam_mat, fovy_deg, width, height, gravity_z,
                          size_obs=None, object_radius=None, size_weight=1.0):
    """size_obs/object_radius: OPTIONAL apparent-size residual, the fix for
    a real, structural ambiguity found empirically (not a numerical bug):
    position-only residuals have a genuine monocular scale/depth degeneracy
    -- a trajectory further from the camera moving faster can project to
    pixel positions nearly indistinguishable from one closer and slower,
    confirmed directly on a real rendered clip (a fit 0.7m from the true
    3D trajectory had a BETTER position-only residual than the true
    parameters did: 0.166 vs 3.355). The object's known physical radius
    (e.g. the ball's 0.15m, same "known scene geometry" status as camera
    calibration) plus its OBSERVED apparent size in pixels is an
    independent depth cue position tracking alone doesn't have -- apparent
    size shrinks/grows with depth via simple similar-triangles, breaking
    the ambiguity. Without it (size_obs=None), falls back to the
    position-only residual exactly as before."""
    n = len(times)
    out = np.empty(3 * n if size_obs is not None else 2 * n)
    f = height / (2 * np.tan(np.radians(fovy_deg) / 2))
    for i, t in enumerate(times):
        X = _ballistic_position(p, t, gravity_z)
        depth, px, py = _project_raw(X, cam_pos, cam_mat, fovy_deg, width, height)
        if size_obs is not None:
            out[3 * i] = px - pixel_obs[i, 0]
            out[3 * i + 1] = py - pixel_obs[i, 1]
            predicted_radius_px = f * object_radius / depth if depth > 1e-3 else 1e4
            out[3 * i + 2] = size_weight * (predicted_radius_px - size_obs[i])
        else:
            out[2 * i] = px - pixel_obs[i, 0]
            out[2 * i + 1] = py - pixel_obs[i, 1]
    return out


def fit_ballistic_trajectory(pixel_obs, times, cam_pos, cam_mat, fovy_deg, width, height,
                              gravity_z=-9.81, n_restarts=8, max_iter=50,
                              size_obs=None, object_radius=None, size_weight=3.0):
    """AGENT.md M4.5's "true free 3D motion" case (e.g. projectile.xml) --
    recovers a full 3D ballistic trajectory (x0, v0) from a sequence of 2D
    pixel observations, known camera calibration, and KNOWN gravity, by
    nonlinear least squares. Reuses this project's established "borrow
    known physics, don't re-derive it" philosophy
    (`detect/physical_constants.py`, `motion_prior.py`'s own deceleration-
    anchored fit): gravity is GIVEN, not fit, which is what turns an
    otherwise underdetermined monocular 3D recovery problem into a
    well-posed one. Every subsequent instant of a ballistic trajectory is
    fully determined by 6 numbers (initial position + velocity) once the
    constant acceleration is known -- unlike `unproject_to_plane`'s method,
    which needs a known plane at each instant, this needs the WHOLE
    trajectory's shape to be known, fit jointly across every observation
    at once, not frame by frame independently.

    pixel_obs: (N, 2) array of observed (px, py). times: (N,) array of
    elapsed seconds since the FIRST observation (not absolute frame index
    or absolute clip time -- caller's job to convert, matching
    `unproject_to_plane`'s own convention of taking exactly what it needs
    and nothing more).

    Multi-start Levenberg-Marquardt, hand-implemented with plain numpy --
    this project has no scipy dependency (`pyproject.toml`: numpy, pyyaml,
    optionally mujoco/pillow) and one small solver for one well-scoped
    problem doesn't need to add one. A single ray gives NO depth
    information at all (the classic monocular scale ambiguity) -- initial
    guess quality matters a lot here, so this restarts from several
    candidate initial depths along the first observation's own camera ray
    and keeps whichever converged to the lowest residual, rather than
    trusting one arbitrary starting point. Numerical (central-difference)
    Jacobian, not hand-derived analytic derivatives -- 6 parameters and a
    handful of observations is cheap either way, and finite differences
    are far less error-prone to get right than differentiating the full
    projection chain (rotation, perspective divide) by hand.

    size_obs, object_radius, size_weight: OPTIONAL apparent-size residual
    -- the fix for a real, structural ambiguity found empirically while
    building this (not a numerical bug): position-only residuals have a
    genuine monocular scale/depth degeneracy. Confirmed directly on a real
    rendered clip: a fit 0.7m from the TRUE 3D trajectory had a BETTER
    position-only residual (cost/n=0.166) than the true parameters
    themselves (cost/n=3.355) -- the optimizer was doing exactly what it
    was told to do, minimizing pixel error, and a genuinely different 3D
    trajectory explained the observed pixel POSITIONS better. `size_obs`
    (observed apparent object radius/size in pixels per frame, e.g. from
    `mask_diagonal_px`) plus the object's known TRUE physical radius adds
    an independent depth cue position tracking alone doesn't have --
    apparent size shrinks/grows with depth (similar triangles), which
    breaks the position-only ambiguity. Strongly recommended whenever a
    size observation is available; omit only if it genuinely isn't (the
    position-only path is kept, not removed, for that case). The SAME size
    cue also re-centers the restart depth sweep below (depth0 =
    f*object_radius/size_obs[0], confirmed accurate to ~6% against ground
    truth) instead of blindly sweeping 2-20m -- this mattered more than
    the residual term alone: with only the residual term, 2 of 4 tested
    seeds still converged to the wrong basin (~0.5m error) because a bad
    starting depth put the optimizer in a different local minimum
    entirely; centering the sweep on the size-implied depth fixed those
    too (~0.10-0.15m error, all seeds, matching ramp_descent's own
    validated 0.11-0.13m). size_weight default (3.0, not 1.0) was chosen
    by sweeping on real rendered data: 3.0 gave mean error 0.126m across 6
    seeds vs 0.161m at weight 1.0.

    Returns (x0, v0) as two (3,) arrays, or (None, None) if no restart
    converged to a low residual -- a genuine "not recoverable from this
    view" result, not a position to silently fabricate, same discipline
    as `unproject_to_plane`'s own None contract."""
    n = len(np.asarray(times))
    if n < 3:
        return None, None   # 6 unknowns, 2-3 residuals/frame -- need at least 3 frames to be well-posed at all
    best_p, normalized_cost = _fit_ballistic_core(
        pixel_obs, times, cam_pos, cam_mat, fovy_deg, width, height, gravity_z=gravity_z,
        n_restarts=n_restarts, max_iter=max_iter, size_obs=size_obs,
        object_radius=object_radius, size_weight=size_weight)
    if os.environ.get("VITALS_RECON_DEBUG"):
        # 2026-08: added while diagnosing GATE 2's own 0/42 ballistic-
        # reconstruction success rate on projectile (every mutant type,
        # including null with good SAM2 IoU) -- the >4.0 accept/reject
        # threshold and size_weight default were only ever validated
        # against PIXEL-PERFECT ground-truth masks (tests/test_reconstruct.
        # py's own `_projectile_phi_traj`, `gt.segmentation` directly, never
        # real SAM2 output), so seeing the ACTUAL normalized cost real SAM2
        # masks produce -- not just pass/fail -- is what distinguishes "the
        # threshold needs retuning" from "the fit is structurally broken on
        # real masks."
        pixel_obs_arr = np.asarray(pixel_obs, dtype=float)
        size_obs_arr = np.asarray(size_obs, dtype=float) if size_obs is not None else None
        times_arr = np.asarray(times, dtype=float)
        size_range = f"[{size_obs_arr.min():.2f},{size_obs_arr.max():.2f}]" if size_obs_arr is not None else "None"
        print(f"DEBUG fit_ballistic_trajectory: n={n} best_p={best_p} "
              f"normalized_cost={normalized_cost:.3f} (accept threshold=4.0) size_obs_range={size_range}")
        print(f"DEBUG times: first={times_arr[0]:.3f} last={times_arr[-1]:.3f} n={len(times_arr)}")
        print(f"DEBUG pixel_obs (px,py) first5={pixel_obs_arr[:5].tolist()}")
        print(f"DEBUG pixel_obs (px,py) last5={pixel_obs_arr[-5:].tolist()}")
        mid = len(pixel_obs_arr) // 2
        print(f"DEBUG pixel_obs (px,py) mid5={pixel_obs_arr[max(0,mid-2):mid+3].tolist()}")
        print(f"DEBUG pixel_obs x range=[{pixel_obs_arr[:,0].min():.1f},{pixel_obs_arr[:,0].max():.1f}] "
              f"y range=[{pixel_obs_arr[:,1].min():.1f},{pixel_obs_arr[:,1].max():.1f}] "
              f"(frame size {width}x{height})")
        if size_obs_arr is not None:
            print(f"DEBUG size_obs first5={size_obs_arr[:5].tolist()} last5={size_obs_arr[-5:].tolist()}")
    if best_p is None or normalized_cost > 4.0:
        return None, None
    return best_p[:3], best_p[3:]


def _fit_ballistic_core(pixel_obs, times, cam_pos, cam_mat, fovy_deg, width, height,
                        gravity_z=-9.81, n_restarts=8, max_iter=50,
                        size_obs=None, object_radius=None, size_weight=3.0):
    """The actual multi-restart Levenberg-Marquardt optimization --
    extracted from `fit_ballistic_trajectory` (2026-08, building
    `fit_piecewise_ballistic_trajectory` below) so the accept/reject
    decision against a fixed cost threshold is the CALLER's own choice,
    not baked into the fitting routine itself. `fit_ballistic_trajectory`
    applies its own fixed >4.0 rule for a single continuous segment;
    `fit_piecewise_ballistic_trajectory` needs the RAW cost at every
    candidate window size to decide where a segment boundary (a real
    bounce) falls -- it cannot get that from a function that already
    collapsed "how good is this fit" down to accept/reject.

    ALWAYS returns its own best (x0,v0) 6-vector and normalized cost, even
    when the fit is bad -- unlike `fit_ballistic_trajectory`'s own
    (None, None) contract for "not recoverable." Returns (None, inf) only
    when n<3 (under-determined, no fit attempted at all)."""
    pixel_obs = np.asarray(pixel_obs, dtype=float)
    times = np.asarray(times, dtype=float)
    n = len(times)
    if n < 3:
        return None, float("inf")
    if size_obs is not None:
        size_obs = np.asarray(size_obs, dtype=float)
        if object_radius is None:
            raise ValueError("size_obs was given but object_radius wasn't -- "
                              "the apparent-size residual needs the object's known TRUE radius.")

    def residuals(p):
        return _ballistic_residuals(p, times, pixel_obs, cam_pos, cam_mat, fovy_deg, width, height,
                                     gravity_z, size_obs=size_obs, object_radius=object_radius,
                                     size_weight=size_weight)

    f = height / (2 * np.tan(np.radians(fovy_deg) / 2))
    dir_cam0 = np.array([(pixel_obs[0, 0] - width / 2) / f, (height / 2 - pixel_obs[0, 1]) / f, -1.0])
    dir_world0 = cam_mat @ dir_cam0
    dir_world0 = dir_world0 / np.linalg.norm(dir_world0)
    dir_cam_last = np.array([(pixel_obs[-1, 0] - width / 2) / f, (height / 2 - pixel_obs[-1, 1]) / f, -1.0])
    dir_world_last = cam_mat @ dir_cam_last
    dir_world_last = dir_world_last / np.linalg.norm(dir_world_last)

    # Restart depths: centered on the size-implied depth when available --
    # confirmed directly to be accurate to ~6% (f*object_radius/size_obs[0]
    # vs the true depth, checked against real rendered data), a FAR better
    # starting point than a blind sweep. Without a size observation, fall
    # back to the original blind 2-20m sweep (unchanged behavior for that
    # case).
    if size_obs is not None and size_obs[0] > 1e-6:
        depth0_center = f * object_radius / size_obs[0]
        depths = np.linspace(max(0.5, depth0_center * 0.6), depth0_center * 1.4, n_restarts)
    else:
        depths = np.linspace(2.0, 20.0, n_restarts)

    best_p, best_cost = None, np.inf
    for depth0 in depths:
        # v0 guess is NOT left at zero -- an all-zero initial velocity gave
        # the optimizer no directional head start at all and it regularly
        # converged to a badly-fit local minimum. Instead, assume the SAME
        # depth0 at the last observation too (crude -- the object has
        # almost certainly moved in depth, but still gives a real
        # directional estimate) and back out an initial velocity from the
        # resulting displacement, correcting the z component for the KNOWN
        # gravity drop over the same interval so the initial guess isn't
        # fighting its own vertical-motion prior.
        x0_guess = cam_pos + depth0 * dir_world0
        x_last_guess = cam_pos + depth0 * dir_world_last
        dt_span = times[-1] - times[0]
        v0_guess = (x_last_guess - x0_guess) / dt_span if dt_span > 1e-6 else np.zeros(3)
        v0_guess[2] -= 0.5 * gravity_z * dt_span   # undo the gravity term baked into a naive displacement/dt estimate
        p = np.concatenate([x0_guess, v0_guess])
        lam = 1e-3
        cost = np.sum(residuals(p) ** 2)
        for _ in range(max_iter):
            r = residuals(p)
            J = np.empty((len(r), 6))
            eps = 1e-5
            for k in range(6):
                dp = np.zeros(6); dp[k] = eps
                J[:, k] = (residuals(p + dp) - r) / eps
            JTJ = J.T @ J
            JTr = J.T @ r
            step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), -JTr)
            p_new = p + step
            cost_new = np.sum(residuals(p_new) ** 2)
            if cost_new < cost:
                p, cost, lam = p_new, cost_new, lam * 0.5
                if abs(cost - cost_new) < 1e-10:
                    break
            else:
                lam *= 3.0
                if lam > 1e10:
                    break
        if cost < best_cost:
            best_p, best_cost = p, cost

    n_resid_per_frame = 3 if size_obs is not None else 2
    normalized_cost = (best_cost / (n * n_resid_per_frame)) if best_p is not None else float("inf")
    return best_p, normalized_cost


def fit_piecewise_ballistic_trajectory(pixel_obs, times, cam_pos, cam_mat, fovy_deg, width, height,
                                       gravity_z=-9.81, size_obs=None, object_radius=None,
                                       size_weight=3.0, min_segment_frames=5, accept_threshold=4.0,
                                       n_restarts=8, max_iter=50):
    """AGENT.md M4.5's own documented gap, fixed (2026-08, "scope and build
    piecewise ballistic fitting now"): `fit_ballistic_trajectory` fits ONE
    continuous global parabola (x0 + v0*t + 0.5*g*t^2) across every
    observation jointly -- correct for a single flight segment, but a real
    bounce is a velocity DISCONTINUITY no single parabola can represent at
    all. Confirmed directly as the root cause of GATE 2's own 0/42
    ballistic-reconstruction success rate on `projectile`: the true
    ground-truth height trace (checked locally, zero GPU cost, by
    re-rolling the same seed) showed 2-3 real bounces within the exact
    89-frame/2.93s window GATE 2 observes, while `fit_ballistic_
    trajectory`'s own prior validation (`tests/test_reconstruct.py::
    test_projectile_ballistic_reconstruction_error_is_bounded`) only ever
    used the first 20 frames -- deliberately or not, entirely within the
    FIRST arc, before any bounce.

    Segment-growing changepoint detection, not a bounce-specific pixel
    heuristic (no reliable geometric "this is a bounce" signal exists from
    2D observations alone -- a bounce is a physical event in 3D, and this
    function only ever sees its own noisy 2D projection): grows a single-
    segment fit (`_fit_ballistic_core`) frame by frame for as long as the
    fit stays acceptable (`normalized_cost <= accept_threshold`, the SAME
    default `fit_ballistic_trajectory` itself uses). The moment extending
    the window would break an otherwise-good fit, that IS the segment
    boundary -- a bounce happened somewhere in the frames just added.
    Starts a fresh segment (fresh restart-depth sweep, fresh (x0, v0) --
    post-bounce velocity is a genuinely new unknown, not derivable from
    the pre-bounce segment without assuming a restitution coefficient this
    function does not want to guess) from there, repeats until every
    frame is covered or the remaining tail is too short to fit at all.

    Degenerates EXACTLY to `fit_ballistic_trajectory`'s own single-segment
    behavior whenever the true trajectory never actually bounces within
    the observed window -- the growing window just keeps extending to the
    end without ever needing to split, using the identical underlying fit
    (`_fit_ballistic_core`) and the identical default accept_threshold.

    Returns a list of (start_idx, end_idx, x0, v0) segments -- `start_idx`/
    `end_idx` are indices into `pixel_obs`/`times` (Python slice
    convention, end EXCLUSIVE), NOT absolute clip frame numbers -- caller's
    job to translate, matching this module's own established "take
    exactly what's needed" convention (`fit_ballistic_trajectory` takes
    times relative to its own first observation for the identical reason).
    A frame that cannot anchor any acceptable `min_segment_frames`-sized
    window (even alone) is left OUT of every segment -- caller must treat
    those positions as unrecoverable (NaN), the same "no correct thing to
    return" discipline as `fit_ballistic_trajectory`'s own (None, None)
    contract for a single segment that never fit at all."""
    pixel_obs = np.asarray(pixel_obs, dtype=float)
    times = np.asarray(times, dtype=float)
    size_obs = np.asarray(size_obs, dtype=float) if size_obs is not None else None
    n = len(times)
    segments = []
    start = 0
    while start < n:
        if n - start < min_segment_frames:
            break   # not enough frames left to even attempt one more segment
        best_end, best_p = None, None
        end = start + min_segment_frames
        broke_due_to_failure = False
        while end <= n:
            sub_times = times[start:end] - times[start]
            sub_pixels = pixel_obs[start:end]
            sub_sizes = size_obs[start:end] if size_obs is not None else None
            p, cost = _fit_ballistic_core(
                sub_pixels, sub_times, cam_pos, cam_mat, fovy_deg, width, height,
                gravity_z=gravity_z, n_restarts=n_restarts, max_iter=max_iter,
                size_obs=sub_sizes, object_radius=object_radius, size_weight=size_weight)
            if p is not None and cost <= accept_threshold:
                best_end, best_p = end, p
                end += 1   # still acceptable -- try extending one more frame
            else:
                broke_due_to_failure = True
                break   # extending broke the fit -- the boundary is somewhere in [end-1, end)
        if best_end is None:
            # not even the minimum window fit acceptably starting here --
            # this frame can't anchor a segment; try the next one instead
            # of getting permanently stuck.
            start += 1
            continue

        # Refinement: the coarse growing search above can OVERSHOOT the
        # true boundary by a few frames before finally breaking -- a
        # couple of contaminating post-bounce points don't always push
        # cost over accept_threshold immediately (confirmed empirically,
        # not assumed: in the exact case this was built for, normalized
        # cost grew 0.0001 -> 0.0001 -> 0.40 -> 2.40 -> 6.34 per added
        # contaminating frame, comfortably under a 4.0 threshold for two
        # frames before finally breaking it on the third -- silently
        # keeping those two contaminated frames in the accepted segment
        # measurably distorted its own fitted velocity in a direct test).
        # A clean single-arc window's own cost is generically far lower
        # than one with contaminating points mixed in, so re-examine a
        # small trailing range and keep whichever nearby window size has
        # the LOWEST cost, not just the longest one that happened to
        # clear the coarse threshold.
        #
        # ONLY when the search broke due to a genuine failed fit --
        # NOT when it simply ran out of observed frames (`end > n` with
        # every window up to n acceptable, i.e. the whole rest of the
        # clip is one clean arc). A real bug, caught by this function's
        # own opt-in-equivalence test before shipping: refining
        # unconditionally sometimes trimmed a few frames off a perfectly
        # clean single arc purely from restart-optimizer numerical noise
        # (a slightly shorter window randomly landing at a marginally
        # lower cost, with nothing actually contaminating it) -- silently
        # breaking the "piecewise_ballistic=True must degenerate EXACTLY
        # to the single-segment result when there's no real bounce"
        # contract this function's own docstring promises.
        if broke_due_to_failure:
            refine_floor = max(start + min_segment_frames, best_end - 5)
            best_cost = None
            for candidate_end in range(best_end, refine_floor - 1, -1):
                sub_times = times[start:candidate_end] - times[start]
                sub_pixels = pixel_obs[start:candidate_end]
                sub_sizes = size_obs[start:candidate_end] if size_obs is not None else None
                p, cost = _fit_ballistic_core(
                    sub_pixels, sub_times, cam_pos, cam_mat, fovy_deg, width, height,
                    gravity_z=gravity_z, n_restarts=n_restarts, max_iter=max_iter,
                    size_obs=sub_sizes, object_radius=object_radius, size_weight=size_weight)
                if p is not None and (best_cost is None or cost < best_cost):
                    best_end, best_p, best_cost = candidate_end, p, cost

        segments.append((start, best_end, best_p[:3], best_p[3:]))
        if os.environ.get("VITALS_RECON_DEBUG"):
            print(f"DEBUG fit_piecewise_ballistic_trajectory: segment [{start},{best_end}) "
                  f"x0={best_p[:3]} v0={best_p[3:]}")
        start = best_end
    return segments


def reconstruct_trajectory(masks, fps, cam_pos, cam_mat, fovy_deg, width, height,
                            plane_z=None, name="ball", planar_motion=True, reid_events=None, T=None,
                            forgiveness_frames=0, known_occluded_frames=None, plane_pieces=None,
                            ballistic=False, object_radius=None, gravity_z=-9.81, size_weight=3.0,
                            unreliable_prefix_frames=0, piecewise_ballistic=False):
    """masks: {frame_idx: (H,W) bool}, as produced by segmentation.py /
    reidentify.py -- REAL tracked masks, never ground truth (this module
    has no privileged access, matching AGENT.md 3.1's spirit even though it
    isn't itself a scoring-path component).

    reid_events: reidentify.py's ReidentifyResult.reid_events, or None if
    the track never needed re-identification (e.g. no occlusion occurred).
    Determines `present` per the module docstring's existence-vs-visibility
    fix -- do not pass None for a track that went through
    track_with_reidentification; that silently reverts to the
    occluded-means-gone bug this module exists to avoid.

    forgiveness_frames: MUST match the value actually passed to
    track_with_reidentification for these exact masks/reid_events (AGENT.md
    defect #20) -- see _existence_mask's own docstring for what goes wrong
    if it doesn't (a benign, un-searched-for flicker gets misread as an
    unresolved existence violation). Defaults to 0 (old behavior, every
    empty frame needs explicit reid_event closure) only for callers that
    genuinely used forgiveness_frames=0; do not rely on this default
    otherwise.

    known_occluded_frames: reidentify.py's ReidentifyResult.
    known_occluded_frames, or None (AGENT.md M5, existence-statistic
    hardening) -- lets a gap entirely explained by a known occluder count
    as present, same status as ordinary state-space-invisible occlusion.
    See _existence_mask's own docstring for the full reasoning and why a
    pure duration-based alternative was rejected. Omit only for a track
    that never had camera/occluder_bounds/deceleration available at all.

    `pos` is NaN for any frame the object isn't directly visible in,
    whether or not `present` ends up True for that frame (position is
    genuinely unmeasured during a gap; sigma_kinematic already handles NaN
    distance correctly, see module docstring). `present` follows
    `_existence_mask`.

    planar_motion must be passed True explicitly -- a reminder, not just a
    flag, that this reconstruction is only valid for flat-ground OR
    known-fixed-plane scenarios (see module docstring limit 1 and
    `plane_pieces` below). Raises if False, since there is no correct
    thing to return for a scenario whose true depth follows no known
    plane at all.

    plane_z: the flat-ground case (occlusion_corridor and similar) --
    unproject every frame against the single horizontal plane z=plane_z.
    Mutually exclusive with plane_pieces; exactly one of the two must be
    given.

    plane_pieces: the PIECEWISE known-plane case (AGENT.md M4.5 --
    ramp_descent and similar, where the object's true motion is confined
    to a KNOWN plane at every instant, but not the SAME plane throughout,
    e.g. an incline that transitions onto flat ground). A list of
    `(plane_point, plane_normal, valid_fn)` tuples, tried IN ORDER for
    each frame's centroid -- the first piece whose `unproject_to_known_
    plane` result satisfies its own `valid_fn(pos3d) -> bool` wins. This
    is a SELF-CONSISTENCY check, not privileged knowledge of which piece
    is "correct": each piece's own known, fixed geometry (e.g. ramp_
    descent's known toe position, scene geometry -- same status as
    occlusion_corridor's own known wall bounds, AGENT.md defect #17's
    established position) defines where ITS OWN unprojection is
    geometrically plausible, and a frame is assigned to whichever known
    plane its own centroid is consistent with. If NO piece's result
    validates, that frame is left unreconstructed (pos stays NaN), same
    as any other degenerate/invalid unprojection -- not a position to
    silently fabricate from whichever piece happened to be tried last.

    T: total frame count. Defaults to max(masks)+1, but pass explicitly
    when the clip may extend past the last frame with any mask at all
    (e.g. the object never returns and the clip still runs to T_max) --
    otherwise the trajectory is silently truncated at the last sighting.

    unreliable_prefix_frames: force `pos` to NaN for the first N frames,
    unconditionally, AFTER the normal per-frame reconstruction above --
    `present` is untouched (the object IS visible/tracked; its POSITION
    just isn't trustworthy yet). For `plane_z`/`plane_pieces` scenes whose
    own object is genuinely airborne (on no registered plane at all) for a
    KNOWN, fixed initial window -- e.g. `ramp_descent`'s own ball, dropped
    from height for "a small, consistent settling fall" (that scene's own
    comment), briefly in free fall before landing on the ramp. Forcing a
    plane-based unprojection during that window doesn't fail loudly (every
    piece's own `valid_fn` can still spuriously validate an airborne
    point's candidate, since `valid_fn` only checks spatial REGION, not
    whether the point is actually near the plane) -- it silently produces
    a large, wrong position instead (AGENT.md M2.6: measured directly,
    0.90m error at frame 0 on `ramp_descent_high_friction`, decaying to
    <0.001m by frame 10, an almost-exact match to that scene's own
    analytical free-fall time). This is a KNOWN, scene-derived constant
    (computable from drop height and gravity, same "borrow known physics,
    don't re-derive it noisily" discipline as `OCCLUSION_CORRIDOR_
    DECELERATION`) -- 0 (default) is a strict no-op for every scene
    without this issue (occlusion_corridor's own ball starts already
    resting on the floor; `ballistic` mode has no plane assumption to
    violate in the first place).

    A per-frame size-consistency check (reject a plane candidate whose
    implied depth disagrees with the object's own known-radius/apparent-
    size depth cue, `fit_ballistic_trajectory`'s own established signal)
    was tried FIRST and measured to be too noisy for this: even at
    already-settled frames where the plane-based reconstruction is
    near-perfect, the size-implied depth carried its own ~1.0-1.3m
    systematic bias on real rendered data -- comparable in magnitude to
    the ~1-2m of EXTRA discrepancy the airborne window itself produces, a
    weak signal-to-noise ratio for a hard per-frame reject test. A known,
    fixed frame count is a strictly more reliable signal for a
    deterministic, scene-governed phenomenon than a noisy per-frame
    heuristic trying to re-detect it from scratch every episode.

    ballistic: AGENT.md M4.5's "true free 3D motion" case (e.g.
    projectile.xml) -- mutually exclusive with plane_z/plane_pieces (see
    below). Set True for a scenario whose motion follows no known plane at
    all (gravity plus an initial launch, nothing constraining it to a
    surface). Unlike the plane methods, which unproject each frame
    independently, this fits ONE global ballistic trajectory (x0, v0)
    across every observed frame jointly via `fit_ballistic_trajectory`,
    then evaluates that single fit at each observed frame's own time --
    see that function's docstring for why (a single ray has no depth
    information at all; only the trajectory's whole parametric shape,
    fit jointly, resolves it). object_radius (the object's known TRUE
    physical radius, e.g. projectile.xml's ball = 0.15) is REQUIRED when
    ballistic=True -- it drives the apparent-size depth cue that
    `fit_ballistic_trajectory`'s own docstring found necessary to resolve
    a genuine monocular scale ambiguity; there is no size-free fallback
    here; gravity_z/size_weight forward to that function unchanged
    (size_weight default 3.0, chosen by sweeping real rendered data --
    see that function's docstring). If the global fit doesn't converge
    (returns None, None -- a genuine "not recoverable from this view"
    result, not a bug), every frame's pos stays NaN and `present` is still
    computed normally, same discipline as an unresolved plane-unprojection
    gap.

    piecewise_ballistic: OPTIONAL, only meaningful when ballistic=True
    (2026-08, AGENT.md M4.5's own documented gap, fixed: a real bounce is
    a velocity discontinuity the single-segment fit above cannot
    represent at all -- confirmed as the root cause of GATE 2's own 0/42
    ballistic-reconstruction success rate on `projectile`). When True,
    uses `fit_piecewise_ballistic_trajectory` instead -- detects segment
    (bounce) boundaries by how far a single continuous parabola can be
    grown before its own fit degrades, fits each segment independently.
    Default False, and DELIBERATELY not folded into `ballistic=True`'s
    own existing behavior as an unconditional replacement -- this keeps
    every already-validated single-segment result (`tests/test_reconstruct.
    py::test_projectile_ballistic_reconstruction_error_is_bounded`, 4
    seeds, <0.2m bound, first 20 frames -- entirely within one arc) byte-
    identical; opt in per scenario via `scene_geometry.py`'s own
    `reconstruct_kwargs` once a scenario's own observed window is known to
    span real bounces, not silently for every ballistic scenario.
    """
    if not planar_motion:
        raise ValueError("reconstruct_trajectory's ray/plane method is only valid for "
                          "flat-ground or known-fixed-plane motion -- see module docstring. "
                          "Do not call this for a scenario whose true depth follows no known plane.")
    n_modes = sum(x is not None for x in (plane_z, plane_pieces)) + bool(ballistic)
    if n_modes != 1:
        raise ValueError("reconstruct_trajectory needs EXACTLY ONE of plane_z (flat-ground), "
                          "plane_pieces (piecewise known-plane, e.g. ramp_descent), or "
                          "ballistic=True (free 3D motion, e.g. projectile.xml) -- got "
                          f"plane_z={plane_z!r}, plane_pieces={'<given>' if plane_pieces else None!r}, "
                          f"ballistic={ballistic!r}.")
    if ballistic and object_radius is None:
        raise ValueError("ballistic=True requires object_radius (the object's known TRUE physical "
                          "radius) -- see docstring, this drives the apparent-size depth cue "
                          "fit_ballistic_trajectory's own docstring found necessary.")

    if T is None:
        T = max(masks.keys()) + 1 if masks else 0
    reid_events = reid_events or []

    t = np.arange(T) / fps
    pos = np.full((T, 1, 3), np.nan)
    quat = np.tile(IDENTITY_QUAT, (T, 1, 1))

    if ballistic:
        obs_frames, pixel_obs_list, size_obs_list = [], [], []
        for i in range(T):
            m = masks.get(i)
            if m is None or m.sum() == 0:
                continue
            cen = centroid(m)
            if cen is None:
                continue
            obs_frames.append(i)
            pixel_obs_list.append(cen)
            size_obs_list.append(np.sqrt(m.sum() / np.pi))
        if len(obs_frames) >= 3:
            obs_frames = np.array(obs_frames)
            rel_times = (obs_frames - obs_frames[0]) / fps
            g_vec = np.array([0.0, 0.0, gravity_z])
            if piecewise_ballistic:
                segments = fit_piecewise_ballistic_trajectory(
                    np.array(pixel_obs_list), rel_times, cam_pos, cam_mat, fovy_deg, width, height,
                    gravity_z=gravity_z, size_obs=np.array(size_obs_list),
                    object_radius=object_radius, size_weight=size_weight)
                for seg_start, seg_end, x0_fit, v0_fit in segments:
                    seg_t0 = rel_times[seg_start]
                    for local_i in range(seg_start, seg_end):
                        i = obs_frames[local_i]
                        rt = rel_times[local_i] - seg_t0
                        pos[i, 0] = x0_fit + v0_fit * rt + 0.5 * g_vec * rt * rt
            else:
                x0_fit, v0_fit = fit_ballistic_trajectory(
                    np.array(pixel_obs_list), rel_times, cam_pos, cam_mat, fovy_deg, width, height,
                    gravity_z=gravity_z, size_obs=np.array(size_obs_list),
                    object_radius=object_radius, size_weight=size_weight)
                if x0_fit is not None:
                    for i, rt in zip(obs_frames, rel_times):
                        pos[i, 0] = x0_fit + v0_fit * rt + 0.5 * g_vec * rt * rt
    else:
        for i in range(T):
            m = masks.get(i)
            if m is None or m.sum() == 0:
                continue
            cen = centroid(m)
            if cen is None:
                continue
            if plane_pieces is not None:
                world = None
                for plane_point, plane_normal, valid_fn in plane_pieces:
                    candidate = unproject_to_known_plane(cen[0], cen[1], cam_pos, cam_mat, fovy_deg,
                                                          width, height, plane_point, plane_normal)
                    if candidate is not None and valid_fn(candidate):
                        world = candidate
                        break
            else:
                world = unproject_to_plane(cen[0], cen[1], cam_pos, cam_mat, fovy_deg, width, height, plane_z)
            if world is None:
                continue
            pos[i, 0] = world

    if unreliable_prefix_frames:
        # Unconditional, AFTER the loop -- overrides whatever the plane
        # unprojection above computed for these frames, not a filter on
        # whether to attempt it (see this parameter's own docstring: the
        # unprojection can SPURIOUSLY validate an airborne point without
        # any error, so skipping it isn't an option -- it must be
        # overridden). present is untouched.
        pos[:unreliable_prefix_frames, 0] = np.nan

    present = _existence_mask(masks, reid_events, T, forgiveness_frames=forgiveness_frames,
                               known_occluded_frames=known_occluded_frames)[:, None]
    return Trajectory(t=t, pos=pos, quat=quat, present=present, names=[name])


def merge_trajectories(trajs):
    """Stack single-object Trajectories (as `reconstruct_trajectory` always
    returns) into one K-object Trajectory, concatenated along the object
    axis in the given order -- object 0 in the merged result is `trajs[0]`,
    object 1 is `trajs[1]`, etc.

    AGENT.md M5.6 (scoped phase-2 multi-object support): rather than
    teaching `reconstruct_trajectory` itself to be natively multi-object,
    each tracked object (primary, with full re-identification; secondary,
    with none -- see `track_with_reidentification`'s own docstring) is
    reconstructed independently through the EXISTING single-object path
    unmodified, then merged here. Smaller diff, and keeps each object's
    reconstruction independently testable exactly as it already is.

    All inputs must share the same `t` (same fps, same T) -- they came from
    the same clip, just different tracked objects, so this is a real
    invariant, not a defensive check for a case that can't happen; raises
    if violated rather than silently reindexing."""
    if len(trajs) < 2:
        raise ValueError("merge_trajectories needs at least 2 trajectories -- for a single "
                          "object, just use reconstruct_trajectory's own output directly.")
    t0 = trajs[0].t
    for traj in trajs[1:]:
        if not np.array_equal(traj.t, t0):
            raise ValueError("merge_trajectories requires identical t across all inputs -- "
                              "got mismatched T/fps, which means these didn't come from the same clip.")
    pos = np.concatenate([traj.pos for traj in trajs], axis=1)
    quat = np.concatenate([traj.quat for traj in trajs], axis=1)
    present = np.concatenate([traj.present for traj in trajs], axis=1)
    names = [nm for traj in trajs for nm in traj.names]
    return Trajectory(t=t0, pos=pos, quat=quat, present=present, names=names)
