"""Physics-prior re-identification: predicts an occluded object's pixel
position at reappearance by extrapolating a kinematic fit over its own
recent SAM2-tracked (not ground-truth) pre-occlusion centroids, then gates
candidate proposals by proximity to that prediction. Complementary to
reidentify.py's DINO appearance matching, not a replacement for it -- see
"How this composes with DINO" below.

Why this exists: DINO appearance matching, even after calibration
improvements, measured 0/N matches on occlusion_corridor's hard velocity
band (v0=4.5-4.9) -- root-caused to genuinely low appearance separability
for a low-texture, rotating, rolling object (see reidentify.py's own
history / AGENT.md). Validated directly (scripts referenced in AGENT.md
"Known defects"): kinematic extrapolation predicts the true reappearance
pixel position to within 2.7-3.9px on those exact hard cases, and the SAM2
candidate nearest that prediction is the true ball (IoU 0.55-0.75) in
10/10 tested episodes -- a clean win precisely where DINO fails, because
this scenario's motion is far more PREDICTABLE than the object's
appearance is DISTINGUISHABLE.

Pure numpy, no torch/CUDA -- this module only fits curves to centroids
already produced by segmentation.py's tracker; it never touches pixels or
models directly. Importable and testable without a GPU.

Two honest scope limits, not yet handled, on the actual TODO list before
this generalizes past synthetic MuJoCo scenes:

1. Static camera + near-constant depth ASSUMED, not detected or corrected
   for. A true constant 3D deceleration only projects to something close
   to constant-deceleration pixel motion when the object's distance to
   camera stays roughly fixed (true here: occlusion_corridor's camera
   views the corridor side-on, so depth barely changes as the ball travels
   along x). If the object moves substantially toward/away from the
   camera, apparent pixel velocity gets warped nonlinearly by 1/depth, and
   this pixel-space fit would need real camera intrinsics + a depth
   estimate (monocular depth, since single-camera video has no depth for
   free) to convert back to metric 3D before extrapolating -- exactly the
   "3D positions: multi-view triangulation, else monocular depth" row
   already planned for Phi (AGENT.md Phi component table), not yet built.
2. Camera assumed static across the gap. A moving camera (robot
   ego-motion, handheld, drone) would need visual odometry/SLAM to
   separate camera motion from object motion before any of this pixel-
   space extrapolation is meaningful. Not built speculatively -- no
   scenario in this repo has a moving camera yet.

Neither of these is a §3.1 violation to fix later (knowing your OWN
camera's calibration is normal instrumentation, not privileged access to
the TARGET's state -- see AGENT.md's note on this). They're just real
engineering the current synthetic-only, static-side-on-camera scenarios
don't yet require. Do not claim results here generalize to a real,
uncalibrated, or moving camera until they're built and validated.

How this composes with DINO: physics-prior gating is PRIMARY for today's
K=1 scenarios -- position alone unambiguously identifies "the object" when
there's only one object. DINO appearance similarity becomes the relevant
signal only once a scenario has K>1 objects that could plausibly occupy
similar predicted positions (not yet built) -- reidentify.py's DINO path
is kept for that future case and as a fallback when a track has too few
pre-occlusion frames to fit a physics prior at all (see MIN_FIT_FRAMES).
"""
from __future__ import annotations
import numpy as np

MIN_FIT_FRAMES = 5          # fewer visible pre-occlusion frames than this -> no reliable fit
MIN_AXIS_VARIANCE = 1.0     # px^2; below this, treat the axis as static rather than fit a
                             # degenerate quadratic to near-zero-variance noise (same principle
                             # as detect/statistics.py's sigma_kinematic spread floor)


def centroid(mask):
    """(px, py) of a boolean mask's True pixels, or None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _fit_axis(t, v, min_r_squared=0.9):
    """Fits v(t) = c0 + c1*t + c2*t^2 by least squares. Returns (coeffs,
    r_squared) if the axis has enough real variance to fit meaningfully and
    the fit clears min_r_squared, else None -- a caller should fall back to
    "last observed value" for a None axis, not force a shaky quadratic
    through near-constant data (mirrors physical_constants.py's own
    r_squared gate and its documented reason: don't return a fit nobody
    should trust)."""
    if np.var(v) < MIN_AXIS_VARIANCE:
        return None
    coeffs = np.polyfit(t, v, 2)
    fitted = np.polyval(coeffs, t)
    ss_res = np.sum((v - fitted) ** 2)
    ss_tot = np.sum((v - v.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if not (r_squared >= min_r_squared):
        return None
    return coeffs, r_squared


class PixelTrackFit:
    """Independent per-axis fits of a tracked object's pixel centroid over
    time. `predict(t)` extrapolates to an arbitrary future time -- a static
    axis (fit is None) predicts its last observed value; a fitted axis
    extrapolates its quadratic, CLAMPED at the fit's own implied stopping
    time (see `_predict_axis`). `t` is measured in the same units passed to
    fit_pixel_track (seconds since track start, matching Trajectory.dt
    convention elsewhere in this repo)."""

    def __init__(self, x_fit, y_fit, last_t, last_pos):
        self.x_fit, self.y_fit = x_fit, y_fit
        self.last_t, self.last_pos = last_t, last_pos

    def predict(self, t):
        px = self._predict_axis(self.x_fit, self.last_pos[0], t)
        py = self._predict_axis(self.y_fit, self.last_pos[1], t)
        return float(px), float(py)

    @staticmethod
    def _predict_axis(fit, last_val, t):
        """A quadratic fit correctly models CONSTANT DECELERATION near its
        own fit window, but has no notion that a real decelerating object
        actually stops -- extrapolated far enough, the same polynomial's
        implied velocity crosses zero and then REVERSES, curving the
        predicted position back the way it came. Found the hard way
        (AGENT.md defect #16): with a ~23-frame occlusion gap here (a
        search that must extrapolate an unknown distance ahead, unlike the
        original candidate-check validation which always extrapolated a
        short, KNOWN distance to the true reappearance frame), the raw
        polynomial prediction traced out a full parabola -- decreasing,
        reaching a minimum, then increasing again -- drifting the search
        AWAY from where the object actually came to rest by the time it
        was checked against real candidates.

        Fix: clamp `t` at the fit's own implied stopping time `t_stop`
        (velocity = c1 + 2*c2*t = 0), evaluating the polynomial there
        instead of at the raw requested `t` beyond that point -- a real
        object's position stays constant once stopped, which is exactly
        what this clamp encodes, still using only quantities the fit
        itself already estimated (no new free parameter)."""
        if fit is None:
            return last_val
        coeffs, _r_squared = fit
        c2, c1, _c0 = coeffs
        if abs(c2) > 1e-9:
            t_stop = -c1 / (2 * c2)
            if 0 < t_stop < t:
                t = t_stop
        return float(np.polyval(coeffs, t))


def fit_pixel_track(masks, frame_indices, dt, min_frames=MIN_FIT_FRAMES, min_r_squared=0.9):
    """masks: {frame_idx: (H,W) bool}, as produced by segmentation.py's
    track_video/track_with_reidentification -- REAL tracked masks, never
    ground truth (this module has no privileged access, matching AGENT.md
    3.1's spirit even though it isn't itself a scoring-path component).
    frame_indices: the consecutive pre-occlusion frames to fit over (the
    caller's job to choose -- typically the last min_frames-or-more visible
    frames immediately before the gap, matching physical_constants.py's
    "the window is the caller's job" separation of concerns).

    Returns a PixelTrackFit, or None if fewer than min_frames frames have a
    non-empty mask (too little history to trust any extrapolation --
    falling back to DINO-only re-id, or no re-id, is the honest move here,
    not forcing a fit through 2 points)."""
    pts = [(i * dt, centroid(masks[i])) for i in frame_indices if i in masks]
    pts = [(t, c) for t, c in pts if c is not None]
    if len(pts) < min_frames:
        return None

    t = np.array([p[0] for p in pts])
    xs = np.array([p[1][0] for p in pts])
    ys = np.array([p[1][1] for p in pts])
    t0 = t[-1]
    t_rel = t - t0   # fit relative to the last observed frame, so predict()'s
                     # input is naturally "seconds since last-seen", matching
                     # how a caller will invoke it (dt to the search frame)

    x_fit = _fit_axis(t_rel, xs, min_r_squared)
    y_fit = _fit_axis(t_rel, ys, min_r_squared)
    return PixelTrackFit(x_fit, y_fit, last_t=t0, last_pos=(float(xs[-1]), float(ys[-1])))


class MetricTrackFit:
    """Metric-space position/velocity at the last pre-occlusion instant
    (from `fit_metric_track`), extrapolated using a CALIBRATED deceleration
    constant -- not re-derived per-episode -- clamped at the implied
    stopping time, then projected back to pixel space via
    `reconstruct.project_to_pixel` for candidate gating.

    This is AGENT.md defect #16's actual fix, not `PixelTrackFit`'s clamp
    alone. The clamp (predict.py's `_predict_axis`) was necessary but not
    sufficient: it correctly stops a quadratic from reversing direction,
    but the STOP POSITION it clamps to is only as good as the per-episode
    quadratic's own re-derived deceleration, fit from a short window of
    real, noisy SAM2-tracked pixels. Measured directly on a real gap: that
    per-episode fit implied the object stops far short of where it
    actually was still travelling to (predicted stop at pixel x=107.9;
    true position at frame 61 was x=68.3 and STILL DECREASING through
    frame 99) -- because the object's true velocity there (~4.5-4.9 m/s)
    needs roughly `v0/a ~ 3.6s ~ 108 frames` to actually stop, at this
    scenario's own calibrated deceleration (`physical_constants.py`,
    a~1.246 m/s^2, CV=0.12%) -- far longer than the short pre-drop window
    available to fit from, so any per-episode re-derivation of a is poorly
    conditioned by construction. Borrowing the already-precisely-calibrated
    constant instead sidesteps that -- the only thing still fit per-episode
    is (x0, v0) at the drop instant, which IS well-conditioned over a short
    window (position and velocity are directly observable; deceleration
    is a second-derivative quantity and much more sensitive to noise)."""

    def __init__(self, x0, v0, lateral_value, motion_axis, lateral_axis, deceleration,
                 cam_pos, cam_mat, fovy_deg, width, height, plane_z, occluder_bounds=None,
                 t0_abs=0.0):
        self.x0, self.v0 = x0, v0
        self.lateral_value = lateral_value
        self.motion_axis, self.lateral_axis = motion_axis, lateral_axis
        self.deceleration = deceleration
        self.cam_pos, self.cam_mat = cam_pos, cam_mat
        self.fovy_deg, self.width, self.height, self.plane_z = fovy_deg, width, height, plane_z
        self.occluder_bounds = occluder_bounds
        # ABSOLUTE clip time of the last pre-occlusion frame -- only needed
        # for the dynamic-occluder case (AGENT.md M5.7): `t` everywhere else
        # in this class is relative to that frame (same convention as
        # PixelTrackFit.predict), but a tracked occluder's own position is
        # indexed by ABSOLUTE frame time, so resolving a callable
        # occluder_bounds needs t0_abs + t, not t alone. Unused (default 0)
        # for the static-tuple/None cases, which don't depend on absolute
        # time at all.
        self.t0_abs = t0_abs

    def _x_motion(self, t):
        a = self.deceleration
        if a > 1e-9 and abs(self.v0) > 1e-9:
            t_stop = abs(self.v0) / a
        else:
            t_stop = float("inf")
        t_eff = min(t, t_stop)
        sign = 1.0 if self.v0 >= 0 else -1.0
        return self.x0 + self.v0 * t_eff - sign * 0.5 * a * t_eff ** 2

    def _occluder_box_at(self, t):
        """Resolves `self.occluder_bounds` into a concrete
        (x_lo, x_hi, y_lo, y_hi) box for THIS instant, or None if there is
        no known occluder right now. Three accepted forms, dispatched by
        type:
          - None: no occluder at all (unchanged from before M5.7).
          - a static (lo, hi) 2-tuple: occlusion_corridor's ORIGINAL,
            permanently-fixed-wall convention -- 1D, along motion_axis
            only. Kept working exactly as before by treating it as
            unbounded in the lateral axis: a permanently-fixed wall was
            always known to span the object's whole lane for the life of
            the clip, so no lateral check was ever needed for it, and
            adding one now would silently change old behavior.
          - a callable: AGENT.md M5.7's DYNAMIC case, built by
            `occluder_bounds_from_track` from a TRACKED second object's
            own masks -- called with ABSOLUTE clip time (t0_abs + t) and
            returns a 4-tuple, or None for "the occluder's own position
            isn't currently measured" (the honest fallback M5.7's own
            scoping note requires, not a fabricated last-known position
            held indefinitely)."""
        if self.occluder_bounds is None:
            return None
        if callable(self.occluder_bounds):
            return self.occluder_bounds(self.t0_abs + t)
        lo, hi = self.occluder_bounds
        return (lo, hi, -np.inf, np.inf)

    def occluded_at(self, t):
        """True iff the extrapolated position at time t falls inside the
        occluder's current box (both axes -- see `_occluder_box_at`) --
        positive evidence the object is still hidden, distinct from
        predict_pixel returning None for other reasons (e.g. behind the
        camera). track_with_reidentification uses this to suppress DINO's
        appearance-only fallback too during a window physics already knows
        is still occluded (AGENT.md defect #19): predict_pixel alone
        collapses "definitely still hidden" and "no opinion" into the same
        None, which is fine for tier 1 (it just skips offering a target)
        but left tier 2 free to search anyway with no awareness that the
        object provably can't be visible yet."""
        box = self._occluder_box_at(t)
        if box is None:
            return False
        x_lo, x_hi, y_lo, y_hi = box
        return (x_lo <= self._x_motion(t) <= x_hi) and (y_lo <= self.lateral_value <= y_hi)

    def predict_pixel(self, t):
        """t: seconds since the last pre-occlusion frame (same convention
        as PixelTrackFit.predict). Returns (px, py), or None if EITHER the
        extrapolated point projects behind the camera (shouldn't normally
        happen for this scenario's geometry, but not silently fabricated if
        it does), OR the predicted metric position still falls inside the
        occluder's current box on BOTH axes (AGENT.md defect #17, extended
        to 2D by M5.7): an accurate prediction that is still genuinely
        behind a KNOWN occluder should not be offered as a search target at
        all, not merely gated on distance to whatever candidate happens to
        be nearest. Found the hard way: once `fit_metric_track`'s
        predictions became accurate (defect #16's fix), the search started
        reliably walking toward the object's true (but still occluded)
        position -- and a static visual feature ON the occluder itself (an
        edge, shadow, or texture patch) sitting directly on that path got
        accepted as a match before the real object had actually cleared it.
        `occluder_bounds` is scene geometry (the wall's/occluder's own
        known extent) plus, in the dynamic case, a TRACKED position -- never
        privileged access to the OBJECT's own state, the same status this
        repo already gives to knowing your own camera calibration (AGENT.md
        T3's note)."""
        x_motion = self._x_motion(t)

        box = self._occluder_box_at(t)
        if box is not None:
            x_lo, x_hi, y_lo, y_hi = box
            if (x_lo <= x_motion <= x_hi) and (y_lo <= self.lateral_value <= y_hi):
                return None

        pos3d = np.zeros(3)
        pos3d[self.motion_axis] = x_motion
        pos3d[self.lateral_axis] = self.lateral_value
        pos3d[2] = self.plane_z   # always exactly plane_z -- structural, not fit (see fit_metric_track)

        from . import reconstruct as recon
        return recon.project_to_pixel(pos3d, self.cam_pos, self.cam_mat, self.fovy_deg, self.width, self.height)


def fit_metric_track(masks, frame_indices, dt, cam_pos, cam_mat, fovy_deg, width, height, plane_z,
                      deceleration, motion_axis=0, min_frames=MIN_FIT_FRAMES, occluder_bounds=None):
    """Unprojects each frame's pixel centroid to metric 3D via
    `reconstruct.unproject_to_plane` -- the same flat-ground ray/plane
    method `reconstruct.py` uses for Trajectory reconstruction, so this
    inherits its documented limitation too: valid only for planar motion
    (occlusion_corridor), not ramp_descent. Fits ONLY (x0, v0) along
    `motion_axis` (a world coordinate index, default 0) via linear least-
    squares over the window -- position and velocity, deliberately NOT
    deceleration; see MetricTrackFit's docstring for why re-deriving that
    per-episode is exactly what defect #16 found unreliable. The other
    horizontal axis ("lateral_axis", `1 - motion_axis` for a single
    ground-plane's x/y) is fixed at its own observed mean over the same
    window -- occlusion_corridor's `perturb_mode="velocity_x_only"` means
    it should carry near-zero real variance, but it's still measured, not
    assumed to be exactly 0.

    v0 from this linear fit is the window's AVERAGE velocity, not the
    instantaneous velocity exactly at the last frame -- a real, minor
    simplification (the object is decelerating even within this short
    window, so true instantaneous v0 is very slightly higher in magnitude
    than this average). Judged small relative to the error this function
    exists to fix; revisit with a recency-weighted fit if it turns out to
    matter.

    occluder_bounds: OPTIONAL, either a static (lo, hi) tuple along
    `motion_axis` (e.g. occlusion_corridor's fixed wall -- see
    `scene_geometry.OCCLUSION_CORRIDOR_WALL_BOUNDS` for the canonical,
    EMPIRICALLY-CORRECTED value and its own derivation; do not requote a
    number here, this docstring's own example value drifted from that one
    for a while before the two were consolidated, AGENT.md M6)
    or a callable built by `occluder_bounds_from_track` (AGENT.md M5.7's
    dynamic, tracked-occluder case). Passed straight through to
    MetricTrackFit; see its own `_occluder_box_at` docstring for how the
    two forms differ, and AGENT.md defect #17 for why this exists at all.
    Known scene geometry (plus, in the dynamic case, a TRACKED position),
    never privileged object state -- the caller's job to supply for
    scenarios that have a relevant known occluder; omit for scenarios
    without one.

    Returns a MetricTrackFit, or None if fewer than min_frames frames
    unproject successfully (too little history, or a degenerate camera
    ray -- caller should fall back to DINO-only re-id or no re-id, the
    same honest-refusal convention as fit_pixel_track)."""
    from . import reconstruct as recon

    lateral_axis = 1 - motion_axis if motion_axis in (0, 1) else 1
    pts = []
    for i in frame_indices:
        if i not in masks:
            continue
        cen = centroid(masks[i])
        if cen is None:
            continue
        pos3d = recon.unproject_to_plane(cen[0], cen[1], cam_pos, cam_mat, fovy_deg, width, height, plane_z)
        if pos3d is None:
            continue
        pts.append((i * dt, pos3d[motion_axis], pos3d[lateral_axis]))
    if len(pts) < min_frames:
        return None

    t = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts])
    lateral = np.array([p[2] for p in pts])
    t0 = t[-1]
    t_rel = t - t0   # same convention as fit_pixel_track: relative to the last observed frame

    A = np.stack([np.ones_like(t_rel), t_rel], axis=1)
    (x0, v0), *_ = np.linalg.lstsq(A, x, rcond=None)

    return MetricTrackFit(x0=float(x0), v0=float(v0), lateral_value=float(lateral.mean()),
                           motion_axis=motion_axis, lateral_axis=lateral_axis, deceleration=deceleration,
                           cam_pos=cam_pos, cam_mat=cam_mat, fovy_deg=fovy_deg,
                           width=width, height=height, plane_z=plane_z, occluder_bounds=occluder_bounds,
                           t0_abs=float(t0))


def occluder_bounds_from_track(masks_secondary, dt, cam_pos, cam_mat, fovy_deg, width, height,
                                occluder_plane_z, x_half_width, y_half_width, max_staleness_frames=3):
    """Builds a per-instant occluder-bounds lookup for MetricTrackFit's
    DYNAMIC-occluder case (AGENT.md M5.7) from a TRACKED second object's own
    masks -- never privileged ground-truth position, same status as this
    module's own ball-position unprojection everywhere else. Returns a
    callable `t_abs -> (x_lo, x_hi, y_lo, y_hi) or None`.

    occluder_plane_z, x_half_width, y_half_width: the occluder's own KNOWN,
    FIXED box half-extents and center height -- e.g.
    occlusion_corridor_moving.xml's occluder geom, size (0.75, 0.6, 0.5) at
    center z=0.5, so occluder_plane_z=0.5, x_half_width=0.75,
    y_half_width=0.6. Only its CURRENT (x, y) center comes from tracking;
    its size is scene geometry, exactly as legitimate to know in advance as
    occlusion_corridor's ORIGINAL static wall's own extent already was
    (fit_metric_track's own occluder_bounds docstring) -- a real occluder
    doesn't change size as it moves.

    max_staleness_frames: `track_with_reidentification`'s shared propagation
    loop tracks the primary and secondary objects frame-by-frame in lockstep
    (AGENT.md M5.6) -- by construction, a search triggered for the primary
    object at frame `next_idx` runs BEFORE `next_idx` itself has been
    propagated, so `masks_secondary[next_idx]` never exists yet at decision
    time (confirmed by reading the loop, not assumed). Refusing to answer
    at all in that case would make the dynamic case permanently blind
    exactly when it matters most. Instead, this looks BACKWARD up to
    `max_staleness_frames` frames for the most recent measured position --
    the occluder moves slowly relative to the frame rate (this scene's own
    nominal 0.7 m/s is ~0.023m/frame at 30fps), so a 1-3 frame-old position
    is a faithful stand-in, not a fabrication. Returns None (the honest
    "not currently measured" signal M5.7's own scoping note requires) only
    once NOTHING within that window is available -- e.g. the secondary
    track has itself been lost, which per M5.6's own scope has no recovery
    machinery and can go permanently unmeasured."""
    from . import reconstruct as recon

    def bounds_at(t_abs):
        target = int(round(t_abs / dt))
        for frame_idx in range(target, target - max_staleness_frames - 1, -1):
            if frame_idx < 0:
                break
            m = masks_secondary.get(frame_idx)
            if m is None or m.sum() == 0:
                continue
            cen = centroid(m)
            if cen is None:
                continue
            pos3d = recon.unproject_to_plane(cen[0], cen[1], cam_pos, cam_mat, fovy_deg,
                                              width, height, occluder_plane_z)
            if pos3d is None:
                continue
            cx, cy = pos3d[0], pos3d[1]
            return (cx - x_half_width, cx + x_half_width, cy - y_half_width, cy + y_half_width)
        return None

    return bounds_at


def score_candidates_by_position(candidates, predicted_pos):
    """Ranks SAM2 automatic-mask-generator candidates by pixel distance
    from predicted_pos. Returns a list of (distance, candidate) sorted
    ascending; candidates with an empty mask are dropped.

    Distance alone -- NOT a hard accept/reject radius; that's
    `mask_diagonal_px` + the caller's own gate (see reidentify.py). This
    function's earlier docstring claimed "the nearest candidate is
    reliable without needing a tuned gate radius" based on validation that
    always ran AT the true, known reappearance frame. A full end-to-end
    run (AGENT.md build-sequence step 27) found that claim false the
    moment it's exercised honestly: search fires as soon as
    `empty_streak > forgiveness_frames`, almost immediately after a loss
    and long before a real occlusion resolves, so there is often no true
    candidate near the prediction AT ALL -- yet this function still
    returns a "nearest" one, and an unconditional caller will wrongly
    accept it. Measured directly: a false match at 23px distance, locked
    onto for the rest of a 150-frame clip (post-reacquisition IoU ~0.002).
    Callers MUST gate on distance now -- see `mask_diagonal_px` below."""
    scored = []
    for c in candidates:
        cen = centroid(c["segmentation"])
        if cen is None:
            continue
        dist = float(np.hypot(cen[0] - predicted_pos[0], cen[1] - predicted_pos[1]))
        scored.append((dist, c))
    scored.sort(key=lambda pair: pair[0])
    return scored


def mask_diagonal_px(mask):
    """Bounding-box diagonal of a mask's True pixels, in pixels -- a
    physically-grounded scale for gating candidate acceptance (see
    score_candidates_by_position's docstring), tied to the object's own
    last-known apparent size rather than an arbitrary pixel constant.
    None if the mask is empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
