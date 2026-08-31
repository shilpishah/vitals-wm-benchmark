"""phi/motion_prior.py -- pure numpy, no GPU/SAM2/torch required, so this
runs unconditionally (unlike test_gates.py/test_physical_constants.py,
which need mujoco). Exercises the fitting + prediction + candidate-gating
logic against synthetic tracked-centroid data standing in for SAM2 output,
matching the kinematic shape (near-constant deceleration in one axis,
static in the other) validated on real occlusion_corridor frames.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from vitals.phi.motion_prior import (fit_pixel_track, score_candidates_by_position, centroid,
                                      fit_metric_track, MetricTrackFit, occluder_bounds_from_track)
from vitals.phi.reconstruct import project_to_pixel


def _decelerating_blob_masks(n_frames, dt, x0=100.0, v0=50.0, a=10.0, y=80, shape=(200, 200)):
    masks = {}
    for i in range(n_frames):
        t = i * dt
        cx = int(round(x0 + v0 * t - 0.5 * a * t ** 2))
        m = np.zeros(shape, bool)
        m[y - 1:y + 2, cx - 1:cx + 2] = True
        masks[i] = m
    return masks


def test_fit_predicts_decelerating_axis_and_treats_static_axis_as_constant():
    dt = 0.1
    masks = _decelerating_blob_masks(8, dt, x0=100.0, v0=50.0, a=20.0, y=80)
    fit = fit_pixel_track(masks, list(range(8)), dt)
    assert fit is not None
    assert fit.y_fit is None, "a genuinely static axis should not get a forced quadratic fit"
    assert fit.x_fit is not None
    assert fit.x_fit[1] > 0.99, f"synthetic noiseless quadratic should fit almost exactly, got R^2={fit.x_fit[1]}"

    # predict a further 0.3s past the last observed frame (t=0.7 relative to start)
    px, py = fit.predict(0.3)
    t_true = 0.7 + 0.3
    x_true = 100.0 + 50.0 * t_true - 0.5 * 20.0 * t_true ** 2
    assert abs(px - x_true) < 2.0, f"prediction should track the true quadratic closely, got {px} vs {x_true}"
    assert py == 80.0


def test_fit_returns_none_with_too_few_frames():
    masks = _decelerating_blob_masks(3, 0.1)
    fit = fit_pixel_track(masks, list(range(3)), 0.1)
    assert fit is None, "too little pre-occlusion history should refuse to fit, not force one"


def test_candidate_gating_prefers_the_near_candidate():
    dt = 0.1
    masks = _decelerating_blob_masks(8, dt, x0=100.0, v0=50.0, a=20.0, y=80)
    fit = fit_pixel_track(masks, list(range(8)), dt)
    pred = fit.predict(0.3)

    near = np.zeros((200, 200), bool)
    near[int(pred[1]) - 1:int(pred[1]) + 2, int(pred[0]) - 1:int(pred[0]) + 2] = True
    far = np.zeros((200, 200), bool)
    far[10:13, 10:13] = True

    scored = score_candidates_by_position([{"segmentation": far}, {"segmentation": near}], pred)
    assert scored[0][1]["segmentation"] is near, "the near candidate must rank first regardless of input order"
    assert scored[0][0] < 2.0
    assert scored[1][0] > 100.0


def test_centroid_of_empty_mask_is_none():
    assert centroid(np.zeros((10, 10), bool)) is None


def _synthetic_camera():
    """A simple, valid (orthonormal) camera looking down the -y axis at a
    flat ground plane -- not meant to match occlusion_corridor's actual
    camera exactly, just a legitimate, checkable stand-in for testing the
    projection math independently of any real scene."""
    cam_pos = np.array([3.0, 9.0, 2.0])
    forward = np.array([0.0, -1.0, -0.2]); forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.2, -1.0]); up /= np.linalg.norm(up)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    cam_mat = np.stack([right, up, -forward], axis=1)
    return dict(cam_pos=cam_pos, cam_mat=cam_mat, fovy_deg=45.0, width=320, height=240)


def test_metric_track_fit_recovers_known_position_and_velocity():
    """fit_metric_track should recover (x0, v0) essentially exactly from
    noiseless synthetic data -- the actual fitting step being validated,
    independent of the extrapolation/clamping logic tested next."""
    cam = _synthetic_camera()
    dt = 1.0 / 30
    plane_z = 0.15
    x0_true, v0_true = 3.0, -4.7   # metric position/velocity at t=0

    frame_indices = list(range(10))
    masks = {}
    for i in frame_indices:
        t = i * dt
        x = x0_true + v0_true * t   # no deceleration yet within this short window -- tests
                                     # the LINEAR (x0, v0) fit alone, matching what
                                     # fit_metric_track actually fits
        pos3d = np.array([x, 0.0, plane_z])
        px, py = project_to_pixel(pos3d, cam["cam_pos"], cam["cam_mat"], cam["fovy_deg"], cam["width"], cam["height"])
        m = np.zeros((cam["height"], cam["width"]), bool)
        cy, cx = int(round(py)), int(round(px))
        m[cy - 1:cy + 2, cx - 1:cx + 2] = True
        masks[i] = m

    fit = fit_metric_track(masks, frame_indices, dt, deceleration=1.246, motion_axis=0, **cam, plane_z=plane_z)
    assert fit is not None
    t0 = frame_indices[-1] * dt
    x0_at_t0_true = x0_true + v0_true * t0
    assert abs(fit.x0 - x0_at_t0_true) < 0.05, f"recovered x0={fit.x0}, expected~{x0_at_t0_true}"
    assert abs(fit.v0 - v0_true) < 0.05, f"recovered v0={fit.v0}, expected~{v0_true}"
    # not < 0.01 -- single-pixel mask quantization (+/-0.5px) under this synthetic
    # camera's fairly shallow viewing angle amplifies into real metric error on the
    # ray/plane-unconstrained axis, the same camera-angle sensitivity already
    # characterized directly elsewhere in this project (reconstruct.py's own
    # elevation=-10 vs -45 finding) -- not a bug in the fit itself
    assert abs(fit.lateral_value) < 0.15, f"lateral_value={fit.lateral_value}, expected near 0"


def test_metric_prediction_matches_analytic_stop_position_far_past_extrapolation():
    """AGENT.md defect #16's actual fix, end to end: predicting far past
    the implied stop time should match the closed-form analytic stopping
    position -- NOT drift away from it the further into the future you
    predict, unlike the old pure pixel-quadratic approach this replaces."""
    cam = _synthetic_camera()
    dt = 1.0 / 30
    plane_z = 0.15
    x0_true, v0_true, a_true = 3.0, -4.7, 1.246

    frame_indices = list(range(10))
    masks = {}
    for i in frame_indices:
        t = i * dt
        x = x0_true + v0_true * t   # short window, effectively still-linear (matches how
                                     # fit_metric_track only ever fits a linear model)
        pos3d = np.array([x, 0.0, plane_z])
        px, py = project_to_pixel(pos3d, cam["cam_pos"], cam["cam_mat"], cam["fovy_deg"], cam["width"], cam["height"])
        m = np.zeros((cam["height"], cam["width"]), bool)
        cy, cx = int(round(py)), int(round(px))
        m[cy - 1:cy + 2, cx - 1:cx + 2] = True
        masks[i] = m

    fit = fit_metric_track(masks, frame_indices, dt, deceleration=a_true, motion_axis=0, **cam, plane_z=plane_z)
    assert fit is not None

    t0 = frame_indices[-1] * dt
    v0_at_t0 = v0_true  # still-linear window above -> velocity hasn't changed from v0_true
    t_stop = abs(v0_at_t0) / a_true
    x_stop_true = (x0_true + v0_true * t0) + v0_at_t0 * t_stop - np.sign(v0_at_t0) * 0.5 * a_true * t_stop ** 2
    expected_pixel = project_to_pixel(np.array([x_stop_true, 0.0, plane_z]),
                                       cam["cam_pos"], cam["cam_mat"], cam["fovy_deg"], cam["width"], cam["height"])

    # predict at increasingly large t past t_stop -- the absolute distance
    # budget (~2px) reflects the SAME pixel-quantization/camera-angle noise
    # already seen in test_metric_track_fit_recovers_known_position_and_
    # velocity, not fresh slack; what actually matters here is that it does
    # NOT grow further with `extra` (that's exactly what PixelTrackFit's
    # raw quadratic failed to do, per defect #16)
    dists = []
    for extra in [0.0, 1.0, 5.0, 20.0]:
        px, py = fit.predict_pixel(t_stop + extra)
        dist = np.hypot(px - expected_pixel[0], py - expected_pixel[1])
        assert dist < 2.0, f"predict_pixel(t_stop+{extra}) drifted {dist:.2f}px from the analytic stop position"
        dists.append(dist)
    assert max(dists) - min(dists) < 0.01, (
        f"distance to the stop position should be IDENTICAL regardless of how far past "
        f"t_stop is requested (clamped), got {dists} -- looks like it's still drifting")


def test_metric_prediction_refuses_a_match_inside_known_occluder_bounds():
    """AGENT.md defect #17: an ACCURATE prediction that is still genuinely
    behind a known occluder should refuse to offer a search target at all
    -- not merely trust the distance gate downstream. Without this, once
    the prediction became accurate (defect #16's fix), the search started
    reliably walking toward the object's true but still-occluded position,
    and a static feature ON the occluder itself (directly on that path,
    since the occluder was placed there on purpose) could get accepted
    before the real object actually cleared it."""
    cam = _synthetic_camera()
    dt = 1.0 / 30
    plane_z = 0.15
    x0_true, v0_true, a_true = 3.0, -4.7, 1.246

    frame_indices = list(range(10))
    masks = {}
    for i in frame_indices:
        t = i * dt
        x = x0_true + v0_true * t
        pos3d = np.array([x, 0.0, plane_z])
        px, py = project_to_pixel(pos3d, cam["cam_pos"], cam["cam_mat"], cam["fovy_deg"], cam["width"], cam["height"])
        m = np.zeros((cam["height"], cam["width"]), bool)
        cy, cx = int(round(py)), int(round(px))
        m[cy - 1:cy + 2, cx - 1:cx + 2] = True
        masks[i] = m

    t0 = frame_indices[-1] * dt
    x0_at_t0 = x0_true + v0_true * t0   # ~1.43, moving toward negative x

    # occluder spanning the object's own path just ahead of where it is at t0
    occluder_bounds = (x0_at_t0 - 1.0, x0_at_t0 - 0.3)

    fit = fit_metric_track(masks, frame_indices, dt, deceleration=a_true, motion_axis=0,
                            occluder_bounds=occluder_bounds, **cam, plane_z=plane_z)
    assert fit is not None

    # while the predicted position is inside the occluder's span, predict_pixel
    # must refuse (None), not offer a pixel target
    t_inside = 0.5 / abs(v0_true)   # predicted x lands ~0.5m from x0_at_t0, inside occluder_bounds
    assert fit.predict_pixel(t_inside) is None, "should refuse a match while still inside the known occluder"

    # once the prediction clears the occluder's far edge, it should resume
    # offering real pixel targets again
    t_clear = 1.5 / abs(v0_true)
    assert fit.predict_pixel(t_clear) is not None, "should resume predicting once past the occluder"

    # occluded_at (AGENT.md defect #19) must agree with predict_pixel's None-ness
    # for the occluder-bounds case specifically -- it's what lets the reid search
    # loop tell "definitely still hidden" apart from "no opinion" and suppress
    # DINO's appearance-only fallback too, not just physics's own candidate gate.
    assert fit.occluded_at(t_inside) is True
    assert fit.occluded_at(t_clear) is False


def test_dynamic_occluder_bounds_tracks_a_moving_occluder():
    """AGENT.md M5.7: occluder_bounds_from_track builds a per-instant box
    from a TRACKED second object's own masks (occlusion_corridor_moving.xml's
    sliding occluder stand-in here), not a fixed constant -- the whole point
    of this scene. Also exercises the staleness fallback: the propagation
    loop that actually calls this (reidentify.py) can only ever have tracked
    UP TO the current frame at decision time, never the exact frame being
    queried (see occluder_bounds_from_track's own docstring for why), so a
    1-3 frame-old position must still resolve, and only a longer gap should
    give up and return None."""
    cam = _synthetic_camera()
    dt = 1.0 / 30
    occ_plane_z, x_half, y_half = 0.5, 0.75, 0.6
    occ_x_true = 5.0

    masks_secondary = {}
    for frame in range(6):
        occ_y_true = -1.0 + 0.3 * frame   # sliding across the corridor
        pos3d = np.array([occ_x_true, occ_y_true, occ_plane_z])
        px, py = project_to_pixel(pos3d, cam["cam_pos"], cam["cam_mat"], cam["fovy_deg"], cam["width"], cam["height"])
        m = np.zeros((cam["height"], cam["width"]), bool)
        cy, cx = int(round(py)), int(round(px))
        m[cy - 1:cy + 2, cx - 1:cx + 2] = True
        masks_secondary[frame] = m
    # frame 3 deliberately left untracked -- stands in for "not yet
    # propagated at decision time", the staleness case this function exists
    # to absorb.
    del masks_secondary[3]

    bounds_fn = occluder_bounds_from_track(masks_secondary, dt, cam_pos=cam["cam_pos"], cam_mat=cam["cam_mat"],
                                            fovy_deg=cam["fovy_deg"], width=cam["width"], height=cam["height"],
                                            occluder_plane_z=occ_plane_z, x_half_width=x_half, y_half_width=y_half,
                                            max_staleness_frames=3)

    # frame 4 exists directly -- should reflect frame 4's own true position
    x_lo, x_hi, y_lo, y_hi = bounds_fn(4 * dt)
    occ_y_true_4 = -1.0 + 0.3 * 4
    assert abs((x_lo + x_hi) / 2 - occ_x_true) < 0.05
    assert abs((y_lo + y_hi) / 2 - occ_y_true_4) < 0.1   # pixel-quantization slack (3x3 mask, int-rounded centroid)
    assert abs((x_hi - x_lo) / 2 - x_half) < 1e-6
    assert abs((y_hi - y_lo) / 2 - y_half) < 1e-6

    # frame 3 is missing -- should fall back to frame 2's own tracked position
    # (within max_staleness_frames), NOT frame 4's, and NOT None
    x_lo3, x_hi3, y_lo3, y_hi3 = bounds_fn(3 * dt)
    occ_y_true_2 = -1.0 + 0.3 * 2
    assert abs((y_lo3 + y_hi3) / 2 - occ_y_true_2) < 0.1

    # far past any tracked frame, beyond the staleness window -- honestly undefined
    assert bounds_fn(20 * dt) is None


def test_metric_prediction_dynamic_occluder_respects_lateral_axis():
    """AGENT.md M5.7's actual point: a moving occluder's x-extent is nearly
    static (barely perturbed) but its y-extent is what's time-varying, so a
    1D (x-only) occluder check -- correct for the ORIGINAL fixed wall -- is
    the WRONG model here: the ball is only genuinely behind it when the
    occluder's CURRENT y-position actually puts the ball's own lane inside
    its y half-width, not merely whenever their x-ranges happen to overlap.
    Built directly against MetricTrackFit (not through fit_metric_track) so
    the occluder_bounds callable's own time-dependence can be controlled
    exactly, independent of any tracking noise."""
    cam = _synthetic_camera()
    x_lo, x_hi = 4.25, 5.75   # the occluder's own near-static x-extent

    def occluder_bounds(t_abs):
        # in the ball's lane (y around 0) only during [1.0, 2.0]s absolute;
        # off in some other lane (y around -2.4) the rest of the time --
        # same physical picture as occlusion_corridor_moving.xml's occluder
        # sliding through y=0 partway through the clip.
        if 1.0 <= t_abs <= 2.0:
            return (x_lo, x_hi, -0.6, 0.6)
        return (x_lo, x_hi, -3.0, -1.8)

    fit = MetricTrackFit(x0=5.0, v0=0.0, lateral_value=0.0, motion_axis=0, lateral_axis=1,
                          deceleration=0.0, occluder_bounds=occluder_bounds, t0_abs=0.0, plane_z=0.15, **cam)
    # ball sits at x=5.0 always (v0=0) -- permanently inside the occluder's
    # own x-range, so x ALONE would call every one of these instants
    # "occluded" under the old 1D-only logic. The y-check is what must
    # actually decide it.
    assert fit.occluded_at(0.5) is False, "occluder is out of the ball's lane at t=0.5 -- must NOT occlude"
    assert fit.predict_pixel(0.5) is not None, "a real target should still be offered at t=0.5"

    assert fit.occluded_at(1.5) is True, "occluder is IN the ball's lane at t=1.5 -- must occlude"
    assert fit.predict_pixel(1.5) is None, "must refuse a target while genuinely behind the occluder"

    assert fit.occluded_at(2.5) is False, "occluder has moved back out of the lane by t=2.5"


def test_static_occluder_bounds_tuple_stays_unbounded_in_lateral_axis():
    """Backward-compatibility guard for M5.7's extension: the ORIGINAL 1D
    (lo, hi) tuple convention (occlusion_corridor's own permanently-fixed
    wall) must keep behaving exactly as before -- occluding whenever x is
    in range, regardless of lateral_value -- since a fixed wall was always
    known to span the object's whole lane for the life of the clip."""
    cam = _synthetic_camera()
    fit = MetricTrackFit(x0=5.0, v0=0.0, lateral_value=999.0, motion_axis=0, lateral_axis=1,
                          deceleration=0.0, occluder_bounds=(4.25, 5.75), plane_z=0.15, **cam)
    assert fit.occluded_at(0.0) is True, "static tuple form must ignore lateral_value entirely"


def test_prediction_clamps_at_implied_stop_instead_of_reversing():
    """AGENT.md defect #16: a raw quadratic fit, extrapolated far enough
    past its own fit window, has no notion that a decelerating object
    actually stops -- its implied velocity crosses zero and then
    REVERSES, curving the predicted position back the way it came. Found
    on a real occlusion_corridor gap needing ~23 frames of extrapolation:
    predicted_pos traced a full parabola (decreasing, hitting a minimum,
    then increasing again) instead of leveling off, drifting the search
    away from where the object actually came to rest."""
    dt = 0.1
    # v0=20, a=20 -> stops (v=0) at absolute t=1.0, i.e. 0.3s after the last
    # observed frame (t=0.7 absolute) -- large enough pixel spread (100->109)
    # to clear MIN_AXIS_VARIANCE and get a real fit, unlike a smaller v0
    masks = _decelerating_blob_masks(8, dt, x0=100.0, v0=20.0, a=20.0, y=80)
    fit = fit_pixel_track(masks, list(range(8)), dt)
    assert fit is not None and fit.x_fit is not None

    t_stop = 0.3
    x_at_stop, _ = fit.predict(t_stop)

    # far past the stop time -- an unclamped quadratic would have reversed
    # direction by now; the correct behavior is to stay at the resting position
    x_far, _ = fit.predict(t_stop + 2.0)
    assert abs(x_far - x_at_stop) < 1e-6, (
        f"prediction should stay clamped at the stop position ({x_at_stop}), "
        f"got {x_far} -- looks like the raw unclamped quadratic reversed direction")

    # sanity: an UNclamped evaluation of the same polynomial genuinely would
    # have reversed (this parabola opens downward -- x0=100, v0=20, a=20 --
    # so it peaks at t_stop then DECREASES past it) -- confirms this test
    # is exercising the bug, not a no-op
    raw_unclamped = np.polyval(fit.x_fit[0], t_stop + 2.0)
    assert raw_unclamped < x_at_stop, "test setup should have a real reversal to clamp against"


if __name__ == "__main__":
    test_fit_predicts_decelerating_axis_and_treats_static_axis_as_constant()
    print("PASS  test_fit_predicts_decelerating_axis_and_treats_static_axis_as_constant")
    test_fit_returns_none_with_too_few_frames()
    print("PASS  test_fit_returns_none_with_too_few_frames")
    test_candidate_gating_prefers_the_near_candidate()
    print("PASS  test_candidate_gating_prefers_the_near_candidate")
    test_centroid_of_empty_mask_is_none()
    print("PASS  test_centroid_of_empty_mask_is_none")
    test_prediction_clamps_at_implied_stop_instead_of_reversing()
    print("PASS  test_prediction_clamps_at_implied_stop_instead_of_reversing")
    test_metric_track_fit_recovers_known_position_and_velocity()
    print("PASS  test_metric_track_fit_recovers_known_position_and_velocity")
    test_metric_prediction_matches_analytic_stop_position_far_past_extrapolation()
    print("PASS  test_metric_prediction_matches_analytic_stop_position_far_past_extrapolation")
    test_metric_prediction_refuses_a_match_inside_known_occluder_bounds()
    print("PASS  test_metric_prediction_refuses_a_match_inside_known_occluder_bounds")
    test_dynamic_occluder_bounds_tracks_a_moving_occluder()
    print("PASS  test_dynamic_occluder_bounds_tracks_a_moving_occluder")
    test_metric_prediction_dynamic_occluder_respects_lateral_axis()
    print("PASS  test_metric_prediction_dynamic_occluder_respects_lateral_axis")
    test_static_occluder_bounds_tuple_stays_unbounded_in_lateral_axis()
    print("PASS  test_static_occluder_bounds_tuple_stays_unbounded_in_lateral_axis")
