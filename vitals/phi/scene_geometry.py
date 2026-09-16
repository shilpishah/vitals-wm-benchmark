"""Canonical per-scenario RECONSTRUCTION geometry -- camera framing, ground/
plane model (flat / piecewise-plane / ballistic), object radius, and
occluder geometry: everything `reconstruct.py` and `motion_prior.py` need to
turn a scenario's own video back into a 3D Trajectory or an occluder-aware
search prior.

Why this exists (AGENT.md M6's own input-normalization protocol draft):
"one adapter per real model, not one per scenario" only actually holds if
the scene-specific half of that adapter -- reconstruction -- is driven by
DATA (this module) rather than by branching code duplicated at every call
site. Before this module, that data lived as independently-drifting copies
across `tests/test_reconstruct.py`, `scripts/run_gate2.py`, and
`remote/modal_app.py`.

The drift was real, not hypothetical -- confirmed while consolidating, not
assumed: `remote/modal_app.py`'s own wall-occlusion bound was already
`(4.39, 5.87)`, an EMPIRICALLY-CORRECTED value (the real visual occlusion
boundary at this camera's own oblique angle -- see its own long comment),
while the constant of the same name and purpose quoted in `motion_prior.py`
docstring examples was still the naive `(4.25, 5.75)` derived directly from
the wall geom's raw XML pos/half-size. Both were "the wall's occluder
bounds" for the identical scene; only one was right. `OCCLUSION_CORRIDOR_
WALL_BOUNDS` below is now the SINGLE source of that number -- import it, do
not redefine it locally, even for "just an example."

Cameras defined here are chosen for MEASUREMENT ACCURACY -- the angle
`tests/test_reconstruct.py` actually validated reconstruction position
error against -- not for how good a video looks.
`scripts/visualize_reference_ensemble.py` deliberately uses different, more
flattering angles for its own ensemble videos (e.g. occlusion_corridor at
elevation=-10 there vs. elevation=-45 here); those are a different camera
for a different purpose and are NOT meant to be unified with this module.

Each scenario's entry is a plain dict of keyword arguments matched to
`reconstruct.reconstruct_trajectory`'s own `mode` contract (exactly one of
plane_z / plane_pieces / ballistic=True) plus whatever `motion_prior.py`
needs for occluder-aware search on that scenario, so a caller can do:

    cfg = SCENES[scenario_name]
    reconstruct_trajectory(masks, fps, cam_pos=..., cam_mat=..., ...,
                            **cfg["reconstruct_kwargs"])

without knowing ahead of time which scenario it's looking at.
"""
import numpy as np

BALL_RADIUS = 0.15   # every scene's ball geom (scenes/*.xml) -- shared across all reconstruction

# --- occlusion_corridor family: flat ground, single fixed plane ------------
# Same camera and ball geometry for every variant (distractor/moving add a
# SECOND body, never change the ball's own geometry or the wall's nominal
# position -- see each scene's own docstring) -- not re-measured per variant.
_OCCLUSION_CORRIDOR_CAM = [dict(lookat=[3.0, 0.0, 0.3], distance=9.5, azimuth=-90, elevation=-45)]

# The wall's TRUE empirical visual occlusion span, not just its own box
# half-extent (scenes/occlusion_corridor.xml: pos="5 0 0.5", size="0.75 0.6
# 0.5" -> naive world-x span = 5 +/- 0.75 = (4.25, 5.75)). Scene geometry,
# not privileged object state -- AGENT.md defect #17's fix -- but the naive
# box-extent value undersold the true occlusion, particularly on the far
# edge (measured 0.12m short), because this camera's oblique viewing angle
# doesn't produce a clean vertical occlusion plane exactly at the box's own
# x-boundary. Measured directly (2026-08), not assumed: swept the ball
# through fixed x positions at 5mm resolution and recorded actual rendered
# visibility -- true span is (4.395, 5.870) (last/first visible just
# outside), not (4.25, 5.75). This 0.12-0.15m gap on the far edge is
# exactly what let several GATE 2 null instances (whose resting position
# landed in that gap) confuse the physics-prior's occluder check: it
# believed they'd cleared the wall when they genuinely had not (AGENT.md
# §7 M5's own writeup on the "coarse-boundary-margin" mechanism). Passed as
# `occluder_bounds` to `motion_prior.fit_metric_track` -- see that
# function's own docstring for the exact convention. THIS is now the one
# place this number is defined -- `remote/modal_app.py` imports it rather
# than keeping its own copy (it originally did, and that copy is exactly
# what had drifted from the naive value still quoted as an example in
# `motion_prior.py`'s own docstring before this module existed).
#
# UNCHANGED by the 2026-08 "lit opening" decal added to the wall's near
# face (scenes/occlusion_corridor*.xml's own `wall_opening`/`occluder_
# opening` geom) -- that decal sits just in front of the SAME wall geom
# this span was measured against, purely cosmetic (a second, thin,
# non-tracked geom), and re-measured directly afterward with the decal
# present to confirm: identical (4.395, 5.870), not re-derived from
# scratch and assumed unchanged.
OCCLUSION_CORRIDOR_WALL_BOUNDS = (4.39, 5.87)

# Reference-ensemble deceleration for the occlusion_corridor family's own
# rolling-then-sliding ball physics (detect/physical_constants.py::
# fit_corridor_deceleration, CV=0.12% across M=30 references -- AGENT.md
# defect #16), consumed by motion_prior.fit_metric_track's physics-prior
# re-identification. Same drift this module exists to prevent (module
# docstring's own wall-bounds story), found a second time (2026-08, M2.5's
# GATE 2 generalization): this constant lived ONLY as `remote/modal_app.
# py::OCCLUSION_CORRIDOR_DECELERATION`, hardcoded directly into the OLD,
# pre-unification `run_gate2_episode` -- never migrated here, so `default_
# phi_reconstruct` (every real-model score on occlusion_corridor, Cosmos
# n=80 and Wan included) has been silently calling `cfg.get("deceleration")`
# and getting None this whole time, un-noticed until GATE 2 was unified
# onto the same generic path and its own velocity_freeze L1 correct-risk
# visibly regressed (5/8, using the hardcoded value directly, to 2/8 once
# it ALSO started reading this same, previously-incomplete config) --
# confirmed by directly re-running GATE 2 before and after this fix, not
# assumed from reading the code.
OCCLUSION_CORRIDOR_DECELERATION = 1.246

# occlusion_corridor_moving's occluder is TRACKED, not fixed (AGENT.md
# M5.7) -- its SIZE is still known scene geometry (scenes/occlusion_
# corridor_moving.xml: occluder box size (0.75, 0.6, 0.5) at center
# z=0.5), passed to `motion_prior.occluder_bounds_from_track` as
# (occluder_plane_z, x_half_width, y_half_width). Its position is NOT
# here -- that comes from tracking, every frame, at call time.
OCCLUSION_CORRIDOR_MOVING_OCCLUDER_GEOMETRY = (0.5, 0.75, 0.6)

# --- ramp_descent: known, fixed, but PIECEWISE plane ------------------------
# scenes/ramp_descent.xml's own comments give the exact top/toe endpoints;
# cross-validated independently here via the box geom's own center/size/euler
# rather than trusting the comment blindly: computed edge points
# (-0.900, 0, 0.00007) and (-4.2856, 0, 1.80002), normal (0.46947, 0,
# 0.88295) -- matches the scene's own claimed (-0.9, 0, 0) / (-4.286, 0, 1.8)
# / slope tan(28 deg)=0.5317 to 4-5 significant figures. AGENT.md M4.5.
RAMP_CAM = [dict(lookat=[0.5, 0.0, 0.6], distance=8.5, azimuth=-90, elevation=-12)]
_RAMP_SURFACE_POINT = np.array([-4.2856, 0.0, 1.80002])
_RAMP_NORMAL = np.array([0.46947, 0.0, 0.88295])
RAMP_TOE_X = -0.900416
RAMP_PLANE_PIECES = [
    # ramp incline -- ball CENTER plane is the surface offset by its own radius
    # along the normal (same "plane_z is the ball's center height, not the floor
    # itself" convention as the flat-ground case), valid while the reconstructed
    # x is still up-ramp of the toe (small margin for discretization slack).
    (tuple(_RAMP_SURFACE_POINT + BALL_RADIUS * _RAMP_NORMAL), tuple(_RAMP_NORMAL),
     lambda p: p[0] <= RAMP_TOE_X + 0.05),
    # flat floor past the toe -- same z=BALL_RADIUS convention as occlusion_corridor's
    # own flat-ground case. Always valid -- the fallback once the ramp piece's own
    # domain check fails, not a claim that every such frame is truly on the floor.
    ((0.0, 0.0, BALL_RADIUS), (0.0, 0.0, 1.0), lambda p: True),
]

# ramp_descent.xml's own ball is DROPPED from height ("a small, consistent
# settling fall", that file's own comment) -- genuinely airborne, on NO
# registered plane at all, for a known, fixed initial window, before
# landing on the ramp. AGENT.md M2.6: forcing a plane-based unprojection
# during that window doesn't fail loudly -- it silently produces a large,
# wrong position (measured directly: 0.90m error at frame 0, decaying to
# <0.001m by frame 10). Analytical free-fall time from the scene's own
# stated drop (start z=1.7, ramp top-surface z=1.28 at the ball's own
# start x=-3.3, both from ramp_descent.xml's own comments):
# sqrt(2*(1.7-1.28)/9.81) = 0.293s = 8.78 frames @ 30fps -- an
# almost-exact match to the empirical measurement above, confirming this
# is a deterministic, scene-governed constant, not something needing
# per-episode detection. 12 frames (0.4s) used below, not the bare
# analytical minimum, for margin against per-episode lambda-perturbation
# variance in the exact drop trajectory -- verified directly against a
# real GATE 2 re-run, not assumed sufficient from the arithmetic alone.
RAMP_SETTLING_FRAMES = 12

# --- projectile: true free 3D motion, no known plane at all -----------------
# AGENT.md M4.5's "requirement class 2" -- camera used throughout this
# scenario's own ballistic-reconstruction debugging and validation.
PROJECTILE_CAM = [dict(lookat=[2.5, 0.0, 0.5], distance=8.0, azimuth=-90, elevation=-20)]

# --- collision: rigid-body collision, flat ground ----------------------------
# 2026-08, Phase 1 scenario diversity -- both balls stay on the floor the
# whole clip (plane_z reconstruction, same mode as occlusion_corridor),
# only a NEW physical regime (real momentum-transfer collision), not a new
# reconstruction capability. lookat centered on the collision zone's own
# midpoint (cue starts x=0, settles ~2.7; target starts x=3, settles
# ~3.1) -- confirmed directly (not assumed) that both balls stay visible
# throughout at this framing, not just at the two endpoints.
COLLISION_CAM = [dict(lookat=[1.75, 0.0, 0.3], distance=9.0, azimuth=-90, elevation=-30)]

# billiards.xml spans a wider x-range than collision.xml (cue starts at
# x=0, ball B rests at x=5.0, and post-impact travel pushes both target
# balls further still) -- lookat/distance widened accordingly so all
# three bodies stay framed through both impacts, not just COLLISION_CAM's
# own narrower reuse.
BILLIARDS_CAM = [dict(lookat=[4.0, 0.0, 0.3], distance=13.0, azimuth=-90, elevation=-30)]

# --- occlusion_reemergence: corridor physics, 3.5m wall, wider camera -------
# 2026-09-11. Same azimuth/elevation as the corridor camera (the occluder-
# aware motion prior and the "near face blocks the ray" geometry were
# validated at elevation -45), pulled back and re-centered so the ball is
# in frame from x=0 to past x=11 (scripts/measure_occluder_visibility.py
# with this camera: visible [-0.5, 5.05), hidden [5.05, 8.50], visible
# (8.50, 11.0]; under the corridor camera the ball left the frame at
# ~8.5 and the "hidden" span ran to the end of the sweep). The hidden
# span is the MEASURED value, not the wall's XML extent [4.90, 8.40]:
# same +0.1..0.15m oblique offset the corridor's own bounds show.
# 2026-09-12, second camera: lookat 4.7 / distance 13 (the first choice)
# left the ball only 36 px in area (~7 px across) at the wall and at its
# re-emergence point ~4m from the look-at -- and GATE 2's remaining 4/10
# null R1 alarms were exactly the corridor's documented small-object
# re-identification failure (the search never matched the tiny re-emerged
# ball, so the whole occlusion gap stayed "unexplained"; the physics
# prior was NOT the cause: this scene's rolling deceleration measured
# 1.246 +- 0.001 m/s^2, identical to the corridor constant). Candidates
# measured with the same sweep: lookat 5.2 / distance 11 keeps x in
# [-0.5, 10.5] in frame with the ball at 51 px (+40%); distance 10 clips
# the start (x >= -0.22). Bounds re-measured under the chosen camera.
OCCLUSION_REEMERGENCE_CAM = [dict(lookat=[5.2, 0.0, 0.3], distance=11.0, azimuth=-90, elevation=-45)]
OCCLUSION_REEMERGENCE_WALL_BOUNDS = (5.04, 8.50)   # measured: ball CENTER x where the GT mask is EMPTY (fully hidden)
# Cross-check of the radius rule below: the span where the GT mask area is
# < 50% of its max, measured directly under this camera, is [4.88, 8.66] --
# the widened bounds (5.04-0.15, 8.50+0.15) = (4.89, 8.65) match to 1cm.
# The span the tracker's own visibility logic must be given is WIDER than
# the fully-hidden span by one object radius on each side -- found
# directly on this scene's first GATE 2 re-run (2026-09-12): 4/10 null
# instances fired R1 at 1.13-1.20s, i.e. at occlusion ENTRY (1.18s),
# because reidentify.py declares the object not-visible once its mask
# drops below 50% of its max area (its center ~one radius short of the
# fully-hidden edge), while the physics fit's `occluded_at` was checked
# against the fully-hidden span -- so the first search attempts landed
# in the partial-occlusion strip, were recorded as NOT known-occluded,
# and the gap went unexplained (reconstruct._existence_mask). This is the
# corridor's own long-documented "coarse boundary margin" false-alarm
# class, made explicit: an object whose CENTER is within one radius of
# the hidden span is, for visibility purposes, inside the occluder. The
# exit-side widening only delays the un-occluded verdict by ~3 frames
# at this speed and costs nothing (SAM2 keeps tracking regardless).
# Deliberately NOT retro-applied to occlusion_corridor's own entry: its
# GATE 2 history and published populations were measured without it.
OCCLUSION_REEMERGENCE_OCCLUDER_BOUNDS = (OCCLUSION_REEMERGENCE_WALL_BOUNDS[0] - BALL_RADIUS,
                                         OCCLUSION_REEMERGENCE_WALL_BOUNDS[1] + BALL_RADIUS)

# --- render-domain intervention (AGENT.md M10, 2026-09-16) --------------------
# Two visual-only variants of occlusion_reemergence (identical physics, XML
# geometry and keyframe): domA changes materials and lighting under the SAME
# camera, so its occlusion span is the base scene's; domB keeps the base
# materials and moves the CAMERA to an oblique view, so its span was
# re-measured under that camera with scripts/measure_occluder_visibility.py
# (--x-lo 4.0 --x-hi 9.5 --step 0.01): hidden [5.05, 8.77], continuous (the
# oblique view hides the ball 0.27m longer on the exit side than the base
# view's [5.04, 8.50]); widened by one radius each side exactly as the base.
OCCLUSION_REEMERGENCE_DOMB_CAM = [dict(lookat=[5.2, 0.0, 0.3], distance=11.0, azimuth=-65, elevation=-32)]
OCCLUSION_REEMERGENCE_DOMB_WALL_BOUNDS = (5.05, 8.77)
OCCLUSION_REEMERGENCE_DOMB_OCCLUDER_BOUNDS = (OCCLUSION_REEMERGENCE_DOMB_WALL_BOUNDS[0] - BALL_RADIUS,
                                              OCCLUSION_REEMERGENCE_DOMB_WALL_BOUNDS[1] + BALL_RADIUS)

# --- block_stack: ball strikes a two-block tower at x=6.5 -----------------------
# 2026-09-11. Collision-style framing widened to hold x in [0, 7.5] plus a
# 0.72m-tall post; verified by rendering frame 0 and t=3s (all three
# bodies in frame, post upright at 0, toppled at 3s).
BLOCK_STACK_CAM = [dict(lookat=[3.5, 0.0, 0.4], distance=11.5, azimuth=-90, elevation=-30)]


# --- soft_drop: deformable ellipsoid dropped from 0.8m with a +x push ----------
# 2026-09-14 (AGENT.md M9). Rest radius 0.175m (8x8x8 flexcomp, 0.05m
# spacing: 7 gaps * 0.05 / 2). The centroid sits at ~0.175m on the floor;
# plane_z unprojection uses that. Close framing: the body spans ~40px at
# the corridor's usual distance, too few pixels for shape descriptors.
# Verified by rendering (frame 0 airborne top-left, landing, roll to
# x~1.3 by 4s all in frame).
SOFT_DROP_RADIUS = 0.175
SOFT_DROP_CAM = [dict(lookat=[0.8, 0.0, 0.4], distance=3.5, azimuth=-90, elevation=-12)]
# The pixel path unprojects the mask centroid onto a horizontal plane at
# the body's RESTING centroid height -- measured 0.112m over 60 references
# (the soft body sits 6cm lower than its rest radius: 7% static deflection,
# scenes/soft_drop.xml) -- and, exactly as ramp_descent's own
# RAMP_SETTLING_FRAMES, declares the frames before the body has settled
# UNRELIABLE for position (NaN, never a spurious plane point): the drop
# from 1.2m lands at frame ~16 and every reference is within 1cm of its
# resting height from frame 38 (1.27s) on; 40 frames for margin. Found
# by the first GATE 2 run (2026-09-14): unprojecting an AIRBORNE body onto
# the floor plane put it 9m away and fired R3 on every null at 0.13s --
# the airborne-reconstruction gap (M2.7) re-found on a new scene, handled
# the same way. Shape descriptors come straight from the mask and are
# NOT subject to this prefix (R6/R7 can fire from the first frame).
# Consequence, stated plainly: with a 1s conditioning prefix the scored
# continuation of this scene is a body settling and then resting -- the
# material channels carry the test; R3 guards drift and sliding.
SOFT_DROP_REST_Z = 0.112
SOFT_DROP_SETTLING_FRAMES = 40

# --- soft_ramp: deformable body released on ramp_descent's incline ----------
# 2026-09-14 (AGENT.md M9.1). Same incline/floor/camera as ramp_descent, so
# the plane pieces are ramp_descent's own with the SOFT body's centroid
# offset instead of the rigid ball's radius: measured over a rollout,
# 0.151 m along the ramp normal while on the incline (std 0.005) and
# 0.142 m above the floor at rest (young-1000 body, 3-5% squashed under
# its own weight; the rigid ball's is 0.15 both places). Release is 0.25 m
# above the surface, so the settling prefix is short: 15 frames.
SOFT_RAMP_RADIUS = 0.175
SOFT_RAMP_OFFSET_INCLINE = 0.151
SOFT_RAMP_OFFSET_FLOOR = 0.142
SOFT_RAMP_SETTLING_FRAMES = 15
SOFT_RAMP_PLANE_PIECES = [
    (tuple(_RAMP_SURFACE_POINT + SOFT_RAMP_OFFSET_INCLINE * _RAMP_NORMAL), tuple(_RAMP_NORMAL),
     lambda p: p[0] <= RAMP_TOE_X + 0.05),
    ((0.0, 0.0, SOFT_RAMP_OFFSET_FLOOR), (0.0, 0.0, 1.0), lambda p: True),
]


def _plane_z_kwargs():
    return dict(mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS))


SCENES = {
    # deceleration=OCCLUSION_CORRIDOR_DECELERATION on all three of this
    # scene's own variants -- same ball, same floor friction, same rolling
    # physics regardless of what else is dressing the scene (a decoy, a
    # moving occluder); only the wall/occluder-bounds config differs
    # between them.
    "occlusion_corridor": dict(
        camera=_OCCLUSION_CORRIDOR_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=OCCLUSION_CORRIDOR_WALL_BOUNDS,
        deceleration=OCCLUSION_CORRIDOR_DECELERATION,
    ),
    "occlusion_corridor_distractor": dict(
        camera=_OCCLUSION_CORRIDOR_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=OCCLUSION_CORRIDOR_WALL_BOUNDS,
        deceleration=OCCLUSION_CORRIDOR_DECELERATION,
    ),
    # Reuses scenes/occlusion_corridor_distractor.xml AS-IS (the manifest's
    # own comment) -- same scene, same camera, same ball, same decoy, so
    # this entry is byte-identical to occlusion_corridor_distractor's own
    # above. What differs is entirely in run_l0_demo.py's/the real-model
    # adapter's own handling of target_property == "P3", not scene
    # geometry. No entry existed here before this (2026-09): this manifest
    # had never been run through anything that calls scene_geometry.get()
    # -- only state-space GATE 1/GATE 1b, which never needs it.
    "occlusion_corridor_interpenetration": dict(
        camera=_OCCLUSION_CORRIDOR_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=OCCLUSION_CORRIDOR_WALL_BOUNDS,
        deceleration=OCCLUSION_CORRIDOR_DECELERATION,
    ),
    "occlusion_corridor_moving": dict(
        camera=_OCCLUSION_CORRIDOR_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=None,   # NOT fixed -- see occluder_geometry below
        occluder_geometry=OCCLUSION_CORRIDOR_MOVING_OCCLUDER_GEOMETRY,
        deceleration=OCCLUSION_CORRIDOR_DECELERATION,
    ),
    # occlusion_reemergence (2026-09-11, scenario scaling): occlusion_corridor's
    # floor/ball/physics with a 3.5m wall -- same rolling deceleration, its
    # own wider camera (the ball travels to x~9.4 by t=3s; under the corridor
    # camera it leaves the frame at x~8.5, measured) and its own measured
    # occlusion span under that camera.
    "occlusion_reemergence": dict(
        camera=OCCLUSION_REEMERGENCE_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=OCCLUSION_REEMERGENCE_OCCLUDER_BOUNDS,   # radius-widened; see its own comment
        deceleration=OCCLUSION_CORRIDOR_DECELERATION,
    ),
    # M10 render-domain variants (2026-09-16): domA = same camera, new
    # materials/lighting; domB = same materials, oblique camera with its
    # own measured occlusion span. Same physics, deceleration and plane.
    "occlusion_reemergence_domA": dict(
        camera=OCCLUSION_REEMERGENCE_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=OCCLUSION_REEMERGENCE_OCCLUDER_BOUNDS,
        deceleration=OCCLUSION_CORRIDOR_DECELERATION,
    ),
    "occlusion_reemergence_domB": dict(
        camera=OCCLUSION_REEMERGENCE_DOMB_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=OCCLUSION_REEMERGENCE_DOMB_OCCLUDER_BOUNDS,
        deceleration=OCCLUSION_CORRIDOR_DECELERATION,
    ),
    # block_stack (2026-09-11, scenario scaling, P3): flat ground, plane_z
    # for the ball AND the bottom cube (its half-size equals BALL_RADIUS so
    # both centroids sit at z=0.15 -- the P3 secondary object is unprojected
    # onto the same plane as the ball, video_model.py's own convention). The
    # post on top is not state-scored (scenes/block_stack.xml's own comment).
    "block_stack": dict(
        camera=BLOCK_STACK_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=None,
    ),
    # soft_drop (2026-09-14, AGENT.md M9): the first soft-body scenario.
    # `soft_body=True` tells the scoring scripts to attach shape descriptors
    # (softbody.attach_shape) and to score R6 conservation / R7 shape.
    "soft_ramp": dict(
        camera=RAMP_CAM, object_radius=SOFT_RAMP_RADIUS,
        mode="plane_pieces",
        reconstruct_kwargs=dict(plane_pieces=SOFT_RAMP_PLANE_PIECES,
                                 unreliable_prefix_frames=SOFT_RAMP_SETTLING_FRAMES, with_shape=True),
        occluder_bounds=None, soft_body=True,
    ),
    "soft_drop": dict(
        camera=SOFT_DROP_CAM, object_radius=SOFT_DROP_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=SOFT_DROP_REST_Z, with_shape=True,
                                                 unreliable_prefix_frames=SOFT_DROP_SETTLING_FRAMES),
        occluder_bounds=None, soft_body=True,
    ),
    "ramp_descent": dict(
        camera=RAMP_CAM, object_radius=BALL_RADIUS,
        mode="plane_pieces",
        reconstruct_kwargs=dict(plane_pieces=RAMP_PLANE_PIECES,
                                 unreliable_prefix_frames=RAMP_SETTLING_FRAMES),
        occluder_bounds=None,
    ),
    # Same physical geometry as ramp_descent -- ramp_descent_high_friction.xml
    # (AGENT.md M2.5) only changes the default friction triple, not the
    # camera, ramp plane equations, ball radius, or drop height, so it
    # reuses this entry's own RAMP_CAM/RAMP_PLANE_PIECES/RAMP_SETTLING_
    # FRAMES rather than re-deriving anything.
    "ramp_descent_high_friction": dict(
        camera=RAMP_CAM, object_radius=BALL_RADIUS,
        mode="plane_pieces",
        reconstruct_kwargs=dict(plane_pieces=RAMP_PLANE_PIECES,
                                 unreliable_prefix_frames=RAMP_SETTLING_FRAMES),
        occluder_bounds=None,
    ),
    # collision (2026-08, Phase 1 scenario diversity): K=2 (cue + target
    # ball), but only the cue (object index 0, the one every mutant/sigma_k
    # `obj=0` default already scores) is reconstructed and scored this
    # pass -- scenes/collision.xml's own module comment has the full v1-
    # scope reasoning (target-ball scoring is a real, separate, not-yet-
    # attempted extension, same "single-object first" discipline already
    # used for P2/P3). plane_z reconstruction -- both balls stay on the
    # floor the whole clip, no new reconstruction capability needed.
    "collision": dict(
        camera=COLLISION_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=None,
    ),
    # 2026-09, billiards/multi-collision scoping: same flat-ground plane_z
    # reconstruction mode as collision -- all three balls stay on the
    # floor the whole clip, no new reconstruction capability needed, only
    # a wider camera (BILLIARDS_CAM) and track_all_objects: true
    # (EpisodeSpec) to actually reconstruct all three bodies for scoring.
    "billiards": dict(
        camera=BILLIARDS_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=None,
    ),
    "projectile": dict(
        camera=PROJECTILE_CAM, object_radius=BALL_RADIUS,
        mode="ballistic", reconstruct_kwargs=dict(ballistic=True, object_radius=BALL_RADIUS,
                                                   piecewise_ballistic=True),
        occluder_bounds=None,
    ),
}


def get(scenario_name):
    """Returns this scenario's canonical reconstruction config, or raises --
    a scenario with no entry here needs one added (see module docstring),
    not a silent default guessed at the call site."""
    if scenario_name not in SCENES:
        raise KeyError(f"no reconstruction geometry registered for scenario {scenario_name!r} -- "
                        f"add an entry to vitals/phi/scene_geometry.py's own SCENES dict, "
                        f"do not redefine one locally at the call site.")
    return SCENES[scenario_name]
