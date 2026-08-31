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
    "occlusion_corridor_moving": dict(
        camera=_OCCLUSION_CORRIDOR_CAM, object_radius=BALL_RADIUS,
        mode="plane_z", reconstruct_kwargs=dict(plane_z=BALL_RADIUS),
        occluder_bounds=None,   # NOT fixed -- see occluder_geometry below
        occluder_geometry=OCCLUSION_CORRIDOR_MOVING_OCCLUDER_GEOMETRY,
        deceleration=OCCLUSION_CORRIDOR_DECELERATION,
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
