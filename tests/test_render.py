"""M3 -- MujocoRenderer sanity checks. Requires MuJoCo; skipped (not
failed) if it isn't installed, same policy as test_gates.py.

Kept deliberately cheap (short horizon, low resolution) -- this validates
the renderer's contract (shapes, value ranges, segmentation finds the
tracked object, depth is finite and positive, camera basis is orthonormal),
not anything about render realism.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

SCENE = str(pathlib.Path(__file__).resolve().parents[1] / "scenes" / "ramp_descent.xml")


def _traj():
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    spec = EpisodeSpec(name="t", scene=SCENE, target_property="P4", band="I",
                        lam=1.0, n_reference=1, horizon_s=1.0, fps=30)
    roll = make_backend("mujoco", scene=SCENE)
    return roll(spec, 1)


def test_render_shapes_and_ranges():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIP  mujoco not installed")
        return
    from vitals.render.mujoco_renderer import MujocoRenderer

    traj = _traj()
    r = MujocoRenderer(SCENE, height=120, width=160)
    frames, gt = r.render(traj)

    T = traj.T
    assert frames.rgb.shape == (T, 120, 160, 3)
    assert frames.rgb.dtype == np.uint8
    assert frames.fps == 30

    assert gt.segmentation.shape == (T, 120, 160)
    assert set(np.unique(gt.segmentation)) <= {-1, 0}   # K=1: background or "ball"
    assert (gt.segmentation == 0).any(), "ball never visible in any frame -- camera framing is wrong"

    assert gt.depth.shape == (T, 120, 160)
    assert np.isfinite(gt.depth).all()
    assert (gt.depth > 0).all()

    assert gt.cam_mat.shape == (3, 3)
    assert np.allclose(gt.cam_mat.T @ gt.cam_mat, np.eye(3), atol=1e-5), "camera basis isn't orthonormal"

    assert gt.names == ["ball"]


def test_moving_occluder_slides_without_stalling():
    """AGENT.md M5.6/M5.7 (phase 4, moving occluder): regression guard for
    a real bug found and fixed while building this scene -- a box given a
    real freejoint + sliding velocity on the floor decelerated to a full
    stop within ~0.1s REGARDLESS of friction coefficient (confirmed down to
    friction=0), because of box-plane contact bounce/tumble, not sliding
    friction. Fixed by making the occluder weightless (gravcomp) and fully
    non-colliding (same treatment the original static wall already had),
    giving clean constant-velocity motion. This test would have caught the
    original bug directly: it asserts the occluder actually crosses the
    corridor, not that it merely twitches near its start position."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIP  mujoco not installed")
        return
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend

    scene = str(pathlib.Path(__file__).resolve().parents[1] / "scenes" / "occlusion_corridor_moving.xml")
    spec = EpisodeSpec(name="t", scene=scene, target_property="P2", band="I",
                        lam=6.0, n_reference=1, horizon_s=8.0, fps=30, perturb_mode="velocity_x_only")
    roll = make_backend("mujoco", scene=scene)
    traj = roll(spec, 1)

    assert traj.names == ["ball", "occluder"]
    occluder_y = traj.pos[:, 1, 1]
    # started at y=-2.5 (occlusion_corridor_moving.xml's own keyframe) -- if
    # the bounce/tumble bug were still present, this would stall within the
    # first few frames and never travel more than a few centimetres.
    assert occluder_y[-1] - occluder_y[0] > 3.0, \
        f"occluder barely moved ({occluder_y[0]:.3f} -> {occluder_y[-1]:.3f}) -- box-contact stall regression?"
    assert np.allclose(traj.pos[:, 1, 2], 0.5, atol=1e-6), "occluder should stay at a constant height (weightless, non-colliding)"


if __name__ == "__main__":
    test_render_shapes_and_ranges()
    print("PASS  test_render_shapes_and_ranges (see SKIP above if mujoco absent)")
    test_moving_occluder_slides_without_stalling()
    print("PASS  test_moving_occluder_slides_without_stalling (see SKIP above if mujoco absent)")
