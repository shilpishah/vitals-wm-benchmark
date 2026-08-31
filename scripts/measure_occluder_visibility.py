"""Empirically measures a scene's occluder visibility span for the ball,
by rendering real MuJoCo frames with the ball SWEPT through a fixed x range
(fine resolution) and reading REAL segmentation off the renderer -- no
assumption from box geometry, no privileged occlusion knowledge, the same
kind of direct measurement `scene_geometry.py`'s own OCCLUSION_CORRIDOR_
WALL_BOUNDS comment describes doing once (2026-08, un-scripted at the time
-- this is that measurement made reusable, needed again for the tunnel
restyle: "make the inside of the corridor lit... so it's clear that it's
open" -- geometry changed from a solid box to a hollow tunnel (side walls +
roof), which visually reads as a passage instead of a dead end, but the
EFFECTIVE camera-relative occlusion span for a hollow shape is not
guaranteed identical to a solid box's -- must be re-measured, not assumed
carried over).

    python3 scripts/measure_occluder_visibility.py --scene scenes/occlusion_corridor.xml
    python3 scripts/measure_occluder_visibility.py --scene scenes/occlusion_corridor.xml \\
        --x-lo 3.5 --x-hi 6.5 --step 0.005
"""
import sys
import pathlib
import argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from vitals.types import Trajectory
from vitals.render.mujoco_renderer import MujocoRenderer
from vitals.phi import scene_geometry as sg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--x-lo", type=float, default=3.5)
    ap.add_argument("--x-hi", type=float, default=6.5)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--y", type=float, default=0.0, help="ball's lateral position -- 0.0 is the ball's own nominal lane")
    ap.add_argument("--camera-lookat", type=float, nargs=3, default=[3.0, 0.0, 0.3])
    ap.add_argument("--camera-distance", type=float, default=9.5)
    ap.add_argument("--camera-azimuth", type=float, default=-90)
    ap.add_argument("--camera-elevation", type=float, default=-45)
    args = ap.parse_args()

    xs = np.arange(args.x_lo, args.x_hi + args.step / 2, args.step)
    T = len(xs)
    pos = np.zeros((T, 1, 3))
    pos[:, 0, 0] = xs
    pos[:, 0, 1] = args.y
    pos[:, 0, 2] = sg.BALL_RADIUS
    quat = np.zeros((T, 1, 4))
    quat[:, 0, 0] = 1.0
    traj = Trajectory(t=np.arange(T) / 30.0, pos=pos, quat=quat,
                       present=np.ones((T, 1), bool), names=["ball"])

    cam = [dict(lookat=args.camera_lookat, distance=args.camera_distance,
                azimuth=args.camera_azimuth, elevation=args.camera_elevation)]
    renderer = MujocoRenderer(args.scene, height=240, width=320)
    frames, gt = renderer.render(traj, cameras=cam)

    visible = np.array([(gt.segmentation[i] == 0).sum() > 0 for i in range(T)])
    print(f"scene={args.scene}  swept x in [{args.x_lo}, {args.x_hi}] step {args.step}, y={args.y}")

    if visible.all():
        print("ball visible at every swept x -- no occlusion found in this range (widen --x-lo/--x-hi?)")
        return
    if not visible.any():
        print("ball invisible at every swept x -- occluder covers the whole range (widen --x-lo/--x-hi?)")
        return

    hidden_idx = np.where(~visible)[0]
    lo, hi = xs[hidden_idx[0]], xs[hidden_idx[-1]]
    # Report any visible gaps WITHIN the hidden span too -- a tunnel's roof
    # or side-wall seams could leak a sightline a solid box never could;
    # silently reporting only the outer envelope would hide that.
    gaps = []
    prev_hidden = True
    for i in range(hidden_idx[0], hidden_idx[-1] + 1):
        if visible[i] and prev_hidden:
            gap_start = xs[i]
        if not visible[i] and not prev_hidden:
            gaps.append((gap_start, xs[i - 1]))
        prev_hidden = not visible[i]

    print(f"first hidden x = {lo:.4f}   last hidden x = {hi:.4f}   (span = {hi - lo:.4f}m)")
    if gaps:
        print(f"WARNING: {len(gaps)} visibility LEAK(S) inside that span (ball briefly visible again):")
        for g_lo, g_hi in gaps:
            print(f"  leak at x in [{g_lo:.4f}, {g_hi:.4f}]")
    else:
        print("occlusion is continuous (no leaks) across the hidden span")


if __name__ == "__main__":
    main()
