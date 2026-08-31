"""MuJoCo-based Renderer: Trajectory -> (Frames, GroundTruth).

The minimal, honest version of AGENT.md's M3. Reuses the exact rendering
mechanism already proven in scripts/visualize_reference_ensemble.py (load
model, set qpos per frame from a Trajectory, mujoco.Renderer.render()),
rather than standing up a separate engine (e.g. Blender, as AGENT.md
originally sketched) -- MuJoCo's own renderer already emits everything the
Renderer protocol needs: RGB, per-pixel instance segmentation, depth, and
camera extrinsics, all from the same renderer instance.

Deliberately NOT high-realism. MuJoCo's default materials/lighting are
simple CG (flat shading, no textures) -- fine for developing and
validating Phi, but AGENT.md flags render realism as the highest residual
risk (T2) for evaluating real video-generation models, which are trained
on natural imagery. `realism` is accepted for forward compatibility (a
future materials/lighting upgrade, domain randomization, or swapping in a
different Renderer implementation via the same protocol) but today only
toggles shadows -- don't read anything more into it, and don't cite this
renderer's output as evidence about realism-sensitivity until that upgrade
exists.

No video encoding anywhere in this module -- see Frames' docstring.
"""
from __future__ import annotations
import numpy as np
import mujoco
from ..types import Trajectory
from . import Frames, GroundTruth

DEFAULT_CAMERA = dict(lookat=[0.5, 0.0, 0.6], distance=8.5, azimuth=-90, elevation=-12)


class MujocoRenderer:
    name = "mujoco"
    deterministic = True
    emits_ground_truth = True

    def __init__(self, scene_path, height=360, width=640):
        self.scene_path = scene_path
        self.height, self.width = height, width

    def render(self, traj: Trajectory, cameras=None, realism="low"):
        cam_kwargs = (cameras[0] if cameras else DEFAULT_CAMERA)
        model = mujoco.MjModel.from_xml_path(self.scene_path)
        # MuJoCo populates scene.camera[] as a stereo PAIR whenever
        # vis.global.ipd (interpupillary distance) is nonzero -- and it
        # defaults to 0.068. camera[0] below is the LEFT eye, offset
        # ipd/2=0.034m from the true monocular camera center along the
        # camera's local right-axis, even though renderer.render() itself
        # still renders the actual RGB/seg/depth from the true centered
        # pose (confirmed directly: projecting a known 3D point through
        # camera[0]'s pose vs. the true center against the ACTUAL rendered
        # pixel -- the true center matches to sub-pixel noise, camera[0]
        # is off by a full pixel). Zeroing ipd collapses the stereo pair to
        # a single, correct camera[0]==camera[1]==true-center pose, so the
        # cam_pos/cam_mat captured below (GroundTruth's own camera
        # extrinsics, consumed by every reconstruct.py unprojection) are
        # no longer silently biased by half an eye-separation. Found via
        # collision's own GATE 2 null false-positive (2026-08): a reference
        # ensemble with a genuinely zero-variance axis was the first thing
        # to expose a bias every other scenario's own nonzero axis
        # variance had been quietly absorbing.
        model.vis.global_.ipd = 0.0
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=self.height, width=self.width)

        cam = mujoco.MjvCamera()
        cam.lookat = cam_kwargs["lookat"]
        cam.distance = cam_kwargs["distance"]
        cam.azimuth = cam_kwargs["azimuth"]
        cam.elevation = cam_kwargs["elevation"]
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = (realism != "off")

        # geom id -> index into traj.names (or -1 if the geom isn't one of
        # the tracked bodies -- floor/ramp/wall/legs are all static scenery
        # attached to the "world" body, never tracked).
        body_to_idx = {}
        for i, nm in enumerate(traj.names):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            body_to_idx[bid] = i
        geom_to_idx = np.array([body_to_idx.get(bid, -1) for bid in model.geom_bodyid])

        T, K = traj.T, traj.K
        rgb = np.zeros((T, self.height, self.width, 3), np.uint8)
        seg = np.full((T, self.height, self.width), -1, np.int32)
        depth = np.zeros((T, self.height, self.width), np.float32)
        cam_pos = cam_mat = None

        for i in range(T):
            for k in range(K):
                data.qpos[k * 7: k * 7 + 3] = traj.pos[i, k]
                data.qpos[k * 7 + 3: k * 7 + 7] = traj.quat[i, k]
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=cam)
            rgb[i] = renderer.render()

            if cam_pos is None:      # static camera -- capture pose once
                c = renderer.scene.camera[0]
                cam_pos = np.array(c.pos, dtype=np.float64)
                forward = np.array(c.forward, dtype=np.float64)
                up = np.array(c.up, dtype=np.float64)
                right = np.cross(forward, up)
                cam_mat = np.stack([right, up, -forward], axis=1)

            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=cam)
            geom_ids = renderer.render()[..., 0]
            valid = geom_ids >= 0
            seg[i][valid] = geom_to_idx[geom_ids[valid]]
            renderer.disable_segmentation_rendering()

            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=cam)
            depth[i] = renderer.render()
            renderer.disable_depth_rendering()

        renderer.close()

        fps = int(round(1.0 / traj.dt))
        frames = Frames(rgb, fps)
        gt = GroundTruth(seg, depth, cam_pos, cam_mat, float(model.vis.global_.fovy), list(traj.names))
        return frames, gt
