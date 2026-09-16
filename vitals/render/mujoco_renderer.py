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


def camera_pose(scene_path, cam_kwargs, height=360, width=640):
    """(cam_pos, cam_mat, fovy_deg) for a free camera on this scene -- the
    SAME pose MujocoRenderer.render captures into GroundTruth, obtained
    the same way (renderer.scene.camera[0] after update_scene, ipd zeroed)
    so that state-space shape descriptors (softbody.attach_shape) are
    projected through exactly the camera the pixels come from. Costs one
    scene update, no frame is rendered."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    model.vis.global_.ipd = 0.0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat = cam_kwargs["lookat"]; cam.distance = cam_kwargs["distance"]
    cam.azimuth = cam_kwargs["azimuth"]; cam.elevation = cam_kwargs["elevation"]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    c = renderer.scene.camera[0]
    cam_pos = np.array(c.pos, dtype=np.float64)
    forward = np.array(c.forward, dtype=np.float64)
    up = np.array(c.up, dtype=np.float64)
    cam_mat = np.stack([np.cross(forward, up), up, -forward], axis=1)
    fovy = float(model.vis.global_.fovy)
    renderer.close()
    return cam_pos, cam_mat, fovy


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
            if bid >= 0:
                body_to_idx[bid] = i
        geom_to_idx = np.array([body_to_idx.get(bid, -1) for bid in model.geom_bodyid])
        # Soft bodies (AGENT.md M9, 2026-09-13): a flex is placed from its
        # kept vertex cloud (meta["flex_vertices"]), and its segmentation
        # pixels carry object type mjOBJ_FLEX with the FLEX id -- a separate
        # id space from geoms (found directly: a flex read as "geom 0").
        from ..physics import softbody as sb
        flex_info = {name: (vadr, vnum) for name, vadr, vnum, _ in sb.flex_objects(model)}
        flex_verts = traj.meta.get("flex_vertices", {})
        flexid_to_idx = {}
        for i, nm in enumerate(traj.names):
            fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, nm)
            if fid >= 0:
                flexid_to_idx[fid] = i
        rigid_slots = [(k, body_to_idx and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm))
                       for k, nm in enumerate(traj.names) if nm not in flex_info]
        # qpos address of each rigid object's free joint (no longer assumed
        # to be k*7: flex vertex joints may precede or interleave them).
        rigid_qadr = {}
        for k, bid in rigid_slots:
            if bid is None or bid < 0:
                continue
            j = int(model.body_jntadr[bid])
            if j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                rigid_qadr[k] = int(model.jnt_qposadr[j])

        T, K = traj.T, traj.K
        rgb = np.zeros((T, self.height, self.width, 3), np.uint8)
        seg = np.full((T, self.height, self.width), -1, np.int32)
        depth = np.zeros((T, self.height, self.width), np.float32)
        cam_pos = cam_mat = None

        for i in range(T):
            for k in range(K):
                nm = traj.names[k]
                if nm in flex_info:
                    if nm in flex_verts:
                        vadr, vnum = flex_info[nm]
                        sb.set_flex_vertices(model, data, vadr, vnum, flex_verts[nm][i])
                    continue
                qa = rigid_qadr.get(k, k * 7)
                data.qpos[qa: qa + 3] = traj.pos[i, k]
                data.qpos[qa + 3: qa + 7] = traj.quat[i, k]
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
            segraw = renderer.render()
            obj_ids, obj_types = segraw[..., 0], segraw[..., 1]
            is_geom = (obj_ids >= 0) & (obj_types == int(mujoco.mjtObj.mjOBJ_GEOM))
            seg[i][is_geom] = geom_to_idx[obj_ids[is_geom]]
            if flexid_to_idx:
                is_flex = (obj_ids >= 0) & (obj_types == int(mujoco.mjtObj.mjOBJ_FLEX))
                for fid, idx in flexid_to_idx.items():
                    seg[i][is_flex & (obj_ids == fid)] = idx
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
