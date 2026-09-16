"""MuJoCo runner. Requires `pip install mujoco`.

Two guarantees this module exists to provide:
  1. rollout() applies Sigma to the initial STATE, never to the model.
  2. fork() gives a bit-identical branch point -- the hard requirement for
     the causal-consistency property. Verify with gate_fork_determinism()
     before building anything else on top.
"""
from __future__ import annotations
import copy
import numpy as np
from ..types import Trajectory
from .perturb import sample_perturbation, sample_velocity_perturbation


def load(scene_path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(scene_path)
    return model, mujoco.MjData(model)


def body_names(model):
    import mujoco
    out = []
    for i in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if nm and nm != "world":
            out.append(nm)
    return out


def rollout(spec, seed, scene_path):
    import mujoco
    from . import softbody as sb
    model, data = load(scene_path)
    rng = np.random.default_rng(seed)
    # Soft bodies (AGENT.md M9, 2026-09-13): a flex's vertices are bodies
    # too (`<flex>_<i>`, three slide joints each) and must NOT be tracked
    # as objects; the flex itself is ONE object whose row is its vertex
    # centroid, with the vertex cloud kept in meta["flex_vertices"].
    # Object order: rigid bodies (as before), then flexes.
    vert_bodies = sb.flex_vertex_body_ids(model)
    names = [n for n in body_names(model)
             if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) not in vert_bodies]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    flexes = sb.flex_objects(model)
    names = names + [f[0] for f in flexes]
    K = len(names)

    # s_0 comes from the scene's keyframe if it has one (e.g. occlusion_corridor's
    # "push", a nonzero baseline qvel -- there's no incline there to accelerate
    # the ball, so the nominal initial configuration has to supply the motion
    # instead). Falls back to the model's static defaults (zero velocity) when
    # there's no keyframe, so scenes like ramp_descent are unaffected.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    # Free bodies are counted from joint types, not `nq // 7` -- that rule
    # miscounts the moment a flex is present (its vertices add 3 dof each;
    # found 2026-09-13: 891 dof -> "127 free bodies"). Each perturbable
    # actor is either a rigid free body (qpos/qvel at its own joint
    # address) or a whole flex (the same displacement/velocity on every
    # vertex -- Sigma acts on the actor's initial state; its material is
    # part of s_0, AGENT.md M9 decision 5).
    free_joints = [int(j) for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)]
    n_actors = len(free_joints) + len(flexes)
    if n_actors:
        # "velocity_x_only_obj0" (2026-09-11, block_stack): perturb ONLY the
        # first actor (the launched ball). Found directly while tuning
        # the stacking scene: the default modes jostle EVERY free body's
        # initial x-position by up to ~0.2m at lam=2, which shoves a 0.12m
        # cube off its 0.15m perch at t=0 in ~1/3 of references -- the
        # ensemble's "ambiguity" was then about whether the tower fell on
        # its own, not about the ball. A stacked/resting configuration is
        # part of s_0, not of Sigma; only the actor is perturbed.
        obj0_only = spec.perturb_mode.endswith("_obj0")
        base_mode = spec.perturb_mode[:-5] if obj0_only else spec.perturb_mode
        if base_mode == "velocity_x_only":
            pert = sample_velocity_perturbation(rng, spec.lam, n_actors)
        else:
            pert = sample_perturbation(rng, spec.lam, n_actors)
        for a in range(1 if obj0_only else n_actors):
            if a < len(free_joints):
                j = free_joints[a]
                qa, da = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
                data.qpos[qa: qa + 3] += pert["dpos"][a]
                data.qvel[da: da + 3] += pert["dvel"][a]
            else:
                _, vadr, vnum, _ = flexes[a - len(free_joints)]
                sb.shift_flex(model, data, vadr, vnum, pert["dpos"][a], pert["dvel"][a])
    mujoco.mj_forward(model, data)

    n = int(spec.horizon_s * spec.fps)
    sub = max(1, int(round((1.0 / spec.fps) / model.opt.timestep)))
    t = np.arange(n) / spec.fps
    pos = np.zeros((n, K, 3)); quat = np.zeros((n, K, 4))
    nr = len(ids)
    flex_verts = {name: np.zeros((n, vnum, 3)) for name, _, vnum, _ in flexes}
    flex_extras = {name: np.zeros((n, 4)) for name, _, _, _ in flexes}   # [volume, ext_major, ext_mid, ext_minor]
    for i in range(n):
        pos[i, :nr] = data.xpos[ids]
        quat[i, :nr] = data.xquat[ids]
        for k, (name, vadr, vnum, elems) in enumerate(flexes):
            V = data.flexvert_xpos[vadr: vadr + vnum].copy()
            flex_verts[name][i] = V
            pos[i, nr + k] = V.mean(axis=0)
            quat[i, nr + k] = (1.0, 0.0, 0.0, 0.0)   # a configuration has no pose; identity placeholder
            flex_extras[name][i, 0] = sb.tet_volume(V, elems) if elems is not None else np.nan
            flex_extras[name][i, 1:] = sb.principal_extents(V)
        for _ in range(sub):
            mujoco.mj_step(model, data)

    meta = {"scene": spec.scene, "lam": spec.lam, "seed": seed}
    if flexes:
        meta["flex_vertices"] = flex_verts
        meta["flex_extras"] = flex_extras
    return Trajectory(t, pos, quat, np.ones((n, K), bool), names, meta=meta)


def fork(data):
    """Exact branch point. Deep-copy of mjData is the entire mechanism."""
    return copy.deepcopy(data)


def gate_fork_determinism(scene_path, n_steps=300):
    """GATE 0. Fork, step both branches with no intervention, assert identity.

    If this returns False, the causal-consistency property is unmeasurable in
    this engine and must be dropped from the design. Run this first.
    """
    import mujoco
    model, data = load(scene_path)
    for _ in range(n_steps // 2):
        mujoco.mj_step(model, data)
    a, b = fork(data), fork(data)
    for _ in range(n_steps // 2):
        mujoco.mj_step(model, a)
        mujoco.mj_step(model, b)
    return bool(np.array_equal(a.qpos, b.qpos) and np.array_equal(a.qvel, b.qvel))