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
    model, data = load(scene_path)
    rng = np.random.default_rng(seed)
    names = body_names(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
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
    nfree = model.nq // 7
    if nfree:
        if spec.perturb_mode == "velocity_x_only":
            pert = sample_velocity_perturbation(rng, spec.lam, nfree)
        else:
            pert = sample_perturbation(rng, spec.lam, nfree)
        for b in range(nfree):
            data.qpos[b * 7: b * 7 + 3] += pert["dpos"][b]
            data.qvel[b * 6: b * 6 + 3] += pert["dvel"][b]
    mujoco.mj_forward(model, data)

    n = int(spec.horizon_s * spec.fps)
    sub = max(1, int(round((1.0 / spec.fps) / model.opt.timestep)))
    t = np.arange(n) / spec.fps
    pos = np.zeros((n, K, 3)); quat = np.zeros((n, K, 4))
    for i in range(n):
        pos[i] = data.xpos[ids]
        quat[i] = data.xquat[ids]
        for _ in range(sub):
            mujoco.mj_step(model, data)

    return Trajectory(t, pos, quat, np.ones((n, K), bool), names,
                      meta={"scene": spec.scene, "lam": spec.lam, "seed": seed})


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