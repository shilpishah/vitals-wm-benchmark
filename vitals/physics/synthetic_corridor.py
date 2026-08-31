"""Analytic ball-rolls-down-a-flat-corridor generator -- the synthetic-backend
counterpart of scenes/occlusion_corridor.xml. Pure existence/occlusion
scenario: flat ground, a baseline push velocity (mirroring the MJCF
<keyframe>), friction deceleration. Deliberately has no incline -- see
synthetic.py for that, kept as a separate module for the same reason the two
MuJoCo scenes are separate (AGENT.md 3.6: a model can nail one necessary
property and hallucinate the other, so each gets an experiment that isolates
it, never a combined one).

No MuJoCo dependency. Same (spec, seed) -> Trajectory contract as
physics/runner.py and physics/synthetic.py.
"""
from __future__ import annotations
import numpy as np
from ..types import Trajectory
from .perturb import sample_velocity_perturbation

G = 9.81
RADIUS = 0.15
V0 = 4.3    # baseline push, m/s -- matches occlusion_corridor.xml's <keyframe>
# Rolling friction picked to match MuJoCo's *measured* deceleration on this
# scene (~1.79 m/s^2, found empirically -- see occlusion_corridor.yaml),
# not derived analytically. The two backends aren't meant to agree bit for
# bit, only to produce the same qualitative behavior (stop-short /
# stop-behind / re-emerge all reachable from perturbation alone).
MU = 0.182


def rollout(spec, seed):
    rng = np.random.default_rng(seed)
    # velocity_x_only, not the isotropic 3D Sigma -- this scenario's only
    # physically-free variable is along-track speed (see perturb.py
    # docstring). Position is always exactly [0, 0, RADIUS].
    pert = sample_velocity_perturbation(rng, spec.lam, nfree=1)

    p = np.array([0.0, 0.0, RADIUS])
    v = np.array([V0, 0.0, 0.0]) + pert["dvel"][0]

    n = int(spec.horizon_s * spec.fps)
    sub = 10
    dt = (1.0 / spec.fps) / sub

    pos = np.zeros((n, 1, 3))
    quat = np.zeros((n, 1, 4)); quat[:, 0, 0] = 1.0

    for i in range(n):
        pos[i, 0] = p
        for _ in range(sub):
            if abs(v[0]) > 1e-4:
                v[0] += -np.sign(v[0]) * MU * G * dt
                if abs(v[0]) < 1e-3:
                    v[0] = 0.0
            p[0] += v[0] * dt

    t = np.arange(n) / spec.fps
    return Trajectory(t, pos, quat, np.ones((n, 1), bool), ["ball"],
                      meta={"scene": "synthetic_occlusion_corridor", "lam": spec.lam, "seed": seed})
