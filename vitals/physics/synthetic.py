"""Analytic ball-down-ramp trajectory generator -- the synthetic-backend
counterpart of scenes/ramp_descent.xml. Pure incline kinematics, deliberately
no occluder (see synthetic_corridor.py for that, kept as a separate module
for the same reason ramp_descent.xml and occlusion_corridor.xml are separate
scenes -- AGENT.md 3.6, each necessary property gets an experiment that
isolates it).

No MuJoCo dependency -- fast, deterministic, for exercising detect/ and
stats/ without an engine. Same contract as physics/runner.py
(spec, seed) -> Trajectory, and Sigma perturbs only the initial state, so
the two backends are interchangeable behind make_backend().

Includes horizontal (rolling) friction on the flat section, unlike an
earlier version of this scene -- without it the ball coasts forever and
velocity_freeze injected late reads as a no-op (AGENT.md 6.5).
"""
from __future__ import annotations
import numpy as np
from ..types import Trajectory
from .perturb import sample_perturbation

G = 9.81
RADIUS = 0.15
RAMP_ANGLE = np.radians(28.0)
RAMP_TOP_X = -3.6
RAMP_TOE_X = -0.4          # ramp -> flat transition, x >= this is flat ground
MU_RAMP = 0.05              # rolling friction on the incline
MU_FLAT = 0.35              # rolling friction on flat ground


def _ramp_surface_z(x):
    """Height of the ramp/floor surface (ball-centre resting height) at x."""
    if x >= RAMP_TOE_X:
        return RADIUS
    return (RAMP_TOE_X - x) * np.tan(RAMP_ANGLE) + RADIUS


def rollout(spec, seed):
    rng = np.random.default_rng(seed)
    pert = sample_perturbation(rng, spec.lam, nfree=1)

    p = np.array([RAMP_TOP_X, 0.0, 0.0]) + pert["dpos"][0]
    p[2] = _ramp_surface_z(p[0])
    v = pert["dvel"][0].copy()

    n = int(spec.horizon_s * spec.fps)
    sub = 10
    dt = (1.0 / spec.fps) / sub

    pos = np.zeros((n, 1, 3))
    quat = np.zeros((n, 1, 4)); quat[:, 0, 0] = 1.0

    for i in range(n):
        pos[i, 0] = p
        for _ in range(sub):
            on_ramp = p[0] < RAMP_TOE_X
            if on_ramp:
                a_x = G * np.sin(RAMP_ANGLE) - MU_RAMP * G * np.cos(RAMP_ANGLE)
            elif abs(v[0]) > 1e-4:
                a_x = -np.sign(v[0]) * MU_FLAT * G
            else:
                a_x = 0.0
            v[0] += a_x * dt
            p[0] += v[0] * dt
            floor_z = _ramp_surface_z(p[0])
            p[2] = floor_z
            v[2] = 0.0
            if not on_ramp and abs(v[0]) < 1e-3:
                v[0] = 0.0

    t = np.arange(n) / spec.fps
    return Trajectory(t, pos, quat, np.ones((n, 1), bool), ["ball"],
                      meta={"scene": "synthetic_ramp_descent", "lam": spec.lam, "seed": seed})
