"""Mutants: trajectories with known defects at known times.

This is the validation engine. Because a mutant corrupts the STATE before
rendering, the resulting video is photometrically identical to a real render
except for the injected defect -- so detector sensitivity, specificity and
event-time bias are all measurable without any world model.

Every mutant declares (risk_expected, t_star) so scoring is automatic.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..types import Trajectory


@dataclass
class Mutant:
    name: str
    traj: Trajectory
    risk_expected: str | None   # None => nothing should fire
    t_star: float | None
    severity: float = 1.0


def null(traj, **kw):
    return Mutant("null", traj.copy(), None, None)


def vanish(traj, t_star=1.0, obj=0, **kw):
    out = traj.copy()
    i = int(t_star / traj.dt)
    out.present[i:, obj] = False
    out.pos[i:, obj] = np.nan
    return Mutant("vanish", out, "R2", t_star)


def duplicate(traj, t_star=1.0, obj=0, offset=0.15, **kw):
    """Clone an object -- existence failure in the other direction."""
    out = traj.copy()
    i = int(t_star / traj.dt)
    out.pos = np.concatenate([out.pos, out.pos[:, obj:obj + 1] + offset], axis=1)
    out.quat = np.concatenate([out.quat, out.quat[:, obj:obj + 1]], axis=1)
    pres = np.zeros((out.T, 1), bool); pres[i:] = True
    out.present = np.concatenate([out.present, pres], axis=1)
    out.names = list(out.names) + [out.names[obj] + "_clone"]
    return Mutant("duplicate", out, "R2", t_star)


def velocity_freeze(traj, t_star=1.0, obj=0, **kw):
    """Hold velocity constant after t* -- physics stops, object drifts."""
    out = traj.copy()
    i = int(t_star / traj.dt)
    if i < 2:
        return Mutant("velocity_freeze", out, "R5", t_star)
    v = (out.pos[i, obj] - out.pos[i - 1, obj]) / out.dt
    for k in range(i, out.T):
        out.pos[k, obj] = out.pos[i, obj] + v * (k - i) * out.dt
    return Mutant("velocity_freeze", out, "R5", t_star)


def wrong_gravity(traj, factor=0.5, obj=0, **kw):
    """Re-integrate vertical motion under the wrong g. Diverges gradually."""
    out = traj.copy()
    z = out.pos[:, obj, 2].copy()   # index 2 is vertical -- world is Z-up (AGENT.md 6.1)
    base = z[0]
    out.pos[:, obj, 2] = base + (z - base) * factor
    return Mutant("wrong_gravity", out, "R5", 0.0, severity=abs(1 - factor))


def teleport(traj, t_star=1.0, obj=0, target=1, **kw):
    """Move an object to overlap another -- interpenetration."""
    out = traj.copy()
    i = int(t_star / traj.dt)
    out.pos[i:, obj] = out.pos[i:, target]
    return Mutant("teleport", out, "R4", t_star)


def jitter(traj, sigma=0.01, obj=0, **kw):
    """Additive noise -- graded severity, for ROC curves."""
    out = traj.copy()
    rng = np.random.default_rng(0)
    out.pos[:, obj] += rng.normal(0, sigma, out.pos[:, obj].shape)
    return Mutant("jitter", out, "R5", 0.0, severity=sigma)


LIBRARY = {
    "null": null, "vanish": vanish, "duplicate": duplicate,
    "velocity_freeze": velocity_freeze, "wrong_gravity": wrong_gravity,
    "teleport": teleport, "jitter": jitter,
}