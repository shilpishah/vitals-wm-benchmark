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
    return Mutant("vanish", out, "R1", t_star)


def duplicate(traj, t_star=1.0, obj=0, offset=0.15, **kw):
    """Clone an object -- existence failure in the other direction."""
    out = traj.copy()
    i = int(t_star / traj.dt)
    out.pos = np.concatenate([out.pos, out.pos[:, obj:obj + 1] + offset], axis=1)
    out.quat = np.concatenate([out.quat, out.quat[:, obj:obj + 1]], axis=1)
    pres = np.zeros((out.T, 1), bool); pres[i:] = True
    out.present = np.concatenate([out.present, pres], axis=1)
    out.names = list(out.names) + [out.names[obj] + "_clone"]
    return Mutant("duplicate", out, "R1", t_star)


def _carry_cloud(out, src, obj):
    """Soft bodies (AGENT.md M9): a centroid mutant must move the vertex
    cloud with the centroid, or the rendered body and the scored row
    disagree. Rigid translation by (new - old) centroid, frame by frame;
    frames whose centroid became NaN keep NaN vertices (rendered parked,
    like a vanished rigid body). No-op for objects without a cloud."""
    verts = out.meta.get("flex_vertices", {})
    nm = out.names[obj]
    if nm not in verts:
        return
    V = verts[nm].copy()
    d = out.pos[:, obj] - src.pos[:, obj]
    V = V + d[:, None, :]
    out.meta["flex_vertices"] = dict(verts, **{nm: V})
    out.shape = None


def velocity_freeze(traj, t_star=1.0, obj=0, **kw):
    """Hold velocity constant after t* -- physics stops, object drifts."""
    out = traj.copy()
    i = int(t_star / traj.dt)
    if i < 2:
        return Mutant("velocity_freeze", out, "R3", t_star)
    v = (out.pos[i, obj] - out.pos[i - 1, obj]) / out.dt
    for k in range(i, out.T):
        out.pos[k, obj] = out.pos[i, obj] + v * (k - i) * out.dt
    _carry_cloud(out, traj, obj)
    return Mutant("velocity_freeze", out, "R3", t_star)


def wrong_gravity(traj, factor=0.5, obj=0, **kw):
    """Re-integrate vertical motion under the wrong g. Diverges gradually."""
    out = traj.copy()
    z = out.pos[:, obj, 2].copy()   # index 2 is vertical -- world is Z-up (AGENT.md 6.1)
    base = z[0]
    out.pos[:, obj, 2] = base + (z - base) * factor
    _carry_cloud(out, traj, obj)
    return Mutant("wrong_gravity", out, "R3", 0.0, severity=abs(1 - factor))


def teleport(traj, t_star=1.0, obj=0, target=1, **kw):
    """Move an object to overlap another -- interpenetration."""
    out = traj.copy()
    i = int(t_star / traj.dt)
    out.pos[i:, obj] = out.pos[i:, target]
    _carry_cloud(out, traj, obj)
    return Mutant("teleport", out, "R2", t_star)


def drift(traj, t_star=1.5, velocity=(0.3, 0.0, 0.0), obj=0, **kw):
    """From t*, the object moves at a constant extra velocity on top of its
    real motion -- a body at rest that starts sliding on its own, the
    failure a resting-phase scene can actually test (2026-09-14, soft_drop
    GATE 2: velocity_freeze has nothing to freeze once the body has
    settled -- it fired 1/8 at L0). Expected R3."""
    out = traj.copy()
    i = int(t_star / traj.dt)
    v = np.asarray(velocity, dtype=float)
    out.pos[i:, obj] += v[None, :] * ((np.arange(i, out.T) - i) * out.dt)[:, None]
    _carry_cloud(out, traj, obj)
    return Mutant("drift", out, "R3", t_star, severity=float(np.linalg.norm(v)))


def jitter(traj, sigma=0.01, obj=0, **kw):
    """Additive noise -- graded severity, for ROC curves."""
    out = traj.copy()
    rng = np.random.default_rng(0)
    out.pos[:, obj] += rng.normal(0, sigma, out.pos[:, obj].shape)
    _carry_cloud(out, traj, obj)
    return Mutant("jitter", out, "R3", 0.0, severity=sigma)


# --- Soft-body mutants (AGENT.md M9, 2026-09-14) ------------------------------
# These corrupt the VERTEX CLOUD in meta["flex_vertices"] (the soft body's
# configuration) and keep the centroid row consistent with it. They set
# `shape` back to None: shape descriptors are camera-dependent and are
# re-attached by the caller (softbody.attach_shape) after mutation, the
# same way a rigid mutant's state is only rendered afterwards. The rigid
# part of each frame's configuration is separated from its deformation by
# a Kabsch fit of the rest cloud (frame 0) onto the frame's cloud, so a
# mutant can scale the DEFORMATION alone while leaving the rigid motion
# (centroid + rotation) exactly as simulated -- that separation is what
# makes wrong_stiffness the attribution test (decision 4).

def _flex_name(traj, obj):
    verts = traj.meta.get("flex_vertices", {})
    nm = traj.names[obj]
    if nm not in verts:
        raise ValueError(f"object {nm!r} has no vertex cloud -- soft-body mutants need a flex object")
    return nm


def _kabsch(P, Q):
    """Rotation R (3x3) minimising |P @ R.T - Q| for centred clouds P, Q."""
    H = P.T @ Q
    U, _, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Wt.T @ U.T))
    return Wt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _rigid_and_deformation(V):
    """V (T, n, 3) -> centroids (T,3), rigid clouds (T,n,3) centred at the
    origin, deformations (T,n,3) = centred cloud - rigid cloud."""
    c = V.mean(axis=1)
    Vc = V - c[:, None]
    V0 = Vc[0]
    rigid = np.empty_like(Vc)
    for t in range(V.shape[0]):
        rigid[t] = V0 @ _kabsch(V0, Vc[t]).T
    return c, rigid, Vc - rigid


def _put_vertices(out, obj, V):
    nm = out.names[obj]
    out.meta["flex_vertices"] = dict(out.meta["flex_vertices"], **{nm: V})
    out.pos[:, obj] = V.mean(axis=1)
    out.shape = None


def wrong_stiffness(traj, factor=2.5, t_star=0.0, obj=0, **kw):
    """Deformation scaled by `factor` (>1 softer, <1 stiffer) from t*, rigid
    motion untouched -- "wrong material, right trajectory". Must fire a
    MATERIAL channel and never R3: that is the attribution test. Which
    material channel: measured on soft_drop's GATE 1 (2026-09-14), a
    too-soft body's first visible symptom is its SILHOUETTE -- it flattens
    into a wider, shorter shape whose projected area leaves the reference
    band (R6, 0.50s) a tenth of a second before its axis ratio does (R7,
    0.6s) -- so R6 is the registered expectation. frozen_deformation is
    the converse case (area within band, ratio out: R7)."""
    out = traj.copy()
    nm = _flex_name(traj, obj)
    V = traj.meta["flex_vertices"][nm].copy()
    c, rigid, d = _rigid_and_deformation(V)
    i = int(t_star / traj.dt)
    V[i:] = c[i:, None] + rigid[i:] + factor * d[i:]
    _put_vertices(out, obj, V)
    return Mutant("wrong_stiffness", out, "R6", t_star, severity=abs(np.log(factor)))


def frozen_deformation(traj, t_star=0.0, obj=0, **kw):
    """Shape held at REST (deformation zeroed) from t* while the centroid
    and rotation keep moving as simulated -- a body that never squashes."""
    out = traj.copy()
    nm = _flex_name(traj, obj)
    V = traj.meta["flex_vertices"][nm].copy()
    c, rigid, _ = _rigid_and_deformation(V)
    i = int(t_star / traj.dt)
    V[i:] = c[i:, None] + rigid[i:]
    _put_vertices(out, obj, V)
    return Mutant("frozen_deformation", out, "R7", t_star)


def wrong_damping(traj, factor=0.5, obj=0, **kw):
    """Post-contact vertical motion scaled about the contact height (the
    rebound is too small for factor<1, too large for >1); the whole cloud
    is shifted with the centroid so shape is untouched -- "wrong
    trajectory, right material": expected R3, not R6/R7."""
    out = traj.copy()
    nm = _flex_name(traj, obj)
    V = traj.meta["flex_vertices"][nm].copy()
    z = V.mean(axis=1)[:, 2]
    i = int(np.argmin(z))                       # first contact minimum
    dz = (z[i:] - z[i]) * (factor - 1.0)        # z' = z_i + (z - z_i) * factor
    V[i:, :, 2] += dz[:, None]
    _put_vertices(out, obj, V)
    return Mutant("wrong_damping", out, "R3", float(i * traj.dt), severity=abs(1 - factor))


def volume_leak(traj, rate=0.15, t_star=1.0, floor=0.4, obj=0, **kw):
    """Cloud scaled toward its centroid by (1 - rate*(t - t*)) per second,
    clipped at `floor` -- the body shrinks without moving: expected R6."""
    out = traj.copy()
    nm = _flex_name(traj, obj)
    V = traj.meta["flex_vertices"][nm].copy()
    c = V.mean(axis=1)
    s = np.clip(1.0 - rate * (traj.t - t_star), floor, 1.0)
    V = c[:, None] + (V - c[:, None]) * s[:, None, None]
    _put_vertices(out, obj, V)
    return Mutant("volume_leak", out, "R6", t_star, severity=rate)


LIBRARY = {
    "null": null, "vanish": vanish, "duplicate": duplicate,
    "velocity_freeze": velocity_freeze, "wrong_gravity": wrong_gravity,
    "teleport": teleport, "jitter": jitter, "drift": drift,
    "wrong_stiffness": wrong_stiffness, "frozen_deformation": frozen_deformation,
    "wrong_damping": wrong_damping, "volume_leak": volume_leak,
}