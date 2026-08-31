"""Sigma: the perturbation model.

Applied to the initial STATE only (position, velocity) -- never to the model
(mass, friction, gravity). lambda is a dimensionless scale, swept, never
fixed (AGENT.md 3.12).
"""
from __future__ import annotations
import numpy as np

POS_SIGMA = 0.05   # metres, at lambda = 1
VEL_SIGMA = 0.05   # metres/second, at lambda = 1


def sample_perturbation(rng, lam, nfree):
    """Returns {"dpos": (nfree, 3), "dvel": (nfree, 3)}, scaled by lam."""
    dpos = rng.normal(0.0, POS_SIGMA * lam, size=(nfree, 3))
    dvel = rng.normal(0.0, VEL_SIGMA * lam, size=(nfree, 3))
    return {"dpos": dpos, "dvel": dvel}


def sample_velocity_perturbation(rng, lam, nfree, axis=0,
                                  pos_sigma=POS_SIGMA, vel_sigma=VEL_SIGMA):
    """Perturbs position and velocity along ONE axis only; every other
    component stays exactly at the scene's nominal s_0.

    Isotropic 3D perturbation (sample_perturbation) is wrong for a scenario
    whose only physically-free variable is along-track position/speed (e.g.
    occlusion_corridor): a ball resting on a floor has an out-of-plane
    contact constraint that is unilateral, not symmetric, so a stray
    z-velocity/z-position offset doesn't cancel out on average -- it gets
    resolved as an actual bounce, which at lambda large enough to get useful
    along-track spread is large enough to visibly disturb the vertical axis
    too. That's not a controlled perturbation of speed anymore, it's an
    uncontrolled perturbation of speed AND bounce height at once, which is
    exactly the kind of cross-contamination isolated per-property scenarios
    (AGENT.md 3.6) are supposed to prevent. (Zeroing position entirely, not
    just the off-axis components, was tried first and lost most of the
    along-track spread -- starting position varies same as starting speed
    does, both are part of s_0 along the one free axis.)"""
    dpos = np.zeros((nfree, 3))
    dpos[:, axis] = rng.normal(0.0, pos_sigma * lam, size=nfree)
    dvel = np.zeros((nfree, 3))
    dvel[:, axis] = rng.normal(0.0, vel_sigma * lam, size=nfree)
    return {"dpos": dpos, "dvel": dvel}
