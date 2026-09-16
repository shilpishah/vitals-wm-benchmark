"""WorldModel protocol + degenerate baselines.

A WorldModel takes a conditioning PREFIX (drawn from a held-out realization,
never from the reference ensemble R -- AGENT.md 3.7) and predicts N
candidate CONTINUATIONS, picking up where the prefix left off. This mirrors
how a real video model actually works -- it doesn't get handed a full
trajectory with placeholder future frames, it generates forward from what
it saw. `concat_trajectory` stitches a continuation back onto its prefix
into one full-length Trajectory, which is what detect/ actually scores.

N>1 exists for the distributional calibration design (P6, M7):
`stats/scoring.py` (built 2026-09-11: fair_crps / fair_energy_score /
fair_energy_distance / rank_histogram / spread_skill, `score_episode`)
scores an ensemble of candidate samples against the held-out truth and
the reference distribution. No population script asks for n_samples > 1
yet -- that is a per-model GPU-cost decision, not a code gap. The event/survival design (what GATE 3 and run_eval.py check)
scores exactly ONE candidate per episode, per AGENT.md's core loop --
pooling multiple correlated samples from the same episode into that
pipeline would violate 3.10 (bootstrap over episodes, never over rollouts).
CopyLastState and ConstantVelocity are deterministic, so their N samples
are identical for now; the interface is shaped for when that stops being
true without a breaking change.

CopyLastState and ConstantVelocity are not meant to be good predictors.
They're the sanity check (AGENT.md M6, GATE 3): ConstantVelocity should show
a long validity interval on close-to-constant-velocity motion and a short
one the instant true dynamics do something nonlinear (a ramp transition, a
bounce). If it doesn't, the metric is wrong, not the baseline.
"""
from __future__ import annotations
from typing import Protocol
import numpy as np
from ..types import Trajectory


class WorldModel(Protocol):
    name: str

    def predict(self, conditioning: Trajectory, horizon_s: float,
                n_samples: int) -> list[Trajectory]:
        """conditioning: the observed prefix (T_c frames). Returns
        n_samples candidate continuations, each `horizon_s` long at
        conditioning's fps, starting one frame after conditioning ends."""
        ...


def concat_trajectory(prefix: Trajectory, continuation: Trajectory) -> Trajectory:
    """Stitches a conditioning prefix and a predicted continuation into one
    full-length Trajectory with a continuous time axis, for scoring by
    detect/ exactly like any reference rollout. Assumes both share fps and
    object identity (K, names) -- true for every WorldModel in this repo,
    since none of them change what objects exist."""
    pos = np.concatenate([prefix.pos, continuation.pos], axis=0)
    quat = np.concatenate([prefix.quat, continuation.quat], axis=0)
    present = np.concatenate([prefix.present, continuation.present], axis=0)
    t = np.arange(pos.shape[0]) * prefix.dt
    return Trajectory(t, pos, quat, present, list(prefix.names), dict(prefix.meta))


def prefix_of(traj: Trajectory, t_c_frames: int) -> Trajectory:
    """The first t_c_frames of traj, as its own standalone Trajectory."""
    return Trajectory(traj.t[:t_c_frames].copy(), traj.pos[:t_c_frames].copy(),
                      traj.quat[:t_c_frames].copy(), traj.present[:t_c_frames].copy(),
                      list(traj.names), dict(traj.meta))


class CopyLastState:
    """Predicts nothing changes after conditioning ends. The floor: any
    model that can't beat this isn't modeling dynamics at all."""
    name = "copy_last_state"

    def predict(self, conditioning, horizon_s, n_samples):
        n = max(1, int(round(horizon_s / conditioning.dt)))
        pos = np.tile(conditioning.pos[-1], (n, 1, 1))
        quat = np.tile(conditioning.quat[-1], (n, 1, 1))
        present = np.tile(conditioning.present[-1], (n, 1))
        t = np.arange(n) * conditioning.dt
        cont = Trajectory(t, pos, quat, present, list(conditioning.names),
                          {"adapter": self.name})
        return [cont.copy() for _ in range(n_samples)]


class ConstantVelocity:
    """Predicts linear extrapolation from the last observed velocity.
    Ignores gravity, friction, contact entirely -- exactly the GATE 3
    discriminator: fine while true motion is close to constant-velocity,
    bad the instant it isn't."""
    name = "constant_velocity"

    def predict(self, conditioning, horizon_s, n_samples):
        n = max(1, int(round(horizon_s / conditioning.dt)))
        dt = conditioning.dt
        v = (conditioning.pos[-1] - conditioning.pos[-2]) / dt      # (K, 3)
        steps = (np.arange(1, n + 1) * dt)[:, None, None]           # (n, 1, 1)
        pos = conditioning.pos[-1][None, :, :] + v[None, :, :] * steps
        quat = np.tile(conditioning.quat[-1], (n, 1, 1))
        present = np.tile(conditioning.present[-1], (n, 1))
        t = np.arange(n) * dt
        cont = Trajectory(t, pos, quat, present, list(conditioning.names),
                          {"adapter": self.name})
        return [cont.copy() for _ in range(n_samples)]


ADAPTERS = {
    "copy_last_state": CopyLastState,
    "constant_velocity": ConstantVelocity,
}
