"""Core data types. Trajectory is the primary artifact in the whole system.

Everything upstream of rendering speaks Trajectory. Mutants corrupt a
Trajectory. Detectors consume Trajectories. Later, the measurement operator
Phi produces a Trajectory *from video* -- which is what makes the L0 (state)
and L1 (pixel) paths directly comparable on identical inputs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Trajectory:
    """Structured physical state over time.

    t:       (T,)        seconds
    pos:     (T, K, 3)   object positions, metres
    quat:    (T, K, 4)   object orientations, wxyz
    present: (T, K) bool object exists in the world (NOT "is visible")
    names:   K object names
    """
    t: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    present: np.ndarray
    names: list
    meta: dict = field(default_factory=dict)

    @property
    def T(self): return len(self.t)

    @property
    def K(self): return len(self.names)

    @property
    def dt(self): return float(self.t[1] - self.t[0])

    def copy(self):
        return Trajectory(self.t.copy(), self.pos.copy(), self.quat.copy(),
                          self.present.copy(), list(self.names), dict(self.meta))

    def index(self, name): return self.names.index(name)


@dataclass
class Event:
    """Terminal event for one rollout: (time, risk) or right-censored."""
    time: float
    risk: str | None
    censored: bool

    @classmethod
    def censor(cls, t_max):
        return cls(time=t_max, risk=None, censored=True)


@dataclass
class EpisodeSpec:
    """Declarative manifest. The corpus is regenerable from these alone."""
    name: str
    scene: str
    target_property: str     # pi* -- which necessary consequence this tests
    band: str                # I | II | III (predictability)
    lam: float               # perturbation scale
    n_reference: int         # M
    horizon_s: float         # T_max
    fps: int = 30
    seed: int = 0
    intervention: dict | None = None
    # "full" = isotropic 3D Sigma (physics/perturb.py::sample_perturbation).
    # "velocity_x_only" = physics/perturb.py::sample_velocity_perturbation --
    # for scenarios whose only physically-free variable is along-track speed
    # (e.g. occlusion_corridor); see that function's docstring for why
    # isotropic perturbation is actively wrong there, not just unnecessary.
    perturb_mode: str = "full"