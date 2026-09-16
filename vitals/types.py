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
    # Soft-body modality (AGENT.md M9, 2026-09-13): a deformable body has a
    # configuration, not a pose. `pos` stays its centroid (the centre of
    # mass obeys the same ballistics, so R1/R3/R4 read it unchanged) and
    # `shape` carries per-object SHAPE descriptors, (T, K, S), None for
    # every rigid scenario -- no rigid code path reads it. The descriptor
    # set is fixed by vitals.physics.softbody.SHAPE_DESCRIPTORS so the
    # state-space (vertex) and pixel (mask) paths compute the same S
    # quantities. The raw vertex cloud, when kept, lives in
    # meta["flex_vertices"][name] as (T, nvert, 3) -- it is what the
    # renderer needs to draw the body and what the mutants deform.
    shape: np.ndarray | None = None

    @property
    def T(self): return len(self.t)

    @property
    def K(self): return len(self.names)

    @property
    def dt(self): return float(self.t[1] - self.t[0])

    def copy(self):
        return Trajectory(self.t.copy(), self.pos.copy(), self.quat.copy(),
                          self.present.copy(), list(self.names),
                          {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.meta.items()},
                          None if self.shape is None else self.shape.copy())

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
    perturb_mode: str = "full"    # "full" | "velocity_x_only" | either + "_obj0" (perturb only the first free body; physics/runner.py)
    # 2026-09, billiards/multi-collision scoping -- a SEPARATE, explicit,
    # pre-registered opt-in for real-model (L1) scoring to reconstruct
    # EVERY object in a K>1 scene (up to K=3), not just the primary.
    # Deliberately NOT inferred from target_property or K (see `adapters.
    # video_model.VideoWorldModel`'s own docstring for why reusing P3's
    # own trigger, or gating on K alone, would both be the kind of silent
    # scope-expansion this project's own R2 gating discipline already
    # warns against). False for every existing manifest -- a strict
    # no-op, including `collision.yaml`, which deliberately stays
    # single-object per its own module docstring.
    track_all_objects: bool = False