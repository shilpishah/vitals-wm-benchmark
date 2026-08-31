"""Renderer protocol + shared types (AGENT.md M3).

Defined here, not inside mujoco_renderer.py, so a different renderer
implementation can be dropped in later without callers changing anything
(AGENT.md M3: "Define it as a protocol from the start"). Nothing in
detect/ or the eval scripts should ever import MujocoRenderer directly --
only Renderer, Frames, GroundTruth.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from ..types import Trajectory


@dataclass
class Frames:
    """Raw rendered pixels -- never compressed video. Phi should never see
    encoding artifacts that didn't come from a real candidate model's own
    output (that's a real, interesting covariate for a video-generation
    model; it is not something the reference/measurement path should add
    on its own)."""
    rgb: np.ndarray   # (T, H, W, 3) uint8
    fps: int


@dataclass
class GroundTruth:
    """Everything a real deployment wouldn't have. Used ONLY for validating
    Phi (the mutant sweep rerun through video, AGENT.md M5/GATE 2) and for
    computing the instrument error floor (AGENT.md 3.1) -- never allowed
    into scoring itself. Phi is only ever allowed the frame-0 mask as a
    segmentation prompt; every other field here is post-hoc validation
    data, not something Phi gets to see while running."""
    segmentation: np.ndarray   # (T, H, W) int32; index into `names`, or -1 for background
    depth: np.ndarray          # (T, H, W) float32, metres
    cam_pos: np.ndarray        # (3,) metres, world space (static camera -- see MujocoRenderer)
    cam_mat: np.ndarray        # (3, 3) columns = [right, up, -forward] in world space
    fovy_deg: float
    names: list


class Renderer(Protocol):
    name: str
    deterministic: bool
    emits_ground_truth: bool

    def render(self, traj: Trajectory, cameras: list[dict] | None,
               realism: str) -> tuple[Frames, GroundTruth | None]:
        ...
