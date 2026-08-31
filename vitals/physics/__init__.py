"""make_backend("synthetic" | "mujoco") -> roll(spec, seed) -> Trajectory.

The one place that knows both backends exist. Everything downstream calls
roll(spec, seed) and never imports synthetic.py / synthetic_corridor.py /
runner.py directly, so swapping backends touches nothing but this factory
(README design decision 1).

For "mujoco", `scene` is the actual .xml path. For "synthetic" there's no
file to point at, but there are two distinct scenarios (ramp_descent,
occlusion_corridor -- deliberately separate, see AGENT.md 3.6), so `scene`
doubles as a scenario name there: pass "occlusion_corridor" (or any string
containing "corridor") to get that one; omit it or pass anything else for
the ramp_descent default.
"""
from __future__ import annotations


def make_backend(name, scene=None):
    if name == "synthetic":
        if scene and "corridor" in scene:
            from . import synthetic_corridor
            return lambda spec, seed: synthetic_corridor.rollout(spec, seed)
        from . import synthetic
        return lambda spec, seed: synthetic.rollout(spec, seed)
    if name == "mujoco":
        if not scene:
            raise ValueError("mujoco backend requires scene=<path to .xml>")
        from . import runner
        return lambda spec, seed: runner.rollout(spec, seed, scene)
    raise ValueError(f"unknown backend: {name!r}")
