"""GATE 0 -- fork determinism. Requires MuJoCo; skipped (not failed) if it
isn't installed, per AGENT.md 7 (M0): P5/causal-consistency is unmeasurable
without it, but that only matters once P5 is in scope (it currently isn't).

Checked against every scene, not just one -- each scene is an independent
physical setup (AGENT.md 3.6) and determinism is a property of the engine +
scene combination, not something that transfers from one scene to another.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SCENES_DIR = pathlib.Path(__file__).resolve().parents[1] / "scenes"
SCENES = {
    "ramp_descent": str(SCENES_DIR / "ramp_descent.xml"),
    "occlusion_corridor": str(SCENES_DIR / "occlusion_corridor.xml"),
}


def test_gate_0_fork_determinism():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIP  mujoco not installed")
        return
    from vitals.physics.runner import gate_fork_determinism
    for name, scene in SCENES.items():
        assert gate_fork_determinism(scene) is True, f"GATE 0 failed for {name}"


if __name__ == "__main__":
    test_gate_0_fork_determinism()
    print("PASS  test_gate_0_fork_determinism, all scenes (see SKIP above if mujoco absent)")
