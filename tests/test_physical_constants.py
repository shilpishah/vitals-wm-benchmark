"""Physical-constant recovery (detect/physical_constants.py). Requires
MuJoCo; skipped (not failed) if it isn't installed, same convention as
test_gates.py.

Checks the two things that matter about this diagnostic: (1) the fit is
tight and reproducible across the reference ensemble when given the
validated transient-clearing window (fit_corridor_deceleration), and
(2) it actually catches a corrupted trajectory -- a diagnostic that can't
fail on bad input isn't a diagnostic (AGENT.md 8.1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SCENE = str(pathlib.Path(__file__).resolve().parents[1] / "scenes" / "occlusion_corridor.xml")


def _make_refs(n=10):
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    spec = EpisodeSpec(name="occlusion_corridor", scene=SCENE, target_property="P2", band="I",
                        lam=6.0, n_reference=n, horizon_s=8.0, fps=30, perturb_mode="velocity_x_only")
    rollout = make_backend("mujoco", scene=SCENE)
    return [rollout(spec, seed) for seed in range(n)], rollout, spec


def test_corridor_deceleration_recovery_is_tight():
    from vitals.detect.physical_constants import fit_corridor_deceleration, reference_constant_band
    refs, _, _ = _make_refs(10)
    a_values = reference_constant_band(refs, fit_corridor_deceleration)
    assert len(a_values) == len(refs), "every reference should fit cleanly (R^2 >= 0.95) in the validated window"
    cv = a_values.std() / a_values.mean()
    assert cv < 0.01, f"reference-ensemble fit should cluster tightly (CV<1%), got CV={cv:.4f}"


def test_corridor_deceleration_flags_corrupted_trajectory():
    import copy
    from vitals.detect.physical_constants import (
        fit_corridor_deceleration, reference_constant_band, constant_recovery_report,
    )
    refs, rollout, spec = _make_refs(10)
    a_values = reference_constant_band(refs, fit_corridor_deceleration)

    held_out = rollout(spec, 99999)
    a_ok, _ = fit_corridor_deceleration(held_out)
    report_ok = constant_recovery_report(a_ok, a_values)
    assert abs(report_ok["z_score"]) < 4, "an ordinary held-out episode should sit inside the reference band"

    corrupted = copy.deepcopy(held_out)
    x0 = corrupted.pos[0, 0, 0]
    corrupted.pos[:, 0, 0] = x0 + (corrupted.pos[:, 0, 0] - x0) * 0.6
    a_bad, _ = fit_corridor_deceleration(corrupted)
    report_bad = constant_recovery_report(a_bad, a_values)
    assert abs(report_bad["z_score"]) > 10, "an implied-physics-changing corruption must be flagged, not missed"


if __name__ == "__main__":
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIP  mujoco not installed")
    else:
        test_corridor_deceleration_recovery_is_tight()
        print("PASS  test_corridor_deceleration_recovery_is_tight")
        test_corridor_deceleration_flags_corrupted_trajectory()
        print("PASS  test_corridor_deceleration_flags_corrupted_trajectory")
