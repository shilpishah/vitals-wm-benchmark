"""scripts/render_survival_report.py -- specifically its category_for()
logic (AGENT.md M6, 2026-08). Regression guard for a real bug: adding
Wan2.1 as a second real model initially got silently miscategorized as a
Baseline, because category_for() hardcoded `backend == "cosmos"` as the
only non-Truth category instead of checking the shared REAL_MODEL_
BACKENDS registry (vitals/adapters/__init__.py).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_every_real_model_backend_categorizes_as_C_not_B():
    """The actual bug: a real model added to REAL_MODEL_BACKENDS after
    the fact must categorize as (C), not silently fall through to (B)."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import render_survival_report as rsr
    from vitals.adapters import REAL_MODEL_BACKENDS

    assert len(REAL_MODEL_BACKENDS) >= 2, "expected at least cosmos and wan registered"
    for backend in REAL_MODEL_BACKENDS:
        assert rsr.category_for(backend) == "C", f"{backend!r} (a real model) must categorize as (C), not fall through to (B)"


def test_mujoco_categorizes_as_T():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import render_survival_report as rsr
    assert rsr.category_for("mujoco") == "T"


def test_unregistered_backend_categorizes_as_B():
    """A baseline (or anything not in REAL_MODEL_BACKENDS/not "mujoco")
    must fall through to (B) -- confirms the fallback direction still
    works, not just the real-model direction the bug was in."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import render_survival_report as rsr
    assert rsr.category_for("constant_velocity") == "B"
    assert rsr.category_for("copy_last_state") == "B"


if __name__ == "__main__":
    test_every_real_model_backend_categorizes_as_C_not_B()
    print("PASS  test_every_real_model_backend_categorizes_as_C_not_B")
    test_mujoco_categorizes_as_T()
    print("PASS  test_mujoco_categorizes_as_T")
    test_unregistered_backend_categorizes_as_B()
    print("PASS  test_unregistered_backend_categorizes_as_B")
