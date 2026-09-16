"""adapters/wan22.py + adapters/ltx.py -- same honest boundary as
test_wan_adapter.py: everything except the live Modal call."""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest


def test_both_require_a_registered_scenario_prompt():
    from vitals.adapters import wan22, ltx
    with pytest.raises(KeyError):
        wan22.make_wan22_generate_fn_modal("not_a_real_scenario")
    with pytest.raises(KeyError):
        ltx.make_ltx_generate_fn_modal("not_a_real_scenario")


def test_both_share_the_identical_prompt_table_and_resampler():
    from vitals.adapters import wan22, ltx, wan, video_utils
    for m in (wan22, ltx):
        assert m.SCENARIO_PROMPTS is wan.SCENARIO_PROMPTS
        assert m.resample_to_target is video_utils.resample_to_target


def test_ltx_frames_are_on_the_8k_plus_1_grid_within_bounds():
    from vitals.adapters.ltx import requested_native_frames as r
    for n in (1, 60, 120, 240, 10_000):
        got = r(n)
        assert (got - 1) % 8 == 0 and 25 <= got <= 121
    assert r(60) == 49    # 2s @30 -> 48 native +1 = 49 exactly
    assert r(120) == 97   # 4s
    assert r(10_000) == 121


def test_registry_entries_and_gates():
    from vitals.adapters import MODEL_REGISTRY, REAL_MODEL_BACKENDS
    assert MODEL_REGISTRY["wan22"]["restricted"] is False and "wan22" in REAL_MODEL_BACKENDS
    assert MODEL_REGISTRY["ltx23"]["restricted"] is True
    assert "$10,000,000" in MODEL_REGISTRY["ltx23"]["restriction_note"]   # the verified Sec. 2.1 threshold
    enabled = {n for n in os.environ.get("VITALS_ENABLE_RESTRICTED_MODELS", "").split(",") if n}
    assert ("ltx23" in REAL_MODEL_BACKENDS) == ("ltx23" in enabled)
