"""adapters/cogvideox.py -- CogVideoX1.5 generate_fn. Same honest boundary
as test_wan_adapter.py: everything except the live Modal call."""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest


def test_requires_a_registered_scenario_prompt():
    from vitals.adapters import cogvideox
    with pytest.raises(KeyError):
        cogvideox.make_cogvideox_generate_fn_modal("not_a_real_scenario")


def test_shares_the_identical_prompt_table_and_resampler():
    from vitals.adapters import cogvideox, wan, video_utils
    assert cogvideox.SCENARIO_PROMPTS is wan.SCENARIO_PROMPTS
    assert cogvideox.resample_to_target is video_utils.resample_to_target


def test_requested_native_frames_are_valid_16k_plus_1_within_the_card_bounds():
    from vitals.adapters.cogvideox import requested_native_frames as r
    for n in (1, 60, 120, 300, 10_000):
        got = r(n)
        assert (got - 1) % 16 == 0, f"{got} is not 16k+1"
        assert 81 <= got <= 161
    assert r(60) == 81        # 2s needs ~33 native -> floor 81 (card's quality floor)
    assert r(240) == 129      # 8s needs 129 native = 16*8+1, exactly
    assert r(10_000) == 161   # cap


def test_registered_restricted_and_opt_in_only():
    """The CogVideoX license's commercial-registration/visit-cap clauses
    put it in the same class as hunyuan: never enabled by default."""
    from vitals.adapters import MODEL_REGISTRY
    assert MODEL_REGISTRY["cogvideox15"]["restricted"] is True
    assert "registering" in MODEL_REGISTRY["cogvideox15"]["restriction_note"]
    # REAL_MODEL_BACKENDS is computed at import from the env var -- verify the gate
    # semantics directly rather than depending on this process's env.
    enabled = {n for n in os.environ.get("VITALS_ENABLE_RESTRICTED_MODELS", "").split(",") if n}
    from vitals.adapters import REAL_MODEL_BACKENDS
    assert ("cogvideox15" in REAL_MODEL_BACKENDS) == ("cogvideox15" in enabled)
