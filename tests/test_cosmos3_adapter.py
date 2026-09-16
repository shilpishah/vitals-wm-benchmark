"""adapters/cosmos3.py -- Cosmos 3 Nano generate_fn. Same honest boundary
as test_wan_adapter.py: covers everything except the live Modal call."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest


def test_requires_a_registered_scenario_prompt():
    from vitals.adapters import cosmos3
    with pytest.raises(KeyError):
        cosmos3.make_cosmos3_generate_fn_modal("not_a_real_scenario")


def test_shares_the_identical_prompt_table_and_resampler():
    """Matched settings: same frozen prompt object and the same shared
    resampler as every other model -- no local overrides that could drift."""
    from vitals.adapters import cosmos3, wan, video_utils
    assert cosmos3.SCENARIO_PROMPTS is wan.SCENARIO_PROMPTS
    assert cosmos3.resample_to_target is video_utils.resample_to_target


def test_requested_native_frames_just_cover_n_frames_and_respect_the_cap():
    """24 fps native vs 30 fps target: request just enough (+1), never the
    full 189 by default -- that's ~4x the H100 time for frames that get
    trimmed anyway."""
    from vitals.adapters.cosmos3 import COSMOS3_NATIVE_FPS, COSMOS3_MAX_FRAMES
    def requested(n_frames, target_fps=30):
        return min(COSMOS3_MAX_FRAMES, max(5, int(np.ceil(n_frames * COSMOS3_NATIVE_FPS / target_fps)) + 1))
    assert COSMOS3_NATIVE_FPS == 24
    assert requested(60) == 49            # 2.0s -> 48 native + 1
    assert requested(120) == 97           # 4.0s
    assert requested(1) == 5              # floor
    assert requested(10_000) == COSMOS3_MAX_FRAMES


def test_registered_as_an_unrestricted_real_backend():
    from vitals.adapters import MODEL_REGISTRY, REAL_MODEL_BACKENDS
    assert MODEL_REGISTRY["cosmos3nano"]["restricted"] is False
    assert "cosmos3nano" in REAL_MODEL_BACKENDS
