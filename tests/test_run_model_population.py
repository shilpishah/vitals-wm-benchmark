"""scripts/run_model_population.py -- resolve_episode_timeout_s only (pure
function, no GPU/network). Extracted 2026-08 after a real incident: a flat
600s default for --episode-timeout-s was comfortably above Cosmos's own
observed latency but SHORTER than Wan's own deployed function's 1200s
server-side timeout, so the client kept abandoning still-legitimately-
running Wan calls and firing overlapping duplicate attempts -- which is
what actually produced 'Function call was cancelled' failures on every
seed of two real re-runs, not a genuine hang."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import run_model_population as rmp


def test_cosmos_default_is_600_when_not_overridden():
    assert rmp.resolve_episode_timeout_s("cosmos", None) == 600


def test_wan_default_is_higher_than_its_own_server_side_timeout():
    # remote/modal_app_wan.py's own deployed `image2video` has timeout=1200
    # -- the client default MUST clear that with margin, or the exact
    # mismatch this function was built to fix recurs.
    assert rmp.resolve_episode_timeout_s("wan", None) > 1200


def test_cosmos14b_default_is_also_higher_than_the_shared_1200s_server_ceiling():
    # cosmos14b shares Cosmos-2B's own deployed `video2world` function
    # (remote/modal_app_cosmos.py), which has the SAME timeout=1200 --
    # 14B just runs much slower underneath (real published benchmarks:
    # ~3.6-4.7x 2B's own latency), so it needs the same Wan-style margin,
    # not 2B's untested-at-this-scale 600s default.
    assert rmp.resolve_episode_timeout_s("cosmos14b", None) > 1200


def test_cosmos720p_default_is_also_higher_than_the_shared_1200s_server_ceiling():
    # Same reasoning as cosmos14b, applied to resolution instead of model
    # size: real published non-NATTEN 720p/2B timing (228.8-378.5s) is
    # the same order of magnitude as 14B's own 480p latency, so it gets
    # the identical margin, not 480p/2B's own narrower-tested 600s.
    assert rmp.resolve_episode_timeout_s("cosmos720p", None) > 1200


def test_explicit_value_always_wins_over_any_model_default():
    assert rmp.resolve_episode_timeout_s("wan", 42.0) == 42.0
    assert rmp.resolve_episode_timeout_s("cosmos", 0.0) == 0.0   # 0 = "disable", must not be treated as unset


def test_unknown_model_falls_back_to_600_not_a_crash():
    assert rmp.resolve_episode_timeout_s("some_future_model", None) == 600


if __name__ == "__main__":
    import inspect
    fns = [obj for name, obj in sorted(globals().items())
           if name.startswith("test_") and inspect.isfunction(obj)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
