"""adapters/wan.py -- Wan2.1 Image2Video generate_fn (AGENT.md M6), the
SECOND real model integrated into this project. Tests only the parts that
don't need a live GPU/Modal deployment -- `generate_fn` itself (the
`modal.Function.from_name(...).remote(...)` call) is NOT covered here,
same honest boundary as `test_cosmos_adapter.py`'s own coverage.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_make_wan_generate_fn_modal_requires_a_registered_scenario_prompt():
    from vitals.adapters import wan
    try:
        wan.make_wan_generate_fn_modal("not_a_real_scenario")
        assert False, "should reject a scenario with no frozen prompt"
    except KeyError:
        pass


def test_wan_and_cosmos_share_the_identical_prompt_text():
    """Input-normalization protocol discipline: two models scored on the
    SAME scenario must see the IDENTICAL prompt wording, or a comparison
    between them isn't actually apples to apples. Both modules import
    from the same scenario_prompts.SCENARIO_PROMPTS -- this test would
    catch either one accidentally shadowing it with a local override."""
    from vitals.adapters import wan, cosmos
    assert wan.SCENARIO_PROMPTS is cosmos.SCENARIO_PROMPTS
    for scenario, prompt in wan.SCENARIO_PROMPTS.items():
        assert cosmos.SCENARIO_PROMPTS[scenario] == prompt


def test_wan_reuses_the_shared_resample_function_not_a_local_copy():
    """Same drift-prevention discipline as scene_geometry.py's own wall-
    bounds regression test -- Wan and Cosmos must resample identically,
    not via two independently-maintained copies of the same logic."""
    from vitals.adapters import wan, video_utils
    assert wan.resample_to_target is video_utils.resample_to_target


if __name__ == "__main__":
    test_make_wan_generate_fn_modal_requires_a_registered_scenario_prompt()
    print("PASS  test_make_wan_generate_fn_modal_requires_a_registered_scenario_prompt")
    test_wan_and_cosmos_share_the_identical_prompt_text()
    print("PASS  test_wan_and_cosmos_share_the_identical_prompt_text")
    test_wan_reuses_the_shared_resample_function_not_a_local_copy()
    print("PASS  test_wan_reuses_the_shared_resample_function_not_a_local_copy")
