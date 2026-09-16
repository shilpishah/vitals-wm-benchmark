"""scripts/render_score_report.py -- the 2026-09 fifth-pass logic:
per-scenario supersession, baseline re-censoring at the model page's own
t_max, the `outlasts_all_baselines` semantics (regression guard for the
inverted `beats_all_baselines` flag), and the scored-channel denominator
for the failure-mode chart. Pure-logic tests on synthetic result dicts;
no PDF is rendered here.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import render_score_report as rsr


def _pop(t_max, times, censored=None, risks=None, with_lh=True, channels=("R1", "R3")):
    censored = censored or [False] * len(times)
    risks = risks or ["R3"] * len(times)
    from vitals.types import Event
    from vitals.stats.survival import validity_interval, termination_profile, kaplan_meier
    evs = [Event(time=t, risk=None if c else r, censored=c) for t, r, c in zip(times, risks, censored)]
    grid, surv = kaplan_meier(evs)
    d = dict(scenario="s", backend="b", t_max=t_max,
             thresholds_median={c: 1.0 for c in channels},
             survival=dict(n=len(evs), vi50=validity_interval(evs), vi50_ci95=[0, 0],
                           censoring_rate=sum(censored) / len(evs), termination_profile=termination_profile(evs),
                           kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
                           events=[dict(time=e.time, risk=e.risk, censored=e.censored) for e in evs]))
    if with_lh:
        d["long_horizon"] = dict(n=len(evs), vi50=validity_interval(evs), vi50_ci95=[0, 0],
                                 censoring_rate=sum(censored) / len(evs),
                                 kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
                                 events=[dict(time=e.time, censored=e.censored) for e in evs])
    return d


def test_outlasts_is_true_only_when_model_fails_later():
    """The actual bug: the old flag was True when the model's VI50 was
    SMALLER (failed faster). 2.0 vs baselines 1.0/1.5 must be True; 1.0 vs
    2.0 must be False."""
    assert rsr.outlasts_all_baselines(2.0, [1.0, 1.5]) is True
    assert rsr.outlasts_all_baselines(1.0, [2.0, 1.5]) is False
    assert rsr.outlasts_all_baselines(1.5, [1.5]) is False          # strict
    inf = float("inf")
    assert rsr.outlasts_all_baselines(inf, [1.0, 2.0]) is True
    assert rsr.outlasts_all_baselines(1.0, [inf]) is False
    assert "undetermined" in rsr.outlasts_all_baselines(inf, [inf, 1.0])
    assert rsr.outlasts_all_baselines(1.0, []) is None


def test_recensor_truncates_and_respects_persistence_window():
    """A full-horizon baseline (t_max 6) re-censored at 3.0 with pi=0.3:
    events at 1.0 and 2.5 survive; 2.8 (inside the window that a
    truncated run could not have confirmed), 4.5 and the original censor
    all become censored at exactly 3.0. VI50/KM/profile are recomputed
    from the clipped events, the original t_max is remembered, and the
    input dict is not mutated."""
    d = _pop(6.0, [1.0, 2.5, 2.8, 4.5, 6.0], censored=[False, False, False, False, True],
             risks=["R3", "R1", "R3", "R3", None])
    before = d["survival"]["vi50"]
    out = rsr.recensor(d, 3.0, pi=0.3)
    assert d["t_max"] == 6.0 and d["survival"]["vi50"] == before, "input must not be mutated"
    assert out["t_max"] == 3.0 and out["recensored_from_t_max"] == 6.0
    evs = out["survival"]["events"]
    assert [e["censored"] for e in evs] == [False, False, True, True, True]
    assert all(e["time"] == 3.0 for e in evs if e["censored"])
    assert out["survival"]["censoring_rate"] == 3 / 5
    assert out["survival"]["termination_profile"] == {"R3": 0.5, "R1": 0.5}
    assert out["survival"]["vi50"] == float("inf")     # only 2 of 5 fired -> S never reaches 0.5
    # long_horizon block re-censored the same way (no risk field)
    assert [e["censored"] for e in out["long_horizon"]["events"]] == [False, False, True, True, True]
    assert "risk" not in out["long_horizon"]["events"][0]


def test_recensor_is_identity_when_already_short_enough():
    d = _pop(3.0, [1.0, 2.0])
    assert rsr.recensor(d, 3.0) is d
    assert rsr.recensor(d, 6.0) is d


def test_supersession_is_per_scenario():
    """cosmos is dropped ONLY where cosmos3nano has results for the same
    scenario; where it doesn't, the cosmos page stays."""
    registry = {"cosmos": dict(superseded_by="cosmos3nano"), "cosmos3nano": {}, "wan": dict(superseded_by="wan22")}
    all_results = {("A", "cosmos"): 1, ("A", "cosmos3nano"): 1, ("B", "cosmos"): 1, ("B", "wan"): 1}
    pages = sorted(all_results)
    kept, omitted = rsr.apply_supersession(pages, all_results, registry)
    assert ("A", "cosmos") not in kept and ("A", "cosmos3nano") in kept
    assert ("B", "cosmos") in kept and ("B", "wan") in kept
    assert omitted == [("A", "cosmos", "cosmos3nano")]


def test_registry_marks_old_versions_superseded():
    from vitals.adapters import MODEL_REGISTRY
    for old in ("cosmos", "cosmos14b", "cosmos720p"):
        assert MODEL_REGISTRY[old]["superseded_by"] == "cosmos3nano"
    assert MODEL_REGISTRY["wan"]["superseded_by"] == "wan22"
    for succ in ("cosmos3nano", "wan22"):
        assert succ in MODEL_REGISTRY and "superseded_by" not in MODEL_REGISTRY[succ]


def test_compute_stats_lists_scored_but_unfired_channels_and_missing_baselines():
    """R2 scored (in thresholds_median) but never fired must still be a
    chart category; an absent baseline must be reported, not dropped; a
    model without a long_horizon block reports lh=None."""
    from vitals.stats.survival import bootstrap_vi_delta
    from vitals.detect.events import Event
    model = _pop(3.0, [1.0] * 12, channels=("R1", "R2", "R3"), with_lh=False)
    cv = _pop(6.0, [1.0] * 12)
    all_results = {("s", "m"): model, ("s", "constant_velocity"): cv}
    st = rsr.compute_stats("s", "m", all_results, bootstrap_vi_delta, Event)
    assert st["channels"] == ["R1", "R2", "R3"]
    assert st["missing_baselines"] == ["copy_last_state"]
    assert st["lh"] is None                                          # model predates R4 -> "not computed" band
    assert [b for b, _ in st["lh_baselines"]] == ["constant_velocity"]  # baseline's own R4 still carried
    assert st["baselines"][0][1]["t_max"] == 3.0   # re-censored to the model page
    assert st["outlasts"] is False                  # 1.0 vs 1.0 is not strictly larger


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
