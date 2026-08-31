"""M7 -- seed_test_retest + minimum_detectable_difference
(vitals/stats/survival.py). Pure numpy on synthetic Event lists, no
physics/GPU -- these are statistics-of-statistics tools, testable
independently of what produced the underlying events.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from vitals.types import Event
from vitals.stats.survival import seed_test_retest, minimum_detectable_difference


def _events(times, risk="R5"):
    return [Event(time=float(t), risk=risk, censored=False) for t in times]


def test_seed_test_retest_near_zero_for_a_deterministic_population():
    """Every episode fails at EXACTLY the same time (zero true variance,
    same idea as bootstrap_vi_delta's own zero-width-CI test for a
    deterministic vanish population) -- any two disjoint halves must
    report the SAME VI_50, so median_abs_delta must be ~0, not just
    small."""
    events = _events([1.5] * 40)
    result = seed_test_retest(events, n_splits=50, seed=0)
    assert result["median_abs_delta"] == 0.0
    assert result["n_valid_splits"] == 50


def test_seed_test_retest_spread_grows_with_underlying_noise():
    """A noisier population must show a LARGER split-to-split spread than
    a tighter one of the same size -- the whole point of this function is
    to be sensitive to that, not just return some positive number."""
    rng = np.random.default_rng(1)
    tight = _events(rng.normal(1.4, 0.02, 80))
    noisy = _events(rng.normal(1.4, 0.5, 80))
    r_tight = seed_test_retest(tight, n_splits=200, seed=0)
    r_noisy = seed_test_retest(noisy, n_splits=200, seed=0)
    assert r_noisy["median_abs_delta"] > r_tight["median_abs_delta"]


def test_seed_test_retest_rejects_too_few_events():
    try:
        seed_test_retest(_events([1.0, 2.0, 3.0]), n_splits=10)
        assert False, "must reject fewer than 4 events"
    except ValueError:
        pass


def test_mdd_power_increases_with_delta():
    """Empirical power must be monotonically non-decreasing (up to Monte
    Carlo noise) as the tested delta grows -- a bigger true difference
    should never become HARDER to detect than a smaller one."""
    rng = np.random.default_rng(2)
    ref = _events(rng.normal(1.4, 0.15, 80))
    result = minimum_detectable_difference(ref, t_max=8.0, n_sim=40, n_boot=150, seed=0)
    powers = [p for _, p in result["power_curve"]]
    # allow small Monte Carlo non-monotonicity, but the LAST point must
    # clearly beat the FIRST -- the real property under test.
    assert powers[-1] > powers[0]


def test_mdd_deterministic_reference_gets_a_nondegenerate_grid():
    """A perfectly deterministic reference population (every episode
    fails at the SAME instant, e.g. Cosmos's own zero-variance result on
    ramp_descent_high_friction) has median_abs_delta=0.0 -- self-scaling
    the grid off that literally would test delta=0 eight times over.
    Must fall back to a t_max-derived floor instead, producing a real,
    non-degenerate, increasing grid."""
    ref = _events([1.5] * 40)
    result = minimum_detectable_difference(ref, t_max=8.0, n_sim=10, n_boot=50, seed=0)
    deltas = [d for d, _ in result["power_curve"]]
    assert deltas[0] > 0.0, "grid must not degenerate to all-zero deltas"
    assert len(set(deltas)) == len(deltas), "grid points must be distinct, not all collapsed to one value"
    assert deltas == sorted(deltas)


def test_mdd_shrinks_with_more_episodes_per_arm():
    """Bigger sample size must detect a SMALLER true difference at the
    same target power -- the actual point of a power analysis. Uses a
    fixed delta_grid so both calls are judged on identical candidate
    deltas, isolating n_per_arm as the only thing that changed."""
    rng = np.random.default_rng(3)
    ref = _events(rng.normal(1.4, 0.2, 200))   # a larger reference pool to draw both arm sizes from
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    small = minimum_detectable_difference(ref, t_max=8.0, n_per_arm=15, delta_grid=grid,
                                           n_sim=40, n_boot=150, seed=0)
    large = minimum_detectable_difference(ref, t_max=8.0, n_per_arm=150, delta_grid=grid,
                                           n_sim=40, n_boot=150, seed=0)
    assert small["mdd"] is not None and large["mdd"] is not None
    assert large["mdd"] < small["mdd"]


def test_mdd_shifted_events_all_past_horizon_reports_nan_not_a_crash():
    """A shift that pushes EVERY arm-B event past t_max makes arm B's own
    VI_50 structurally undefined (all-censored -> inf), same as bootstrap_
    vi_delta's own documented behavior for a structurally-censored
    population (its own docstring/test: GATE 1a's near-never-terminates
    null population correctly returns NaN, not a fabricated number) --
    this must propagate to an honest NaN power here too, not crash and
    not silently invent a detection rate."""
    ref = _events([7.9] * 30)
    result = minimum_detectable_difference(ref, t_max=8.0, delta_grid=[1.0], n_sim=5, n_boot=50, seed=0)
    assert result["power_curve"][0][0] == 1.0
    assert np.isnan(result["power_curve"][0][1])
    assert result["mdd"] is None


def test_mdd_partial_shift_stays_within_horizon_and_produces_a_probability():
    """A moderate shift that does NOT push every event past t_max must
    produce a real power value in [0, 1], not NaN -- the counterpart to
    the all-past-horizon case above, confirming the capping logic only
    censors the events that actually cross t_max, not the whole arm."""
    rng = np.random.default_rng(4)
    ref = _events(rng.uniform(1.0, 3.0, 60))   # comfortably below t_max=8.0 even after a 1s shift
    result = minimum_detectable_difference(ref, t_max=8.0, delta_grid=[1.0], n_sim=20, n_boot=100, seed=0)
    power = result["power_curve"][0][1]
    assert not np.isnan(power)
    assert 0.0 <= power <= 1.0


if __name__ == "__main__":
    test_seed_test_retest_near_zero_for_a_deterministic_population()
    print("PASS  test_seed_test_retest_near_zero_for_a_deterministic_population")
    test_seed_test_retest_spread_grows_with_underlying_noise()
    print("PASS  test_seed_test_retest_spread_grows_with_underlying_noise")
    test_seed_test_retest_rejects_too_few_events()
    print("PASS  test_seed_test_retest_rejects_too_few_events")
    test_mdd_power_increases_with_delta()
    print("PASS  test_mdd_power_increases_with_delta")
    test_mdd_deterministic_reference_gets_a_nondegenerate_grid()
    print("PASS  test_mdd_deterministic_reference_gets_a_nondegenerate_grid")
    test_mdd_shrinks_with_more_episodes_per_arm()
    print("PASS  test_mdd_shrinks_with_more_episodes_per_arm")
    test_mdd_shifted_events_all_past_horizon_reports_nan_not_a_crash()
    print("PASS  test_mdd_shifted_events_all_past_horizon_reports_nan_not_a_crash")
    test_mdd_partial_shift_stays_within_horizon_and_produces_a_probability()
    print("PASS  test_mdd_partial_shift_stays_within_horizon_and_produces_a_probability")
