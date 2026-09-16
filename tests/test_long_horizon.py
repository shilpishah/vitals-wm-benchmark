"""vitals/detect/long_horizon.py -- R4 (long-horizon/cumulative
consistency). Pure numpy, no MuJoCo/GPU needed -- every fixture here is a
plain array or a trivial closure over one, isolating this module's own
CUSUM/combination logic from the rest of the detector stack."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from vitals.detect.long_horizon import cusum_from_sigma, make_long_horizon_sigma
from vitals.detect.thresholds import estimate_threshold
from vitals.detect.events import extract_event


def test_cusum_stays_near_zero_when_running_near_typical():
    rng = np.random.default_rng(0)
    baseline = np.full(200, 2.0)
    theta = np.full(200, 4.0)   # gap of 2.0 between typical and instant threshold
    sigma = 2.0 + rng.normal(0, 0.1, size=200)   # r ~ 0.05, well below the default 0.3 slack
    S = cusum_from_sigma(sigma, baseline, theta, slack=0.3)
    assert S.max() < 1.0, f"noise near the typical value should never accumulate, got max={S.max()}"


def test_cusum_grows_under_a_persistent_bias():
    baseline = np.full(200, 2.0)
    theta = np.full(200, 4.0)
    sigma = np.full(200, 3.5)   # r = 0.75, well above the 0.3 slack, still under theta itself
    S = cusum_from_sigma(sigma, baseline, theta, slack=0.3)
    assert S[-1] > S[50], "a sustained bias must keep accumulating over the horizon, not plateau early"
    assert np.all(np.diff(S) >= -1e-9), "a strictly one-sided persistent bias should never let S decrease"


def test_cusum_decays_back_down_after_a_transient_spike():
    T = 400
    baseline = np.full(T, 2.0)
    theta = np.full(T, 4.0)
    sigma = np.full(T, 2.2)   # r = 0.1, below slack -- decays toward 0 at rest
    sigma[10:15] = 20.0        # one brief, large spike, then back to normal
    S = cusum_from_sigma(sigma, baseline, theta, slack=0.3)
    assert S[14] > 0, "the spike itself must register"
    assert S[-1] < S[14], "S must decay back down once the signal returns to normal, not stay elevated forever"
    assert S[-1] == 0.0, "given enough normal frames afterward, S must fully return to its floor, not linger forever"


def test_cusum_holds_previous_value_through_nan_never_resets_or_propagates():
    baseline = np.full(50, 2.0)
    theta = np.full(50, 4.0)
    sigma = np.full(50, 3.5)   # steadily above the slack fraction of the gap
    sigma[20:25] = np.nan
    S = cusum_from_sigma(sigma, baseline, theta, slack=0.3)
    assert S[19] == S[20] == S[21] == S[24], "NaN frames must hold S at its last defined value, not reset it"
    assert not np.any(np.isnan(S)), "S itself must never be NaN, even where the input was"
    assert S[25] > S[24], "accumulation must resume correctly once real evidence returns"


def test_cusum_treats_a_degenerate_zero_gap_as_undefined_not_infinite():
    """The actual bug found and fixed building this: theta_R1 (existence)
    calibrates to EXACTLY the same value as its own baseline (both 0.0)
    in practice, confirmed directly against real reference-ensemble data,
    not assumed -- a channel with NO gap between typical and threshold
    has no sub-threshold regime to accumulate at all, and must contribute
    nothing, never blow up."""
    baseline = np.zeros(50)
    theta = np.zeros(50)      # the actual, real theta_R1/baseline_R1 case found
    sigma = np.full(50, 0.05)  # a small, real, legitimate deviation
    S = cusum_from_sigma(sigma, baseline, theta, slack=0.3)
    assert np.all(S == 0.0), f"a degenerate zero gap must contribute nothing at all, got max={S.max()}"


class _FakeTraj:
    def __init__(self, arr):
        self.arr = arr


def _synthetic_ref(rng, T, base_level, noise):
    return base_level + rng.normal(0, noise, size=T)


def test_make_long_horizon_sigma_matches_fn_traj_refs_contract():
    """sigma_R4 must be directly usable by estimate_threshold/extract_event
    -- the whole point of matching the existing fn(traj, refs) contract."""
    rng = np.random.default_rng(1)
    T = 120
    refs = [_FakeTraj(_synthetic_ref(rng, T, 2.0, 0.1)) for _ in range(30)]
    base_stats = {"R3": (lambda traj, others: traj.arr)}
    thetas = {k: estimate_threshold(refs, fn, alpha=0.01) for k, fn in base_stats.items()}

    sigma_R4 = make_long_horizon_sigma(base_stats, refs, thetas)
    theta_R4 = estimate_threshold(refs, sigma_R4, alpha=0.01)
    assert theta_R4.shape == (T,)

    candidate = _FakeTraj(np.full(T, 2.0))
    s = sigma_R4(candidate, refs)
    assert s.shape == (T,)
    ev = extract_event({"R4": s}, {"R4": theta_R4}, dt=1 / 30, t_max=T / 30)
    assert ev.censored, "a candidate matching the reference's own typical level should not fire R4"


def test_make_long_horizon_sigma_is_max_across_active_channels():
    """Two base channels, only one drifting -- sigma_R4 must reflect the
    WORSE of the two at every frame, not average them away."""
    T = 60
    refs = [_FakeTraj(None) for _ in range(10)]   # unused by these fake fns, present for the contract

    def fn_flat(traj, others):
        return np.full(T, 1.0)

    def fn_drifting(traj, others):
        return np.full(T, 1.0) + np.linspace(0, 3.0, T)

    thetas = {"A": np.full(T, 4.0), "B": np.full(T, 4.0)}
    base_stats = {"A": fn_flat, "B": fn_drifting}
    sigma_R4 = make_long_horizon_sigma(base_stats, refs, thetas)
    s = sigma_R4(_FakeTraj(None), refs)

    solo_B = make_long_horizon_sigma({"B": fn_drifting}, refs, {"B": thetas["B"]})
    s_B_only = solo_B(_FakeTraj(None), refs)
    assert np.allclose(s, s_B_only), "the flat channel must never suppress the drifting one's own signal"


def test_r4_catches_a_subthreshold_drift_that_evades_first_sustained_crossing():
    """The actual point of building this: a slow, persistent bias whose
    per-frame deviation NEVER crosses the standard per-bin calibrated
    threshold (so the existing first-sustained-crossing detector reports
    a clean, censored episode) must still be caught by R4's own
    cumulative statistic."""
    rng = np.random.default_rng(3)
    T = 300
    M = 40
    refs = [_FakeTraj(_synthetic_ref(rng, T, 2.0, 0.15)) for _ in range(M)]
    base_stats = {"R3": (lambda traj, others: traj.arr)}
    thetas = {k: estimate_threshold(refs, fn, alpha=0.01) for k, fn in base_stats.items()}
    theta_R3 = thetas["R3"]

    # a short ramp up to a SUSTAINED plateau, deliberately kept just under
    # the calibrated per-frame threshold at every single instant -- a
    # brief ramp alone doesn't accumulate enough CUSUM signal to matter;
    # sustained duration is exactly the thing this detector exists to
    # exploit that a per-frame threshold can't.
    creep = np.concatenate([np.linspace(2.0, 2.55, 30), np.full(T - 30, 2.55)])
    assert np.all(creep < theta_R3 * 0.99), "test fixture must actually stay sub-threshold everywhere to be meaningful"
    candidate = _FakeTraj(creep)

    primary_event = extract_event({"R3": base_stats["R3"](candidate, refs)}, {"R3": theta_R3}, dt=1 / 30, t_max=T / 30)
    assert primary_event.censored, "fixture must genuinely evade the existing detector, or this test proves nothing"

    sigma_R4 = make_long_horizon_sigma(base_stats, refs, thetas)
    theta_R4 = estimate_threshold(refs, sigma_R4, alpha=0.01)
    s = sigma_R4(candidate, refs)
    r3_event = extract_event({"R4": s}, {"R4": theta_R4}, dt=1 / 30, t_max=T / 30)
    assert not r3_event.censored, "R4 must catch the sustained drift the primary per-frame detector missed"


if __name__ == "__main__":
    test_cusum_stays_near_zero_when_running_near_typical()
    test_cusum_grows_under_a_persistent_bias()
    test_cusum_decays_back_down_after_a_transient_spike()
    test_cusum_holds_previous_value_through_nan_never_resets_or_propagates()
    test_cusum_treats_a_degenerate_zero_gap_as_undefined_not_infinite()
    test_make_long_horizon_sigma_matches_fn_traj_refs_contract()
    test_make_long_horizon_sigma_is_max_across_active_channels()
    test_r4_catches_a_subthreshold_drift_that_evades_first_sustained_crossing()
    print("PASS  all long_horizon tests")
