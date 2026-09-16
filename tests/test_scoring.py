"""vitals/stats/scoring.py (P6) -- checked against closed forms and
against the calibration behaviors the scores are supposed to detect, not
against themselves.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from math import erf, exp, pi, sqrt
from vitals.stats.scoring import (fair_crps, plain_crps, fair_energy_score, fair_energy_distance,
                                  spread_skill, spread_skill_ratio, rank_histogram,
                                  score_episode, aggregate_episodes, shared_object_count)
from vitals.types import Trajectory


def _crps_gaussian(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting & Raftery 2007)."""
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + erf(z / sqrt(2)))
    phi = exp(-0.5 * z * z) / sqrt(2 * pi)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))


def test_fair_crps_matches_gaussian_closed_form_for_large_ensemble():
    rng = np.random.default_rng(0)
    ens = rng.normal(0.0, 1.0, size=(4000,))
    for y in (-1.5, 0.0, 0.7, 2.0):
        assert abs(fair_crps(ens, np.float64(y)) - _crps_gaussian(y, 0.0, 1.0)) < 0.03


def test_fair_estimator_is_unbiased_at_small_N_and_plain_is_biased_high():
    """The whole point of §3.9: averaged over many draws, a 5-member fair
    CRPS equals the infinite-ensemble value; the plain estimator sits
    visibly above it."""
    rng = np.random.default_rng(1)
    y = 0.3
    truth = _crps_gaussian(y, 0.0, 1.0)
    ens = rng.normal(0.0, 1.0, size=(5, 20000))            # 20000 independent 5-member ensembles
    obs = np.full(20000, y)
    fair = fair_crps(ens, obs).mean()
    plain = plain_crps(ens, obs).mean()
    assert abs(fair - truth) < 0.01, (fair, truth)
    assert plain - truth > 0.05, (plain, truth)             # bias ~ E|x-x'|/(2N) = (2/sqrt(pi))/10 ~ 0.11


def test_energy_score_equals_crps_in_one_dimension():
    rng = np.random.default_rng(2)
    ens = rng.normal(size=(30, 7, 1))
    obs = rng.normal(size=(7, 1))
    es = fair_energy_score(ens, obs)                        # norm over the trailing D=1 axis
    cr = fair_crps(ens[..., 0], obs[..., 0])
    assert np.allclose(es, cr)


def test_energy_score_is_proper_prefers_true_distribution():
    """Strict propriety, empirically: the ensemble drawn from the truth's
    own distribution scores better in expectation than an ensemble that is
    shifted, too narrow, or too wide."""
    rng = np.random.default_rng(3)
    cases = 4000
    obs = rng.normal(size=(cases, 3))
    good = rng.normal(size=(20, cases, 3))
    shifted = good + 0.7
    narrow = 0.3 * good
    wide = 2.5 * good
    s_good = fair_energy_score(good, obs).mean()
    for bad in (shifted, narrow, wide):
        assert fair_energy_score(bad, obs).mean() > s_good + 0.02


def test_energy_distance_zero_for_same_distribution_positive_otherwise():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(40, 500, 3))
    b = rng.normal(size=(60, 500, 3))
    same = fair_energy_distance(a, b).mean()
    assert abs(same) < 0.02, same                           # unbiased: ~0, may be slightly negative
    c = rng.normal(loc=1.0, size=(60, 500, 3))
    assert fair_energy_distance(a, c).mean() > 0.3
    d = 3.0 * rng.normal(size=(60, 500, 3))                 # same mean, wrong spread -> still caught
    assert fair_energy_distance(a, d).mean() > 0.3


def test_rank_histogram_flat_when_calibrated_peaked_when_overdispersed():
    rng = np.random.default_rng(5)
    n, cases = 9, 20000
    obs = rng.normal(size=cases)
    calibrated = rng.normal(size=(n, cases))
    counts, ri = rank_histogram(calibrated, obs, rng=rng)
    assert counts.shape == (n + 1,)
    assert ri < 0.05, ri                                    # ~flat (expected |f-1/10| sum from sampling noise ~0.02)
    over = 3.0 * rng.normal(size=(n, cases))
    counts_o, ri_o = rank_histogram(over, obs, rng=rng)
    mid, ends = counts_o[3:7].sum(), counts_o[[0, 1, 8, 9]].sum()
    assert mid > 2 * ends and ri_o > 0.3                    # truth sits in the middle of an over-wide ensemble
    under = 0.3 * rng.normal(size=(n, cases))
    counts_u, _ = rank_histogram(under, obs, rng=rng)
    assert counts_u[[0, n]].sum() > 0.5 * counts_u.sum()    # U-shape: truth falls outside a too-narrow ensemble


def test_spread_skill_ratio_near_one_when_calibrated():
    rng = np.random.default_rng(6)
    n, cases = 8, 20000
    obs = rng.normal(size=cases)
    sp, sk = spread_skill(rng.normal(size=(n, cases)), obs)
    r = spread_skill_ratio(sp, sk, n)
    assert 0.95 < r < 1.05, r
    sp2, sk2 = spread_skill(0.4 * rng.normal(size=(n, cases)), obs)
    assert spread_skill_ratio(sp2, sk2, n) < 0.6            # over-confident


def _traj(pos, present=None):
    T, K, _ = pos.shape
    present = np.ones((T, K), bool) if present is None else present
    return Trajectory(t=np.arange(T) / 30.0, pos=pos, quat=np.tile([1, 0, 0, 0.0], (T, K, 1)),
                      present=present, names=[f"o{i}" for i in range(K)], meta={})


def test_score_episode_handles_K_mismatch_and_absence_and_prefix():
    """§3.8: candidate K=1 vs reference K=2 scores the shared object; an
    absent frame is NaN, not 0; the conditioning prefix is excluded."""
    rng = np.random.default_rng(7)
    T, tc = 40, 10
    truth_pos = np.cumsum(rng.normal(size=(T, 2, 3)), axis=0)
    truth = _traj(truth_pos)
    samples = []
    for _ in range(6):
        p = truth_pos[:, :1].copy()
        p[tc:] += rng.normal(scale=0.5, size=(T - tc, 1, 3))
        pres = np.ones((T, 1), bool); pres[35:, 0] = False
        samples.append(_traj(p, pres))
    refs = [_traj(truth_pos + rng.normal(scale=0.5, size=truth_pos.shape)) for _ in range(12)]
    assert shared_object_count(samples, [truth], refs) == 1
    out = score_episode(samples, truth, refs, t_c_frames=tc)
    assert out["K"] == 1 and out["n_samples"] == 6 and out["n_refs"] == 12
    assert out["crps"].shape == (T - tc,) and out["energy_score"].shape == (T - tc,)
    assert np.all(np.isfinite(out["crps"][: 35 - tc])) and np.all(np.isnan(out["crps"][35 - tc:]))
    assert np.all(np.isnan(out["energy_score"][35 - tc:]))
    assert out["energy_distance"] is not None and out["energy_distance"].shape == (T - tc,)
    agg = aggregate_episodes([out, out])
    assert agg["n_episodes"] == 2 and agg["crps"].shape == (T - tc,) and np.isfinite(agg["spread_skill_ratio"])


def test_score_episode_requires_an_ensemble():
    p = np.zeros((5, 1, 3))
    try:
        score_episode([_traj(p)], _traj(p))
    except ValueError as e:
        assert "n_samples >= 2" in str(e)
    else:
        raise AssertionError("a single sample must be rejected -- it has no spread to score")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
