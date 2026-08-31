"""Core tests -- no MuJoCo required. Exercises the whole state-space path.

Runnable directly (python3 tests/test_core.py) or via pytest.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from vitals.types import Event, EpisodeSpec, Trajectory
from vitals.physics import make_backend
from vitals.physics.perturb import sample_perturbation
from vitals.mutants import library as mut
from vitals.detect.statistics import sigma_existence, sigma_kinematic, sigma_interpenetration, kinematic_axes_for
from vitals.detect.thresholds import estimate_threshold
from vitals.detect.events import extract_event
from vitals.stats.survival import kaplan_meier, validity_interval, termination_profile

SPEC = EpisodeSpec(name="t", scene="synthetic", target_property="P2", band="I",
                    lam=1.0, n_reference=12, horizon_s=3.0, fps=30)


def _refs():
    roll = make_backend("synthetic")
    return [roll(SPEC, 1000 + i) for i in range(SPEC.n_reference)]


def _thetas(refs):
    return {"R2": estimate_threshold(refs, sigma_existence, alpha=0.01),
            "R5": estimate_threshold(refs, sigma_kinematic, alpha=0.01)}


def test_perturbation_scales_with_lambda():
    small = sample_perturbation(np.random.default_rng(0), 0.1, 1)
    large = sample_perturbation(np.random.default_rng(0), 4.0, 1)
    assert np.abs(large["dpos"]).mean() > np.abs(small["dpos"]).mean()


def test_synthetic_rollout_shape():
    traj = make_backend("synthetic")(SPEC, 42)
    n = int(SPEC.horizon_s * SPEC.fps)
    assert traj.pos.shape == (n, 1, 3)
    assert traj.T == n and traj.K == 1


def test_vanish_mutant_flips_present_and_keeps_K():
    traj = make_backend("synthetic")(SPEC, 1)
    m = mut.vanish(traj, t_star=1.0)
    i = int(1.0 / traj.dt)
    assert not m.traj.present[i:, 0].any()
    assert m.traj.K == traj.K
    assert m.risk_expected == "R2"


def test_duplicate_changes_K():
    traj = make_backend("synthetic")(SPEC, 1)
    m = mut.duplicate(traj, t_star=1.0)
    assert m.traj.K == traj.K + 1


def test_threshold_null_false_positive_rate_near_alpha():
    refs = _refs()
    thetas = _thetas(refs)
    held_out = [make_backend("synthetic")(SPEC, 9000 + i) for i in range(30)]
    events = []
    for h in held_out:
        sigmas = {"R2": sigma_existence(h, refs), "R5": sigma_kinematic(h, refs)}
        events.append(extract_event(sigmas, thetas, refs[0].dt, float(refs[0].t[-1])))
    fp_rate = sum(not e.censored for e in events) / len(events)
    assert fp_rate < 5 * 0.01 + 0.1   # loose bound; GATE 1a in run_l0_demo.py is the real check


def test_vanish_fires_on_r2_not_r5():
    refs = _refs()
    thetas = _thetas(refs)
    base = make_backend("synthetic")(SPEC, 555)
    m = mut.vanish(base, t_star=1.0)
    sigmas = {"R2": sigma_existence(m.traj, refs), "R5": sigma_kinematic(m.traj, refs)}
    ev = extract_event(sigmas, thetas, base.dt, float(base.t[-1]))
    assert ev.risk == "R2"


def test_velocity_freeze_fires_on_r5():
    refs = _refs()
    thetas = _thetas(refs)
    base = make_backend("synthetic")(SPEC, 777)
    m = mut.velocity_freeze(base, t_star=0.6)   # while the ball is still on the ramp
    sigmas = {"R2": sigma_existence(m.traj, refs), "R5": sigma_kinematic(m.traj, refs)}
    ev = extract_event(sigmas, thetas, base.dt, float(base.t[-1]))
    assert ev.risk == "R5"


def _single_obj_traj(pos_over_time, T=30, dt=1.0 / 30):
    """Hand-built K=1 Trajectory with an explicit (T,3) position array --
    the minimal fixture sigma_kinematic's own axes-restriction logic
    needs (AGENT.md M2.6): full control over per-axis deviation, not
    achievable through the synthetic backend's own fixed physics."""
    t = np.arange(T) * dt
    pos = np.asarray(pos_over_time, dtype=float).reshape(T, 1, 3)
    quat = np.tile(np.array([1.0, 0, 0, 0]), (T, 1, 1))
    present = np.ones((T, 1), bool)
    return Trajectory(t, pos, quat, present, ["ball"])


def test_kinematic_axes_for_is_a_strict_no_op_outside_ramp_descent():
    assert kinematic_axes_for("occlusion_corridor") is None
    assert kinematic_axes_for("projectile") is None
    assert kinematic_axes_for("ramp_descent") == (0, 2)
    assert kinematic_axes_for("ramp_descent_high_friction") == (0, 2)


def test_sigma_kinematic_axes_ignores_excluded_axis_deviation():
    """AGENT.md M2.6: a candidate that deviates ONLY on the excluded axis
    (Y here) must be flagged with axes=None (all 3 axes, the default) but
    NOT with axes=(0, 2) -- the whole point of restricting to physically-
    meaningful axes is that a deviation confined to an axis with near-zero
    reference variance (instrument noise, not a real defect) stops being
    able to dominate the combined statistic."""
    T = 30
    refs = [_single_obj_traj(np.tile([0.0, 0.0, 0.0], (T, 1))) for _ in range(12)]
    # candidate: 0.5m off on Y only, dead-on for X/Z
    cand = _single_obj_traj(np.tile([0.0, 0.5, 0.0], (T, 1)))

    sigma_all = sigma_kinematic(cand, refs)
    sigma_xz = sigma_kinematic(cand, refs, axes=(0, 2))

    assert np.all(sigma_all > 1.0), "a real Y-only deviation must still register with all 3 axes"
    assert np.all(sigma_xz == 0.0), "a Y-only deviation must be invisible when Y is excluded"


def _two_obj_traj(sep=2.0, T=30, dt=1.0 / 30, present0=None, present1=None):
    """Hand-built K=2 Trajectory -- obj0 at x=0, obj1 at x=sep, both
    stationary otherwise. No backend needed (the synthetic backend is
    K=1-only); this is the minimal fixture sigma_interpenetration's own
    obj=0/target=1 pairwise-distance logic needs."""
    t = np.arange(T) * dt
    pos = np.zeros((T, 2, 3))
    pos[:, 1, 0] = sep
    quat = np.tile(np.array([1.0, 0, 0, 0]), (T, 2, 1))
    present = np.ones((T, 2), bool)
    if present0 is not None:
        present[:, 0] = present0
    if present1 is not None:
        present[:, 1] = present1
    return Trajectory(t, pos, quat, present, ["obj0", "obj1"])


def _interpen_refs(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [_two_obj_traj(sep=2.0 + rng.normal(0, 0.02)) for _ in range(n)]


def test_sigma_interpenetration_flags_planted_overlap():
    """teleport (obj0 -> obj1's own position from t_star onward) must stay
    near the ratio's own typical value (~1.0 -- candidate separation
    matches the reference ensemble's own ~2.0m) before t_star, and cross
    FAR above 1.0 after t_star (separation collapses to ~0, so the
    reference-typical/actual ratio explodes)."""
    refs = _interpen_refs()
    base = _two_obj_traj(sep=2.0)
    m = mut.teleport(base, t_star=0.5, obj=0, target=1)
    assert m.risk_expected == "R4"
    sigma = sigma_interpenetration(m.traj, refs)
    i = int(0.5 / base.dt)
    assert np.nanmax(sigma[:i]) < 1.5, "pre-teleport separation matches the reference typical distance -- ratio should sit near 1.0, not flag"
    assert np.nanmin(sigma[i:]) > 50.0, "post-teleport separation collapses to ~0 -- the ratio must explode well past the ~1.0 typical value"


def test_sigma_interpenetration_nan_when_target_absent():
    """An existence failure (target object gone) is R2's job, not R4's --
    sigma_interpenetration must report NaN (undefined), never a large
    positive value, when either tracked object is absent."""
    refs = _interpen_refs()
    present1 = np.ones(30, bool)
    present1[10:] = False
    cand = _two_obj_traj(sep=2.0, present1=present1)
    sigma = sigma_interpenetration(cand, refs)
    assert np.all(np.isnan(sigma[10:])), "target absent -- must be NaN (undefined), not a manufactured R4 signal"
    assert not np.isnan(sigma[5]), "target still present here -- should be a real, defined value"


def test_sigma_interpenetration_is_one_sided():
    """Moving apart (farther than the reference ensemble ever gets) is not
    an interpenetration defect -- one-sided in EFFECT (the ratio formula
    makes it small, never able to cross a threshold calibrated near/above
    1.0), not by an explicit clip. Candidate separation (6.0m) vs.
    reference typical (~2.0m) should give a ratio well below 1.0."""
    refs = _interpen_refs()
    cand = _two_obj_traj(sep=6.0)   # well beyond the ~2.0m reference typical separation
    sigma = sigma_interpenetration(cand, refs)
    assert np.all(sigma < 0.5), "farther apart than usual must never approach a value large enough to register as a defect"


def test_sigma_existence_obj_restriction_is_a_noop_for_k1():
    """obj=0 must reproduce whole-count behavior exactly for a K=1
    trajectory -- count and single-object presence are identical there
    (GATE 2's own reason for binding obj=0 unconditionally, not just for
    K>1 scenarios -- see run_gate2.py's own STATS comment)."""
    roll = make_backend("synthetic")
    refs = [roll(SPEC, 1000 + i) for i in range(SPEC.n_reference)]
    cand = roll(SPEC, 42)
    whole_count = sigma_existence(cand, refs)
    obj0 = sigma_existence(cand, refs, obj=0)
    assert np.allclose(whole_count, obj0, equal_nan=True)


def test_sigma_existence_obj_restriction_fixes_the_k_mismatch_bug():
    """The actual bug found running GATE 2 on occlusion_corridor_
    distractor for the first time (2026-08): a K=1 candidate (every real-
    video/real-model L1 reconstruction, `track_and_reconstruct`'s own
    single-object return) scored with WHOLE-COUNT existence against a K=2
    reference ensemble (ball + an always-present decoy) spuriously fires
    at every frame regardless of tracking quality, because candidate
    count (1) can never match reference mean count (~2). obj=0 must fix
    this: a K=1 candidate whose OWN tracked object matches the
    reference's own object-0 presence exactly must NOT flag, even though
    the reference ensemble itself is K=2."""
    T = 30
    # References: K=2, both objects present the whole clip (occlusion_
    # corridor_distractor's own null-episode shape -- ball + inert decoy).
    refs = [Trajectory(t=np.arange(T) / 30.0, pos=np.zeros((T, 2, 3)),
                       quat=np.tile([1.0, 0, 0, 0], (T, 2, 1)),
                       present=np.ones((T, 2), bool), names=["ball", "decoy"])
            for _ in range(12)]
    # Candidate: K=1 (no decoy at all -- exactly what track_and_reconstruct
    # actually returns), object 0 present the whole clip, tracked fine.
    cand = Trajectory(t=np.arange(T) / 30.0, pos=np.zeros((T, 1, 3)),
                      quat=np.tile([1.0, 0, 0, 0], (T, 1, 1)),
                      present=np.ones((T, 1), bool), names=["ball"])

    whole_count = sigma_existence(cand, refs)
    assert np.all(whole_count > 100), \
        "sanity check: whole-count comparison really does spuriously explode on a K-mismatched candidate"

    obj0 = sigma_existence(cand, refs, obj=0)
    assert np.all(obj0 < 1e-6), \
        "obj=0 restriction must see this as a perfectly-tracked, always-present primary object -- no false flag"


def test_kaplan_meier_is_monotone_nonincreasing():
    events = [Event(1.0, "R2", False), Event(2.0, "R5", False), Event(3.0, None, True)]
    _, surv = kaplan_meier(events)
    assert np.all(np.diff(surv) <= 1e-12)


def test_validity_interval_and_termination_profile():
    events = [Event(1.0, "R2", False), Event(1.0, "R2", False),
              Event(5.0, "R5", False), Event(6.0, None, True)]
    assert validity_interval(events, 0.5) > 0
    prof = termination_profile(events)
    assert abs(sum(prof.values()) - 1.0) < 1e-9


def test_bootstrap_vi_delta_separates_clearly_different_populations():
    """2026-08: the STEAM-style 'good rollout vs bad rollout, take the
    delta' comparison, generalized to any two Event populations. A
    population that terminates early, every time, must show a clearly
    negative delta (with a CI that excludes 0) against one that survives
    much longer every time -- and comparing a population against ITSELF
    must center on zero."""
    from vitals.stats.survival import bootstrap_vi_delta
    early = [Event(1.0, "R2", False) for _ in range(20)]
    late = [Event(9.0, "R2", False) for _ in range(20)]

    point, lo, hi = bootstrap_vi_delta(early, late, q=0.5, n_boot=200, seed=0)
    assert point < 0, "early-terminating population should show a negative VI delta vs the late one"
    assert hi < 0, "a population that ALWAYS terminates earlier than another should have a CI entirely below zero"

    point_self, lo_self, hi_self = bootstrap_vi_delta(early, early, q=0.5, n_boot=200, seed=0)
    assert point_self == 0.0
    assert lo_self <= 0.0 <= hi_self, "comparing a population against itself must not show a spurious nonzero effect"


def test_spearman_rank_correlation_identical_and_reversed_and_ties():
    """M7's lambda-sweep 'rank correlation between levels' -- identical
    order -> rho=1, fully reversed order -> rho=-1, and the tie/undefined
    edge cases must return NaN (not a silently wrong 0 or 1)."""
    from vitals.stats.survival import spearman_rank_correlation
    import math

    assert spearman_rank_correlation([1, 2, 3], [10, 20, 30]) == 1.0
    assert spearman_rank_correlation([1, 2, 3], [30, 20, 10]) == -1.0

    # ties in one sequence still produce a defined, finite value
    rho = spearman_rank_correlation([1, 2, 2, 4], [1, 2, 3, 4])
    assert not math.isnan(rho) and rho > 0

    # every value tied -> rank variance is zero -> undefined, not 0
    assert math.isnan(spearman_rank_correlation([5, 5, 5], [1, 2, 3]))
    # fewer than 2 points -> undefined
    assert math.isnan(spearman_rank_correlation([1], [1]))
    # mismatched lengths -> undefined
    assert math.isnan(spearman_rank_correlation([1, 2], [1, 2, 3]))


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
