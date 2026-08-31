"""Survival estimation under competing risks.

Kaplan-Meier for overall survival S(t); Aalen-Johansen for cause-specific
cumulative incidence F_k(t). Treating competing events as ordinary censoring
biases absolute risk -- so do not substitute independent KM curves per risk.
"""
from __future__ import annotations
import numpy as np
from ..types import Event


def _tabulate(events):
    times = np.array([e.time for e in events], float)
    risks = np.array([e.risk if e.risk else "" for e in events], object)
    obs = np.array([not e.censored for e in events], bool)
    order = np.argsort(times)
    return times[order], risks[order], obs[order]


def kaplan_meier(events):
    """Returns (t_grid, S(t))."""
    times, _, obs = _tabulate(events)
    n = len(times)
    grid, s, surv, at_risk = [0.0], 1.0, [1.0], n
    for i, t in enumerate(times):
        if obs[i]:
            s *= (1 - 1 / at_risk)
            grid.append(t); surv.append(s)
        at_risk -= 1
    return np.array(grid), np.array(surv)


def validity_interval(events, q=0.5):
    """VI_q: first time the survival curve drops to or below q."""
    grid, surv = kaplan_meier(events)
    below = np.where(surv <= q)[0]
    return float(grid[below[0]]) if len(below) else float("inf")


def aalen_johansen(events, risks=None):
    """Cause-specific cumulative incidence. Returns (t_grid, {risk: F_k})."""
    times, rk, obs = _tabulate(events)
    risks = risks or sorted({r for r in rk if r})
    n = len(times)
    grid, cif = [0.0], {r: [0.0] for r in risks}
    s, at_risk = 1.0, n
    for i, t in enumerate(times):
        if obs[i]:
            haz = 1 / at_risk
            grid.append(t)
            for r in risks:
                inc = s * haz if rk[i] == r else 0.0
                cif[r].append(cif[r][-1] + inc)
            s *= (1 - haz)
        at_risk -= 1
    return np.array(grid), {r: np.array(v) for r, v in cif.items()}


def termination_profile(events, risks=None):
    """Normalised share of terminations by cause at T_max."""
    rk = [e.risk for e in events if not e.censored]
    risks = risks or sorted(set(rk))
    total = len(rk)
    if total == 0:
        return {r: 0.0 for r in risks}
    return {r: rk.count(r) / total for r in risks}


def bootstrap_vi(events, q=0.5, n_boot=1000, seed=0):
    """Percentile CI on VI_q. Resample episodes, not rollouts."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(events), len(events))
        v = validity_interval([events[i] for i in idx], q)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def bootstrap_vi_delta(events_a, events_b, q=0.5, n_boot=1000, seed=0):
    """Percentile CI on VI_q(a) - VI_q(b) -- the general form of "compare a
    good rollout population against a bad one and look at the delta"
    (2026-08, requested directly rather than something this project
    invented in isolation): works on ANY two populations of Events, not
    just a real model eventually -- two mutant conditions, two lambda
    levels, two scenarios today, two real models' measured rollouts once
    M6 exists.

    Independent two-sample bootstrap: each population is resampled ON ITS
    OWN (not a matched/paired resample using the same indices in both).
    This is the right default when the two populations aren't the same
    underlying episodes measured twice -- e.g. two different scenarios, or
    two mutant types drawn from independent seeds, which is the ONLY case
    this repo currently has data for. If the two populations genuinely ARE
    the same seeds run through two different measurement conditions (e.g.
    the same 80 rollouts scored under model A's continuation vs model B's),
    a PAIRED bootstrap -- resampling the same episode indices for both
    populations on each iteration -- would be a strictly more powerful
    test (it cancels shared per-episode variance instead of adding both
    populations' sampling noise together) and is NOT what this function
    does; do not use this function's CI width as evidence a paired design
    would show the same thing.

    Returns (point_estimate, ci_lo, ci_hi). If the CI excludes 0, that's
    evidence the two populations' VI_q differ by more than independent
    sampling noise alone would produce. This is NOT a formal p-value and
    NOT the full picture this project's own M7 acceptance criterion asks
    for: a single-lambda delta can flip sign at a different lambda, and
    the actual acceptance criterion is RANK-ORDER STABILITY across the
    lambda sweep, not a single comparison at whatever lambda happened to
    be used (§7 M7's own docstring). Treat a significant delta at one
    lambda as suggestive, not conclusive, until checked across the sweep.

    Same censoring caveat as `bootstrap_vi`: a resample where EITHER
    population's VI_q comes back infinite (censored past t_max, not
    reached) is dropped entirely, not treated as a large-but-finite delta
    -- can silently shrink n_boot a lot under high censoring, same
    limitation, not newly introduced here."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx_a = rng.integers(0, len(events_a), len(events_a))
        idx_b = rng.integers(0, len(events_b), len(events_b))
        va = validity_interval([events_a[i] for i in idx_a], q)
        vb = validity_interval([events_b[i] for i in idx_b], q)
        if np.isfinite(va) and np.isfinite(vb):
            deltas.append(va - vb)
    if not deltas:
        return (np.nan, np.nan, np.nan)
    point = validity_interval(events_a, q) - validity_interval(events_b, q)
    return (point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def seed_test_retest(events, q=0.5, n_splits=200, seed=0):
    """Empirical run-to-run reproducibility of VI_q, using ONLY a single
    already-collected population (2026-08, M7's own "seed test-retest" --
    "how much VI_50 spread exists from finite-sample noise ALONE, even
    between two runs of the identical condition"). Repeatedly splits
    `events` into two DISJOINT random halves (sampling WITHOUT
    replacement, unlike `bootstrap_vi`'s resample-WITH-replacement from
    the same n) and computes each half's own VI_q -- the difference is a
    genuine estimate of "if this experiment had instead drawn a
    different, equally-sized batch of held-out seeds, would VI_q have
    come back the same," which is closer to true seed-to-seed
    reproducibility than a same-sample bootstrap CI is (a bootstrap
    resample still contains up to 100% overlap with any other resample;
    a disjoint half never does). Needs no second real model call --
    reuses an existing population's own already-paid-for episodes, same
    "don't re-run what you already have" discipline as everywhere else
    real GPU cost is involved in this project.

    Needs len(events) >= 4 (>=2 per half) or raises -- a population this
    small can't say anything about its own reproducibility; `Wan`'s
    current n=5 populations are right at this floor and any test-retest
    read from them should be treated as barely-informative, not ignored
    outright (see this function's own docstring precedent in `bootstrap_
    vi`/`bootstrap_vi_delta` for why partial information is still
    reported, not refused).

    Returns dict(median_abs_delta, p90_abs_delta, n_valid_splits,
    n_splits, half_n) -- splits where EITHER half's VI_q comes back
    infinite (censored past t_max) are dropped, same convention as
    `bootstrap_vi`/`bootstrap_vi_delta`; `n_valid_splits` reports how many
    of the `n_splits` attempts actually contributed, since high censoring
    can shrink this a lot silently otherwise."""
    if len(events) < 4:
        raise ValueError(f"seed_test_retest needs at least 4 events (>=2 per half), got {len(events)}")
    rng = np.random.default_rng(seed)
    n = len(events)
    half = n // 2
    deltas = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a_idx, b_idx = idx[:half], idx[half:2 * half]
        va = validity_interval([events[i] for i in a_idx], q)
        vb = validity_interval([events[i] for i in b_idx], q)
        if np.isfinite(va) and np.isfinite(vb):
            deltas.append(abs(va - vb))
    if not deltas:
        return dict(median_abs_delta=float("nan"), p90_abs_delta=float("nan"),
                    n_valid_splits=0, n_splits=n_splits, half_n=half)
    arr = np.array(deltas)
    return dict(median_abs_delta=float(np.median(arr)), p90_abs_delta=float(np.percentile(arr, 90)),
                n_valid_splits=len(deltas), n_splits=n_splits, half_n=half)


def minimum_detectable_difference(events_reference, t_max, q=0.5, n_per_arm=None, target_power=0.8,
                                   n_sim=50, n_boot=200, delta_grid=None, seed=0):
    """Simulation-based minimum detectable difference in VI_q (M7's own
    "how large does a delta need to be before it's even worth
    bootstrapping in the first place" -- `bootstrap_vi_delta`'s own
    docstring flags this as the missing calibration step, this is it).

    The smallest true shift `delta` (seconds) between two `n_per_arm`-
    sized populations, BOTH drawn from `events_reference`'s own empirical
    failure-time distribution (a REAL collected population -- e.g. an
    existing `results/l0_demo_*.json`'s events -- not an assumed
    parametric shape, matching this project's own "real data over
    synthetic assumption" discipline everywhere else), such that
    `bootstrap_vi_delta` correctly flags the two arms as significantly
    different (its own 95% CI excludes 0) in at least `target_power`
    fraction of `n_sim` independent simulated replicates.

    Each simulated arm-B event is an arm-A-style resampled time PLUS
    `delta`, RE-CENSORED at `t_max` if the shift pushes it past the
    clip's own observed horizon (an event can't be observed happening
    later than the clip runs, so a naive unbounded shift would silently
    fabricate evidence beyond what any real episode could show).

    n_per_arm: defaults to len(events_reference) -- "how big a delta
    would I need to detect AT THE SAMPLE SIZE I ALREADY HAVE," the
    directly useful question for judging an existing result; pass a
    different n_per_arm to ask "how big would my NEXT run need to be."

    delta_grid: candidate deltas to test (seconds). Default: 8 points
    from 0.25x to 4x this SAME population's own `seed_test_retest`
    median split delta -- self-scaling to whatever noise level this
    particular population/scenario actually has (a fixed guess like
    "try 0.1s, 0.5s, 1s" would be meaningless on a scenario whose whole
    clip is 1.5s vs. one that's 8s).

    Significance is judged at `bootstrap_vi_delta`'s own fixed 95%
    two-sided CI (alpha=0.05 by construction, not independently
    parameterized here -- `bootstrap_vi_delta` itself has no alpha knob).

    Returns dict(mdd, power_curve=[(delta, power), ...], n_per_arm,
    target_power, n_sim) -- `mdd` is the smallest grid delta whose
    empirical power >= target_power, or None if NO grid point reached it
    (the honest answer is "report the curve, don't extrapolate past what
    was actually tested," not a guessed larger number)."""
    rng = np.random.default_rng(seed)
    n_per_arm = n_per_arm or len(events_reference)

    if delta_grid is None:
        tr = seed_test_retest(events_reference, q=q, n_splits=100, seed=seed)
        # A PERFECTLY deterministic reference population (e.g. Cosmos's
        # own zero-variance ramp_descent_high_friction result -- every
        # episode fails at literally the same instant) gives
        # median_abs_delta=0.0, which is finite but degenerates the
        # self-scaled grid to all-zero deltas -- floor it against a
        # fraction of the clip's own horizon instead of silently testing
        # the same (trivial, always-detectable-at-only-noise-level) delta
        # eight times over.
        base = tr["median_abs_delta"]
        if not np.isfinite(base) or base <= 0:
            base = max(t_max * 0.05, 0.1)
        delta_grid = [base * m for m in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)]

    ref_times = np.array([e.time for e in events_reference])
    ref_risks = [e.risk for e in events_reference]
    ref_censored = np.array([e.censored for e in events_reference])
    n_ref = len(events_reference)

    def resample_arm(shift, sub_rng):
        idx = sub_rng.integers(0, n_ref, n_per_arm)
        arm = []
        for i in idx:
            if ref_censored[i]:
                arm.append(Event(time=t_max, risk=None, censored=True))
                continue
            t = float(ref_times[i]) + shift
            if t >= t_max:
                arm.append(Event(time=t_max, risk=None, censored=True))
            else:
                arm.append(Event(time=t, risk=ref_risks[i], censored=False))
        return arm

    power_curve = []
    for delta in delta_grid:
        detected, valid = 0, 0
        for _ in range(n_sim):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(sub_seed)
            arm_a = resample_arm(0.0, sub_rng)
            arm_b = resample_arm(delta, sub_rng)
            point, lo, hi = bootstrap_vi_delta(arm_a, arm_b, q=q, n_boot=n_boot, seed=sub_seed)
            if np.isnan(point):
                continue
            valid += 1
            if hi < 0 or lo > 0:
                detected += 1
        power = (detected / valid) if valid else float("nan")
        power_curve.append((float(delta), float(power)))

    mdd = next((d for d, p in power_curve if not np.isnan(p) and p >= target_power), None)
    return dict(mdd=mdd, power_curve=power_curve, n_per_arm=n_per_arm,
                target_power=target_power, n_sim=n_sim)


def _average_ranks(values):
    """Rank 1..n, ties given the mean of the ranks they span -- the
    standard tie handling for Spearman's rho (`scipy.stats.rankdata`'s
    default), reimplemented here in plain numpy since this project has no
    scipy dependency anywhere else."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rank_correlation(values_a, values_b):
    """Spearman's rho between two equal-length sequences of scalars --
    M7's own "rank correlation between [lambda sweep] levels" (AGENT.md),
    used here to ask whether the relative ORDER of several conditions
    (e.g. Truth / ConstantVelocity / CopyLastState VI_50s) stays the same
    as the reference ensemble's own spread parameter (lambda) changes. A
    positive answer (rho close to 1 between adjacent levels) is the actual
    acceptance criterion; a rho that drops or goes negative is a real
    ranking-instability finding, not a bug in this function.

    Implemented as Pearson correlation of average ranks (ties handled via
    `_average_ranks`) rather than importing scipy, which this project does
    not otherwise depend on. Returns NaN -- not 0, not 1 -- whenever the
    result is genuinely undefined: fewer than 2 values, or either sequence
    has zero rank variance (every value tied, e.g. every condition
    censored at the identical horizon), since Pearson correlation divides
    by a rank standard deviation that would be zero there."""
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])