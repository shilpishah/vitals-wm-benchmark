"""Reference-derived thresholds.

theta_k(lambda) is calibrated so that a genuine held-out reference is falsely
declared failed **somewhere over the whole horizon** at rate alpha -- not
just at any single instant. Those are not the same target, and only the
first is the one GATE 1a actually checks.

A naive per-time-bin (1-alpha) quantile controls the exceedance probability
at each instant, but "first sustained crossing anywhere in T_max" is a
repeated test over ~n_bins quasi-independent windows: even if each window
is individually correct at alpha, the whole-path false-alarm rate inflates
to roughly 1-(1-alpha)^n_bins (union bound). Empirically this was ~12-14%
against a target of ~1% -- confirmed independent of n_bins, and confirmed by
many held-out references firing at *identical* times clustered at bin edges,
which is the signature of repeated-testing inflation, not of a broken
statistic.

The fix: separate *shape* from *level*. scale(t) is a per-bin robust local
scale (median of the leave-one-out curves -- deliberately not alpha-derived,
just a normalizer for the growing process variability). Each leave-one-out
curve is reduced to a single number, its own worst deviation over the whole
path: max_t sigma_i(t) / scale(t). theta(t) = c * scale(t), where c is the
(1-alpha) quantile of those M whole-path maxima. This directly calibrates
the event alpha claims to have, by construction, on the same statistic that
GATE 1a measures.
"""
from __future__ import annotations
import numpy as np


def leave_one_out_sigma(refs, stat_fn):
    """sigma of each reference against the others. Returns (M, T)."""
    out = []
    for i in range(len(refs)):
        others = [r for j, r in enumerate(refs) if j != i]
        out.append(stat_fn(refs[i], others))
    T = min(len(s) for s in out)
    return np.stack([s[:T] for s in out])


def _bin_scale(loo, n_bins):
    """Per-time-bin median of the LOO curves -- a shape reference, not a
    threshold. Floored away from zero (e.g. R2 is exactly 0 almost
    everywhere for a defect-free scene, which is correct, not degenerate)."""
    M, T = loo.shape
    edges = np.linspace(0, T, n_bins + 1).astype(int)
    scale = np.ones(T)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        vals = loo[:, lo:hi]
        finite = vals[np.isfinite(vals)]
        scale[lo:hi] = max(float(np.median(finite)), 1e-6) if finite.size else 1e-6
    return scale


def _defined_cutoff(loo, min_defined_frac):
    """Last index (exclusive) up to which at least min_defined_frac of the
    M leave-one-out curves are still FINITE -- excluding both NaN
    ("undefined, no evidence yet") AND +inf ("this reference has gone
    permanently and is now the worst possible value, forever"). Getting
    this wrong is an easy trap, hit directly while building this: +inf
    IS real information, but it's exactly the thing that swamps the
    whole-path scalar `c` once too many references carry it -- counting
    it as "defined" (as an earlier version of this function did) finds no
    cutoff at all, since a reference that goes permanently inf still
    counts as "has a value" at every later t. What actually matters is
    whether a reference has gone permanently inf YET, which -- because
    that transition is one-way for a reference that never returns to
    visibility -- is exactly captured by "is this reference's CURRENT
    value finite" at each t. Returns T (no truncation) if the
    finite-fraction never drops below the floor.

    Why this is needed, not just the NaN/inf fix above: reducing every LOO
    curve to a SINGLE whole-path scalar `c` (the whole design in this
    module's own module docstring) means one badly-behaved region degrades
    `theta` EVERYWHERE, not just there -- `theta(t) = c * scale(t)`, and if
    `c` is legitimately inf because most references have a permanent,
    undefined tail (occlusion_corridor's own by-design "stops behind the
    wall forever" outcome for ~2/3 of realizations), `theta` becomes inf
    across the WHOLE trajectory, including an early window where the
    statistic was perfectly well-behaved and could have caught a real
    defect. Measured directly on occlusion_corridor: without this cutoff,
    theta is inf at every single time bin, not just the occluded tail.
    Bounding calibration to where the ensemble stays sufficiently observed
    restores a meaningful threshold for the well-observed region and is
    honest about the rest -- R5 genuinely can't assess what it can't see,
    for most of the ensemble, past that point (AGENT.md defect #13)."""
    finite_frac = np.isfinite(loo).mean(axis=0)
    below = np.where(finite_frac < min_defined_frac)[0]
    return int(below[0]) if len(below) else loo.shape[1]


def estimate_threshold(refs, stat_fn, alpha=0.01, n_bins=10, min_defined_frac=None):
    """Whole-path calibrated threshold. Returns (T,).

    NaN vs inf in `loo`, preserved through this function rather than
    collapsed (AGENT.md defect #13): a sigma_k of NaN at some instant means
    "undefined, no evidence either way" (e.g. a reference is temporarily,
    legitimately unmeasurable -- occluded but not gone); a sigma_k of inf
    means "a genuine, maximal violation" (e.g. `sigma_kinematic`'s handling
    of an object that's actually gone). An earlier version forced every
    non-finite ratio to inf here, which silently promoted ordinary,
    expected occlusion into "maximal violation" for whole-path calibration
    -- fine for a scenario where occlusion is rare, but on
    occlusion_corridor enough LOO curves ended up with an all-inf tail that
    `np.quantile`'s tie-interpolation hit `inf - inf = NaN`, breaking
    threshold estimation entirely.

    min_defined_frac bounds calibration to the prefix where at least this
    fraction of the reference ensemble is still finite (see
    `_defined_cutoff`) -- for `t` past that point, `theta(t) = inf`: the
    honest statement that this statistic can no longer be meaningfully
    calibrated there, not a claim that nothing can go wrong. Precedence
    (events.py) means a higher-ranked risk (e.g. R2/existence) remains the
    operative test during that stretch regardless.

    Defaults to `1 - alpha`, not some fixed value like 0.5 -- and this
    isn't a stylistic choice, it's forced by what `method="higher"` (the
    quantile call below) actually needs. With M references and target
    false-positive rate alpha, the (1-alpha) quantile is approximately the
    ceil(alpha*M)-th worst of the M path_max values; at M=30, alpha=0.01,
    that's the SINGLE worst value (ceil(0.3)=1) -- so even one reference
    with a single inf entry anywhere inside the calibration window is
    enough to make `c` (and therefore theta EVERYWHERE, not just near that
    entry -- see this function's own note above) infinite again. A fixed
    0.5 threshold was tried first and measured directly to be nowhere near
    strict enough at this M/alpha (confirmed: theta stayed inf across the
    ENTIRE trajectory, not just the intended tail). Requiring
    min_defined_frac=1-alpha by default means the window can tolerate
    roughly alpha's own share of dropouts before truncating, matching the
    tolerance the quantile itself has -- self-consistent rather than a
    second, independently-guessed knob. For a statistic/scenario with no
    NaN/inf at all (the overwhelmingly common case -- ramp_descent,
    sigma_existence anywhere, ground-truth trajectories in general),
    finite_frac is 1.0 throughout regardless of threshold and this is a
    strict no-op: cutoff=T, identical behavior to before this parameter
    existed.
    """
    loo = leave_one_out_sigma(refs, stat_fn)          # (M, T)
    T = loo.shape[1]
    if min_defined_frac is None:
        min_defined_frac = 1 - alpha
    cutoff = _defined_cutoff(loo, min_defined_frac)
    loo_cal = loo[:, :cutoff]

    scale_cal = _bin_scale(loo_cal, n_bins)             # (cutoff,)
    ratio = loo_cal / scale_cal                          # NaN/inf pass through scale (always finite, positive)
    with np.errstate(invalid="ignore"):
        path_max = np.nanmax(ratio, axis=1)             # (M,) worst DEFINED deviation per LOO curve
    # a reference with literally no defined evidence anywhere on the
    # calibration window (all-NaN row) contributes nothing, not a spurious
    # value -- push it to the bottom of the distribution rather than
    # propagate NaN into quantile
    path_max = np.where(np.isnan(path_max), -np.inf, path_max)

    # method="higher" (an actual order statistic, not linear interpolation)
    # is required, not stylistic: when the true (1-alpha) quantile position
    # falls among TIED +inf entries -- which is the correct, legitimate
    # answer exactly when >alpha fraction of references have a genuine,
    # permanent violation within the calibration window -- numpy's default
    # linear interpolation computes inf-inf internally and returns NaN even
    # though every value involved is a real, valid inf. "higher" also
    # happens to be the conservative choice for the false-positive-rate
    # guarantee this function exists to provide (3.3): it never
    # UNDER-estimates c relative to the target alpha.
    c = np.quantile(path_max, 1 - alpha, method="higher")

    theta = np.full(T, np.inf)
    theta[:cutoff] = c * scale_cal
    return theta