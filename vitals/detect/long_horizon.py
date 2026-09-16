"""R4 (long-horizon / cumulative consistency) -- `AGENT.md`'s own "Future
measurement gaps" writeup (2026-09, repurposed from the original,
never-built "identity" reading of R4 -- see that writeup for why identity
didn't hold up as a distinct, irreducible category).

The gap this closes: every other detector here (`detect/events.py`'s own
`_first_sustained_crossing`) is calibrated per time bin and requires a
SUSTAINED crossing to count -- correct for rejecting single-frame noise,
but structurally blind to a slow, systematic, individually-sub-threshold
deviation that only becomes real evidence once accumulated over the whole
horizon (a bounce sequence gaining a little height each cycle; a model
that briefly loses and reacquires an object many times, each blip too
short to trip R1's own persistence window but cumulatively suspicious).

NOT a new physical property, NOT energy conservation (every real scenario
here is deliberately dissipative -- a naive "must stay constant" rule
would be wrong by design on every one of them). It's a different
STATISTICAL LENS applied to whichever of the existing, already-calibrated
`sigma_k(t)` signals (R1/R2/R3) are active for a scenario, producing a
callable that matches the exact same `fn(traj, refs) -> sigma(t)`
contract every other detector here already has, so it plugs into
`estimate_threshold`/`extract_event` unchanged.

Deliberately scored as an INDEPENDENT event, never merged into the same
`STATS` dict as R5/R1/R2/R3 -- see `AGENT.md`'s own writeup: `extract_
event` returns exactly one winning Event per episode, so merging R4 in
would let an early R1/R2/R3 hit silently suppress a real long-horizon
finding, the opposite of what a per-failure-mode-specific detector family
is for.

TWO real bugs found and fixed here (2026-09) by validating against real
reference-ensemble data before trusting the design, not by inspection
alone -- worth recording, both are genuine traps this codebase's OTHER
detectors already had to learn to avoid:

1. First version normalized the candidate's own sigma(t) against
   `thresholds._bin_scale`'s raw per-bin reference MEDIAN. Exactly the
   degenerate-floor trap `sigma_interpenetration`'s own docstring already
   documents for a different reason: that raw baseline floors to
   `thresholds.py`'s `1e-6` "no signal" fallback almost everywhere for
   R1 (existence) in a clean reference ensemble -- dividing any nonzero
   candidate deviation by that floor blew the ratio up by 6+ orders of
   magnitude (observed directly: ~6e12 against real cached cosmos data,
   not a subtle bug).

2. Switching to `sigma(t) / theta(t)` (the already-calibrated threshold)
   fixed the floor problem but traded it for a worse one: `theta_R1`
   calibrates to EXACTLY 0.0 in practice (confirmed directly against
   real reference data, not assumed -- correct on R1's own terms, an
   object's presence never fractionally deviates in a clean reference,
   so ANY nonzero deviation is already real evidence). Dividing by an
   exact 0 is undefined, and even where theta is merely CLOSE to the
   reference's own typical value (checked directly: real R3 calibrations
   here often sit only 5-30% above typical, not some large multiple),
   `sigma/theta` leaves too little headroom for a fixed slack to
   distinguish a real drift from the reference ensemble's OWN ordinary
   fluctuation -- verified directly: an honest sub-threshold-drift test
   fixture stayed censored even though the drift was real, because
   references were ALSO accumulating CUSUM signal under pure noise.

   Fixed by normalizing the GAP between typical and threshold instead:
   `r(t) = (sigma(t) - baseline(t)) / (theta(t) - baseline(t))` --
   `baseline(t)` is `thresholds._bin_scale`'s own per-bin reference
   MEDIAN (reused, not reimplemented), `theta(t)` the calibrated
   threshold; r=0 means exactly typical, r=1 means exactly at the
   instant-crossing threshold. Bounded, channel-scale-independent, and
   `theta(t) - baseline(t) <= 0` (R1's own case: both are exactly 0) is
   the CORRECT degeneracy condition -- it directly says "there is no gap
   between typical and threshold for this channel," which is precisely
   when there is no sub-threshold regime left to accumulate.
"""
from __future__ import annotations
import numpy as np

from .thresholds import leave_one_out_sigma, _bin_scale


def cusum_from_sigma(sigma_t, baseline_t, theta_t, slack=0.3):
    """One-sided CUSUM (cumulative sum control chart), in a bounded,
    channel-scale-independent GAP-FRACTION space -- `r(t) = (sigma(t) -
    baseline(t)) / (theta(t) - baseline(t))`: 0 at the reference's own
    per-bin typical value, 1 exactly at the instant-crossing threshold
    (see module docstring for why NOT a raw ratio against either
    `baseline` or `theta` alone -- both were tried, both found broken
    against real data before this one was). `S(t) = max(0, S(t-1) +
    (r(t) - slack))` -- grows LINEARLY under a persistent bias that keeps
    running more than `slack` fraction of the way from typical toward
    the instant threshold, resets toward zero whenever the candidate
    falls back below that fraction, so a single transient spike gets
    absorbed then decays away over subsequent normal frames rather than
    being remembered forever (unlike an unclamped running sum).

    `slack=0.3`: accumulate once running more than 30% of the way from
    "typical" to "instant failure" -- meaningful relative to what "too
    much" already means for THAT channel's own calibrated gap, not an
    arbitrary absolute number. A STARTING value, not a measured one --
    check directly against real reference-ensemble CUSUM distributions
    once this is calibrated against a real scenario, same "measured
    directly, not assumed" discipline M2.5 already applied scaling R2's
    own M from 30 to 100, rather than trusting a first guess.

    NaN in `sigma_t`, OR `theta_t - baseline_t <= 0` (the real,
    confirmed-not-hypothetical R1 case -- see module docstring), carries
    `S` forward UNCHANGED at that frame, never resets it and never
    fabricates a value -- same "hold the last defined value" convention
    `sigma_interpenetration` already uses past its own `Tc`. A channel
    with no gap between typical and threshold has no sub-threshold
    regime to contribute here at all, correctly contributing nothing
    rather than blowing up."""
    T = len(sigma_t)
    sigma_t = np.asarray(sigma_t, dtype=float)
    baseline_t = np.asarray(baseline_t, dtype=float)
    theta_t = np.asarray(theta_t, dtype=float)
    denom = theta_t - baseline_t
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 1e-9, (sigma_t - baseline_t) / denom, np.nan)
    S = np.zeros(T)
    prev = 0.0
    for t in range(T):
        if np.isfinite(r[t]):
            prev = max(0.0, prev + (r[t] - slack))
        S[t] = prev
    return S


def make_long_horizon_sigma(base_stats, refs, thetas, n_bins=10, slack=0.3):
    """base_stats: the SAME {risk: sigma_k_fn} dict already built for
    primary R5/R1/R2/R3 scoring on this scenario -- reused directly, not
    rebuilt, so R4 always tracks whichever channels are actually active
    there (e.g. R2 only joins for a P3 manifest, and R4 follows that same
    scenario-driven set automatically, no separate gating logic needed
    here). `refs`/`thetas`: the SAME reference ensemble and {risk:
    theta(t)} dict the caller already built via `estimate_threshold(refs,
    fn, alpha)` for those exact channels -- `thetas` reused directly
    (never recomputed); `refs` used only to derive each channel's own
    per-bin baseline via `thresholds._bin_scale` on its own leave-one-out
    curves (the SAME internal step `estimate_threshold` already performs
    for calibration -- reused, not duplicated).

    Returns ONE callable, `sigma_R4(traj, others) -> S(t)`, matching
    every other detector's own `fn(traj, refs)` contract -- the max,
    over every base channel, of that channel's own CUSUM signal
    (`cusum_from_sigma` above). max, not mean: one channel running a
    persistent bias is real evidence regardless of how well-behaved the
    others are (`sigma_interpenetration`'s own one-sided logic, reused
    again, not reinvented a third time)."""
    baselines = {}
    for risk, fn in base_stats.items():
        loo = leave_one_out_sigma(refs, fn)
        baselines[risk] = _bin_scale(loo, n_bins)

    def sigma_R4(traj, others):
        signals = []
        for risk, fn in base_stats.items():
            sigma_t = fn(traj, others)
            base = baselines[risk]
            theta_t = thetas[risk]
            T = min(len(sigma_t), len(base), len(theta_t))
            signals.append(cusum_from_sigma(sigma_t[:T], base[:T], theta_t[:T], slack=slack))
        if not signals:
            return np.array([])
        T = min(len(s) for s in signals)
        with np.errstate(invalid="ignore"):
            return np.max(np.stack([s[:T] for s in signals]), axis=0)

    return sigma_R4
