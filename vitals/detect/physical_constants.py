"""Physical-constant recovery: a second, independent operationalization of
"is the physics right," complementary to the R1-R5 threshold-crossing
detectors in statistics.py/events.py. Inspired by WorldBench's parameter-
estimation track (arXiv:2601.21282 / world-bench.github.io) -- fit a known
physical constant from tracked motion, compare the fitted value against
the reference ensemble's own distribution of fitted values, the same
leave-one-out philosophy as thresholds.py.

Why this is a genuinely different check, not a duplicate of sigma_kinematic:
sigma_kinematic asks "did this trajectory's raw position deviate from the
reference band" -- a MODEL-FREE curve comparison. This asks "does the
IMPLIED PHYSICAL PARAMETER (a deceleration, a gravitational constant) match
the reference distribution" -- a MODEL-BASED, single-number-per-episode
comparison. A rollout could conceivably stay inside the sigma_kinematic
band while implying subtly wrong physics (e.g. correct position at the
sampled instants but the wrong underlying deceleration law), or could
fail sigma_kinematic for reasons that have nothing to do with the
constant itself (a one-frame position glitch). The two are meant to
corroborate or disagree, not to duplicate each other -- report both.

We already have exact ground-truth camera intrinsics/extrinsics from
render/ (no checkerboard calibration needed, unlike WorldBench's real-data
setup), so this works directly in metric 3D (from Trajectory.pos) for now.
A pixel-space version (2D centroid + known camera + depth assumption,
matching WorldBench's own method exactly) is the natural extension once
Phi produces tracked positions instead of ground truth (M4/M5 -- see
GATE 2, the instrument-tax pattern, applies here too: run this same fit on
Phi-measured positions and report the L0-vs-L1 delta as this component's
own error floor, once pose/3D-position extraction exists).

Currently implemented: deceleration recovery on occlusion_corridor's flat
rolling phase (one unknown, clean near-linear v(t) -- see module tests).
NOT implemented: gravity recovery on ramp_descent's incline (two entangled
unknowns, g and the incline friction coefficient, from one fit -- needs a
second data phase or a fixed known friction to disentangle; more setup than
the corridor case buys correspondingly little for a first version).

Important physics finding from validating this on the reference ensemble,
recorded here because it is easy to get wrong again: the ball starts with
pure translational velocity and zero spin (keyframe qvel has 0 angular
component), so motion has TWO physically distinct phases, not one constant
deceleration -- (1) a brief SLIDING phase (measured ~2.3-2.6 m/s^2,
duration t* ~ 0.0485*v0 seconds, standard rigid-body result for a sphere
transitioning to rolling-without-slipping under Coulomb friction) while
angular velocity spins up, then (2) a much longer, very stable ROLLING
phase (measured ~1.246 m/s^2, CV=0.12% across M=30 references) once
v = omega*r is reached. A naive "start the fit at t=0" window blends both
phases into one number that matches neither -- clustered tightly (CV=0.47%)
but at the WRONG value (~1.264, an artifact of how much transient leaked
into that particular window, not a real physical constant). Fitting must
start after the transient has cleared.

DEFAULT_CORRIDOR_WINDOW below (0.5s, 1.3s) was chosen with margin on both
sides: t*_max ~ 0.3s for the fastest reference in the lam=6.0 ensemble
(v0 up to ~4.6), so t_start=0.5 clears it; the slowest reference (v0~3.1)
doesn't stop until ~2.5s, so t_end=1.3 has ample margin before any
contact/settling artifacts near zero velocity.

On "true_value" comparison for this scenario: there isn't a clean textbook
ground truth to compare against, and constant_recovery_report's true_value
argument should NOT be fed synthetic_corridor.py's MU_FLAT=0.182 (a coarser,
whole-trajectory-averaged estimate from earlier calibration work, blending
both phases) -- doing so would silently assert a false ground truth,
exactly what this project's validation philosophy exists to prevent.
MuJoCo's rolling-friction model isn't a simple mu*g formula, so there is no
equally clean analytic value to assert here; the reference-ensemble
comparison (z-score against M references) is the legitimate check for this
scenario, and is also the one that generalizes to real, non-synthetic
video where no true_value could ever exist anyway.
"""
from __future__ import annotations
import numpy as np

# See "Important physics finding" above -- clears the sliding-to-rolling
# transient on one side, clears the earliest stop across the lam=6.0
# reference ensemble on the other. Validated by reference_constant_band
# giving CV=0.12% (vs 0.47% for a naive from-t=0 window) on M=30 refs.
DEFAULT_CORRIDOR_WINDOW = (0.5, 1.3)


def fit_deceleration(traj, obj=0, t_start=0.0, t_end=None, min_r_squared=0.95):
    """Fits x(t) = x0 + v0*t - 0.5*a*t^2 (constant deceleration) over
    [t_start, t_end] via least-squares on the quadratic coefficients, then
    reports (a_hat, r_squared). Window should be visible, pre-occlusion,
    genuinely-decelerating motion -- the caller's job to choose (this
    function does not know about occlusion or walls, it only fits what
    it's given, matching detect/statistics.py's separation of concerns).

    r_squared is a real fit-quality diagnostic, not decoration: a low value
    means the window wasn't actually well-described by constant
    deceleration (wrong phase selected, non-constant friction, noise) --
    per AGENT.md 8.1, "a measurable instrument knows when it has failed."
    Raises if r_squared < min_r_squared rather than silently returning a
    fit nobody should trust.
    """
    t = traj.t
    x = traj.pos[:, obj, 0]
    mask = (t >= t_start) & (t <= (t_end if t_end is not None else t[-1]))
    if mask.sum() < 4:
        raise ValueError(f"window has only {mask.sum()} samples, need >= 4 to fit a quadratic")

    tw, xw = t[mask], x[mask]
    # x = c0 + c1*t + c2*t^2  =>  a_hat = -2*c2
    coeffs = np.polyfit(tw, xw, 2)
    fitted = np.polyval(coeffs, tw)
    ss_res = np.sum((xw - fitted) ** 2)
    ss_tot = np.sum((xw - xw.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    a_hat = -2.0 * coeffs[0]
    if r_squared < min_r_squared:
        raise ValueError(f"fit quality too low (R^2={r_squared:.3f} < {min_r_squared}) -- "
                          f"window likely includes a non-constant-deceleration phase")
    return a_hat, r_squared


def fit_corridor_deceleration(traj, obj=0, min_r_squared=0.95):
    """fit_deceleration with DEFAULT_CORRIDOR_WINDOW -- the validated,
    transient-clearing window for occlusion_corridor specifically (see
    module docstring's "Important physics finding"). Use this, not
    fit_deceleration directly with hand-picked bounds, for anything
    touching occlusion_corridor: it's the one place the sliding-vs-rolling
    phase distinction is easy to silently get wrong again."""
    return fit_deceleration(traj, obj=obj, t_start=DEFAULT_CORRIDOR_WINDOW[0],
                             t_end=DEFAULT_CORRIDOR_WINDOW[1], min_r_squared=min_r_squared)


def reference_constant_band(refs, fit_fn, **fit_kwargs):
    """Fits the same constant on every reference realization independently
    (not leave-one-out in the sigma_k sense -- there's no "compare against
    the others" step needed to fit ONE trajectory's own constant, unlike a
    (T,) deviation curve). Returns the array of M fitted values, the
    natural object to compute a mean/std/CI band from for comparing a
    candidate's own fitted value against."""
    values = []
    for r in refs:
        try:
            a_hat, r_sq = fit_fn(r, **fit_kwargs)
            values.append(a_hat)
        except ValueError:
            continue   # a reference whose window doesn't fit well is excluded, not faked
    return np.array(values)


def constant_recovery_report(candidate_value, reference_values, true_value=None):
    """Per-episode diagnostic, not a survival-curve contributor (a fitted
    constant is one number per episode, not a (T,) curve with a
    meaningful crossing time -- see module docstring). z-score against the
    reference ensemble's own fitted-constant distribution; optionally also
    against a known true value, WHEN ONE GENUINELY EXISTS -- not every
    synthetic scenario has one. A textbook constant we set directly (e.g.
    g=9.81 for a future free-fall/gravity fit) qualifies; a derived/coarse
    estimate like occlusion_corridor's rolling deceleration does NOT (see
    module docstring -- MuJoCo's rolling friction isn't a clean mu*g
    formula, so there's no equally clean analytic value to assert there).
    Passing a true_value that isn't a real ground truth silently fakes
    validation -- don't. The reference-ensemble comparison (always
    available, real ground truth or not) is the one that generalizes to
    real-world video, where a true_value could never exist anyway."""
    ref_mean, ref_std = float(np.mean(reference_values)), float(np.std(reference_values))
    z = (candidate_value - ref_mean) / ref_std if ref_std > 1e-9 else float("inf")
    report = {
        "candidate_value": float(candidate_value),
        "reference_mean": ref_mean,
        "reference_std": ref_std,
        "n_reference": len(reference_values),
        "z_score": float(z),
    }
    if true_value is not None:
        report["true_value"] = float(true_value)
        report["reference_bias_vs_true"] = ref_mean - true_value
    return report
