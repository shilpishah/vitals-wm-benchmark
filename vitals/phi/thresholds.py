"""Threshold estimation for the re-identification similarity cutoff. Same
philosophy as detect/thresholds.py: estimated from data, never hand-picked
(AGENT.md 3.3) -- "never hardcode a threshold to make a test pass."

What "positive" and "negative" mean here, and an honest limitation:

  Positive: DINO similarity between an object's embedding just before it's
  lost and its own embedding once genuinely re-visible, one pair per
  reference rollout that has a real occlusion+reappearance. No
  leave-one-out needed the way detect/thresholds.py needs it -- each
  reference supplies its own positive pair directly, there's no "compare
  against the other M-1" step because there's nothing else to compare a
  same-object gap against.

  Negative: similarity between that same pre-loss embedding and every
  OTHER candidate the automatic mask generator proposes on the
  reappearance frame (background clutter, render artifacts, shadow
  regions) -- collected as a free byproduct of the real search, not
  synthesized separately.

  Honest limitation: current scenarios are single-object (K=1), so there
  is no genuinely different, similar-looking SECOND object to calibrate
  against yet -- only "the object vs. background clutter." That bounds how
  informative the negative distribution can be. Revisit the moment a
  multi-object scenario exists; don't quietly assume this generalizes
  until then.
"""
from __future__ import annotations
import numpy as np


def estimate_similarity_threshold(negative_scores, alpha=0.01):
    """theta = the (1-alpha) quantile of the NEGATIVE distribution -- a
    spurious/background candidate is accepted as a match at rate alpha,
    the same stated-false-accept-rate construction as every sigma_k
    threshold in detect/thresholds.py.

    Known failure mode, measured directly on this repo's own data: this
    rule is blind to where the POSITIVE distribution sits. With M=6
    calibration trajectories it produced theta=0.322 while a genuine
    positive pair (verified via the ground-truth mask, not just a noisy
    automatic-mask-generator candidate) measured 0.245-0.295 -- comfortably
    BELOW threshold. Controlling the false-accept rate alone doesn't
    control the false-REJECT rate, and when the two distributions overlap
    (as they clearly do here), a pure-FPR threshold can end up rejecting
    most genuine matches. Kept for reference / comparison; prefer
    estimate_balanced_threshold when positive pairs are available, which
    they always are here (each calibration trajectory supplies its own)."""
    negative_scores = np.asarray(negative_scores, dtype=float)
    if len(negative_scores) == 0:
        raise ValueError("no negative scores to calibrate against")
    return float(np.quantile(negative_scores, 1 - alpha))


def estimate_balanced_threshold(positive_scores, negative_scores):
    """theta = the cutoff that maximizes Youden's J (true-accept rate minus
    false-accept rate) over the observed scores. Still estimated, not
    chosen (3.3) -- the CRITERION (maximize J) is fixed and principled;
    only the resulting theta comes from data, and it comes from BOTH
    distributions, not just the negative one. This is the standard,
    well-established way to set a binary threshold when both an accept-
    and a reject-error matter and the two score distributions may overlap
    -- exactly this repo's situation, per estimate_similarity_threshold's
    docstring."""
    positive_scores = np.asarray(positive_scores, dtype=float)
    negative_scores = np.asarray(negative_scores, dtype=float)
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        raise ValueError("need both positive and negative scores to balance")
    candidates = np.unique(np.concatenate([positive_scores, negative_scores]))
    best_theta, best_j = float(candidates[0]), -1.0
    for theta in candidates:
        tpr = (positive_scores >= theta).mean()
        fpr = (negative_scores >= theta).mean()
        j = tpr - fpr
        if j > best_j:
            best_j, best_theta = j, float(theta)
    return best_theta


def separability_report(positive_scores, negative_scores, theta):
    """Diagnostic, not a pass/fail gate by itself: if positive scores don't
    sit clearly above theta, DINO isn't separating this case, and that's a
    finding to report (per AGENT.md 8.1's "a measurable instrument knows
    when it has failed"), not a threshold to force until it looks right."""
    positive_scores = np.asarray(positive_scores, dtype=float)
    negative_scores = np.asarray(negative_scores, dtype=float)
    return {
        "theta": theta,
        "n_positive": len(positive_scores),
        "n_negative": len(negative_scores),
        "positive_mean": float(positive_scores.mean()) if len(positive_scores) else float("nan"),
        "positive_min": float(positive_scores.min()) if len(positive_scores) else float("nan"),
        "negative_mean": float(negative_scores.mean()) if len(negative_scores) else float("nan"),
        "negative_max": float(negative_scores.max()) if len(negative_scores) else float("nan"),
        "true_accept_rate": float((positive_scores >= theta).mean()) if len(positive_scores) else float("nan"),
        "false_accept_rate": float((negative_scores >= theta).mean()) if len(negative_scores) else float("nan"),
    }
