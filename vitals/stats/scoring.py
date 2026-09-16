"""P6 (uncertainty calibration) -- proper scoring rules for a model's
ENSEMBLE of continuations, the half of the design the survival pipeline
does not cover. Built 2026-09-11 (AGENT.md layout: "scoring.py MISSING --
needed for P6 (M7)"); nothing in the repo consumed it until now because
every population scores `candidates[0]` only. It becomes live the moment a
population asks a model for n_samples > 1.

Two questions, two kinds of score (AGENT.md §3.9: FAIR versions only):

1. **Skill against the realization.** The model's N samples are the
   ensemble, the held-out true rollout (never in the reference set, §3.7)
   is the observation. `fair_crps` per coordinate and `fair_energy_score`
   over the whole state vector are strictly proper: minimized in
   expectation only by the true predictive distribution, so a model
   cannot game them by hedging (too wide) or by collapsing to a point
   (too narrow). Both use the FAIR estimator -- the ensemble-spread term
   divides by N(N-1), not N^2 -- which removes the finite-N bias (Ferro
   2014; Zamo & Naveau 2018) so N=5 and N=50 are comparable and neither
   is rewarded for being larger. `rank_histogram` and `spread_skill`
   read calibration directly: a calibrated ensemble ranks the truth
   uniformly and has spread ~= its own error.

2. **Spread against the real ambiguity.** The reference ensemble R (M
   perturbed rollouts from the same conditioning) IS the ambiguity the
   physics leaves open. `fair_energy_distance` between the model's
   sample cloud and R is zero in expectation iff the two distributions
   coincide (Szekely & Rizzo) -- an unbiased two-sample statistic, so a
   model whose samples are physically plausible but too confident (or
   too diffuse) is caught here even when each sample would pass the
   R1-R4 detectors on its own. This is the score the survival pipeline
   structurally cannot produce: a first-crossing test looks at ONE
   sample at a time.

Everything is vectorized over an arbitrary trailing shape with the
ensemble on axis 0, returns per-element values (never pre-averaged, so a
caller can plot a score against horizon), and propagates NaN where an
input is NaN (an undefined comparison stays undefined -- the same
convention every detector here follows). The trajectory-level wrappers
at the bottom apply the object-count rule of §3.8: candidate and
reference may disagree on K; only the shared leading objects are scored,
and frames where an object is absent are NaN, never coerced to 0.
"""
from __future__ import annotations
import warnings
import numpy as np


def _nanmean(a, axis):
    """np.nanmean that stays silent on an all-NaN slice -- an all-absent
    frame is a legitimate, expected NaN here, not a warning condition."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


# ---------------------------------------------------------------------------
# array-level scoring rules (ensemble on axis 0)
# ---------------------------------------------------------------------------

def _check_ens(ens, obs):
    ens = np.asarray(ens, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if ens.ndim < 1 or ens.shape[0] < 2:
        raise ValueError(f"ensemble needs >= 2 members on axis 0, got shape {ens.shape}")
    if ens.shape[1:] != obs.shape:
        raise ValueError(f"ensemble trailing shape {ens.shape[1:]} != observation shape {obs.shape}")
    return ens, obs


def fair_crps(ens, obs):
    """Fair CRPS of a univariate ensemble, elementwise over the trailing
    shape. ens: (N, ...), obs: (...). Returns (...).

        CRPS_fair = mean_i |x_i - y| - (1 / (2 N (N-1))) sum_{i != j} |x_i - x_j|

    The plain estimator divides the second term by 2N^2 and is biased
    high by a factor that shrinks with N -- rewarding larger ensembles
    (§3.9's own reason for this rule)."""
    ens, obs = _check_ens(ens, obs)
    n = ens.shape[0]
    term1 = np.abs(ens - obs[None]).mean(axis=0)
    diff = np.abs(ens[:, None] - ens[None, :])          # (N, N, ...)
    term2 = diff.sum(axis=(0, 1)) / (2.0 * n * (n - 1))   # diagonal is 0, so the sum is over i != j
    return term1 - term2


def plain_crps(ens, obs):
    """The BIASED estimator, exposed only so its bias can be demonstrated
    in tests and never mistaken for the fair one. Do not score with it."""
    ens, obs = _check_ens(ens, obs)
    n = ens.shape[0]
    term1 = np.abs(ens - obs[None]).mean(axis=0)
    term2 = np.abs(ens[:, None] - ens[None, :]).sum(axis=(0, 1)) / (2.0 * n * n)
    return term1 - term2


def fair_energy_score(ens, obs, axis=-1):
    """Fair energy score of a MULTIVARIATE ensemble: the Euclidean norm is
    taken over `axis` (the state dimension) and the score is elementwise
    over what remains. ens: (N, ..., D), obs: (..., D). Returns (...).
    Reduces exactly to fair CRPS when D == 1."""
    ens, obs = _check_ens(ens, obs)
    n = ens.shape[0]
    term1 = np.linalg.norm(ens - obs[None], axis=axis).mean(axis=0)
    pair = np.linalg.norm(ens[:, None] - ens[None, :], axis=axis)   # (N, N, ...)
    term2 = pair.sum(axis=(0, 1)) / (2.0 * n * (n - 1))
    return term1 - term2


def fair_energy_distance(ens_a, ens_b, axis=-1):
    """Unbiased two-sample energy distance between two sample clouds of
    the same trailing shape (state dim on `axis`):

        ED = 2 E||a - b|| - E||a - a'|| - E||b - b'||

    with the within-sample expectations taken over DISTINCT pairs (the
    U-statistic), so E[ED] = 0 exactly when both clouds come from the same
    distribution and > 0 otherwise. ens_a: (Na, ..., D), ens_b: (Nb, ..., D).
    Returns (...). This is the P6 "does the model's spread match the real
    ambiguity" score: ens_a = model samples, ens_b = reference ensemble."""
    a = np.asarray(ens_a, dtype=np.float64)
    b = np.asarray(ens_b, dtype=np.float64)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"trailing shapes differ: {a.shape[1:]} vs {b.shape[1:]}")
    na, nb = a.shape[0], b.shape[0]
    if na < 2 or nb < 2:
        raise ValueError("both samples need >= 2 members")
    cross = np.linalg.norm(a[:, None] - b[None, :], axis=axis).mean(axis=(0, 1))
    within_a = np.linalg.norm(a[:, None] - a[None, :], axis=axis).sum(axis=(0, 1)) / (na * (na - 1))
    within_b = np.linalg.norm(b[:, None] - b[None, :], axis=axis).sum(axis=(0, 1)) / (nb * (nb - 1))
    return 2.0 * cross - within_a - within_b


def spread_skill(ens, obs):
    """Spread-skill decomposition, elementwise over the trailing shape.
    spread = ensemble standard deviation (unbiased), skill = |ensemble
    mean - obs|. A calibrated ensemble has E[spread^2] = E[skill^2]
    (+ 1/N correction), so averaging ratio = sqrt(mean spread^2 /
    mean skill^2) over many cases gives ~1; << 1 is over-confident,
    >> 1 is under-confident. Returns (spread, skill), each (...)."""
    ens, obs = _check_ens(ens, obs)
    spread = ens.std(axis=0, ddof=1)
    skill = np.abs(ens.mean(axis=0) - obs)
    return spread, skill


def spread_skill_ratio(spread, skill, n_members):
    """Aggregate the per-case (spread, skill) arrays into ONE calibration
    ratio, with the finite-ensemble correction (Fortin et al. 2014):
    a perfectly calibrated N-member ensemble has E[skill^2] =
    (1 + 1/N) E[spread^2]. NaN cases are dropped pairwise."""
    spread = np.asarray(spread, dtype=np.float64).ravel()
    skill = np.asarray(skill, dtype=np.float64).ravel()
    ok = np.isfinite(spread) & np.isfinite(skill)
    if not ok.any():
        return float("nan")
    return float(np.sqrt((1.0 + 1.0 / n_members) * np.mean(spread[ok] ** 2) / max(np.mean(skill[ok] ** 2), 1e-300)))


def rank_histogram(ens, obs, rng=None):
    """Rank of each observation within its N-member ensemble (0..N,
    ties broken at random), pooled over the whole trailing shape.
    Returns (counts, reliability_index): counts is (N+1,), uniform for a
    calibrated ensemble; U-shaped = under-dispersed (truth falls outside
    the ensemble), hump-shaped = over-dispersed. reliability_index is
    sum_k |f_k - 1/(N+1)| over bins (0 = perfectly flat, 2(1-1/(N+1))
    = all mass in one bin), Delle Monache et al. 2006. NaN cases are
    dropped."""
    ens, obs = _check_ens(ens, obs)
    rng = rng if rng is not None else np.random.default_rng(0)
    n = ens.shape[0]
    flat_e = ens.reshape(n, -1)
    flat_o = obs.reshape(-1)
    ok = np.isfinite(flat_o) & np.all(np.isfinite(flat_e), axis=0)
    flat_e, flat_o = flat_e[:, ok], flat_o[ok]
    below = (flat_e < flat_o[None]).sum(axis=0)
    tied = (flat_e == flat_o[None]).sum(axis=0)
    ranks = below + np.floor(rng.random(flat_o.shape) * (tied + 1)).astype(int)
    counts = np.bincount(ranks, minlength=n + 1).astype(np.float64)
    if counts.sum() == 0:
        return counts, float("nan")
    f = counts / counts.sum()
    ri = float(np.abs(f - 1.0 / (n + 1)).sum())
    return counts, ri


# ---------------------------------------------------------------------------
# trajectory-level wrappers (AGENT.md §3.8: K may differ; absent = NaN)
# ---------------------------------------------------------------------------

def _stack_positions(trajs, K):
    """(N, T, K, 3) positions from a list of Trajectory, restricted to the
    first K objects, NaN where an object is absent in that frame."""
    out = []
    for tr in trajs:
        p = tr.pos[:, :K].astype(np.float64).copy()
        p[~tr.present[:, :K]] = np.nan
        out.append(p)
    T = min(p.shape[0] for p in out)
    return np.stack([p[:T] for p in out])


def shared_object_count(*groups):
    """The object count every trajectory in every group shares (§3.8: a
    K mismatch is a finding, not an error -- score what is comparable)."""
    return min(tr.pos.shape[1] for g in groups for tr in g)


def score_episode(samples, truth, refs=None, t_c_frames=0):
    """P6 scores for ONE episode. samples: the model's N continuations
    (list[Trajectory], full-length, i.e. already concat'd with the prefix
    like everything detect/ scores); truth: the held-out realization;
    refs: the reference ensemble (list[Trajectory]) for the energy-
    distance-to-ambiguity score, optional. Frames before `t_c_frames`
    (the shared conditioning prefix) are excluded -- every sample is
    identical there by construction and would only dilute the scores.

    Returns a dict of PER-FRAME arrays (shape (T_scored,), NaN where
    undefined) plus the per-frame (spread, skill) pair, so the caller
    decides how to aggregate (a calibration-vs-horizon curve is the M7
    deliverable, not a single number):
      crps            fair CRPS, averaged over objects and coordinates
      energy_score    fair energy score over the full (K*3) state vector
      energy_distance fair energy distance model-vs-reference, or None
      spread, skill   for spread_skill_ratio / rank diagnostics
      n_samples, n_refs, K
    """
    if len(samples) < 2:
        raise ValueError("P6 scoring needs n_samples >= 2 -- a single sample has no spread to score")
    groups = [samples, [truth]] + ([refs] if refs else [])
    K = shared_object_count(*groups)
    S = _stack_positions(samples, K)[:, t_c_frames:]           # (N, T, K, 3)
    Y = _stack_positions([truth], K)[0, t_c_frames:S.shape[1] + t_c_frames]   # (T, K, 3)
    T = min(S.shape[1], Y.shape[0])
    S, Y = S[:, :T], Y[:T]

    crps = _nanmean(fair_crps(S, Y).reshape(T, -1), axis=1) if K else np.full(T, np.nan)
    es = fair_energy_score(S.reshape(S.shape[0], T, -1), Y.reshape(T, -1))
    spread, skill = spread_skill(S, Y)
    out = dict(crps=crps, energy_score=es, spread=spread, skill=skill,
               n_samples=len(samples), n_refs=len(refs) if refs else 0, K=K, energy_distance=None)
    if refs:
        R = _stack_positions(refs, K)[:, t_c_frames:t_c_frames + T]
        Tr = min(T, R.shape[1])
        out["energy_distance"] = fair_energy_distance(S[:, :Tr].reshape(S.shape[0], Tr, -1),
                                                      R[:, :Tr].reshape(R.shape[0], Tr, -1))
    return out


def aggregate_episodes(episode_scores):
    """Population-level P6 summary from a list of score_episode() dicts:
    per-frame means of each score across episodes (a calibration-vs-
    horizon curve, NaN-aware), the finite-ensemble-corrected spread-skill
    ratio, and the pooled rank-histogram reliability index computed from
    per-episode (spread, skill) is NOT possible -- ranks need the raw
    ensembles, so callers wanting the histogram must call rank_histogram
    on the stacked samples themselves. Returns a dict."""
    if not episode_scores:
        return {}
    T = min(len(e["crps"]) for e in episode_scores)
    def mean_curve(key):
        arrs = [e[key][:T] for e in episode_scores if e.get(key) is not None]
        return _nanmean(np.stack(arrs), axis=0) if arrs else None
    n = episode_scores[0]["n_samples"]
    spread = np.concatenate([e["spread"].ravel() for e in episode_scores])
    skill = np.concatenate([e["skill"].ravel() for e in episode_scores])
    return dict(n_episodes=len(episode_scores), n_samples=n, T=T,
                crps=mean_curve("crps"), energy_score=mean_curve("energy_score"),
                energy_distance=mean_curve("energy_distance"),
                spread_skill_ratio=spread_skill_ratio(spread, skill, n))
