"""sigma_k(candidate, others) -> (T,) deviation statistic.

Every statistic here must tolerate candidate.K != len(others[i]).K (AGENT.md
3.8) -- never zip or index candidate/reference objects against each other by
position. That is why sigma_existence compares object *counts*, and
sigma_kinematic compares a single tracked object's position against the
ensemble's own centroid/spread rather than against any one reference by index.

sigma_interpenetration (R2 / P3, 2026-08) is the L0-only (state-space)
version -- calibrated the same way P2/P4 were before either ever had an L1
(video/Phi) pass, not a violation of AGENT.md 6.2/10's own scoping of the
FULL, learned-relation-head P3 (video-side contact/support inference) as a
still-deferred research problem. Re-read 10 before extending this one
further than L0.
"""
from __future__ import annotations
import numpy as np


def _common_T(traj, others):
    return min([traj.T] + [o.T for o in others])


def sigma_existence(traj, others, obj=None):
    """|present-count(t) - ref mean| / ref std, in reference-count units
    (obj=None, the default -- unchanged from before this parameter
    existed). Whole-scene object COUNT, not any one object's own
    presence, is the right default: it is what makes `duplicate` (an
    EXTRA object appearing, at a K the reference ensemble never had)
    detectable at all -- see this module's own AGENT.md-3.8 note above on
    tolerating candidate.K != reference.K.

    obj: OPTIONAL, restricts the comparison to ONE specific object index's
    own presence (candidate vs. every reference, same index) instead of
    the whole-scene count (2026-08, a real bug found running GATE 2 on
    `occlusion_corridor_distractor` for the first time: its reference
    ensemble is K=2 -- ball + decoy, both present essentially always --
    but every real-video/real-model L1 candidate is K=1 (`track_and_
    reconstruct`'s own single-object return, M5.6's scoped design --
    the decoy/occluder is never merged into the returned Trajectory).
    Whole-count comparison then compares candidate_count=1 against
    reference_mean~2 at EVERY frame regardless of tracking quality --
    confirmed directly: null episodes with perfect SAM2 tracking (IoU
    0.93-0.95, zero re-id events) still fired R1 100% of the time,
    immediately, at t=0). A fixed index (obj=0, the primary/first-
    declared body -- deterministic MuJoCo scene construction, the SAME
    object every single rollout of the identical scene XML, not a
    per-rollout SAM2 detection order this module's own 3.8 note warns
    against trusting) is safe here, the same established precedent as
    `sigma_interpenetration`'s own fixed obj=0/target=1 above. Bind this
    ONCE per calling script's own STATS construction (same discipline as
    `sigma_kinematic`'s `kinematic_axes_for`), not per-call -- GATE 2
    (`run_gate2.py`) binds obj=0 for every scenario (a strict no-op for
    K=1 scenes, since count and single-object presence are identical
    there); `run_l0_demo.py`'s own GATE 1a/1b/survival scoring
    deliberately does NOT, since THAT is where `duplicate` is actually
    exercised and must stay detectable."""
    Tc = _common_T(traj, others)
    if obj is None:
        cand_count = traj.present[:Tc].sum(axis=1).astype(float)
        ref_counts = np.stack([o.present[:Tc].sum(axis=1).astype(float) for o in others])
    else:
        cand_count = traj.present[:Tc, obj].astype(float)
        ref_counts = np.stack([o.present[:Tc, obj].astype(float) for o in others])
    ref_mean = ref_counts.mean(axis=0)
    ref_std = np.maximum(ref_counts.std(axis=0), 1e-6)

    sigma = np.full(traj.T, np.nan)
    sigma[:Tc] = np.abs(cand_count - ref_mean) / ref_std
    if Tc < traj.T:
        sigma[Tc:] = sigma[Tc - 1] if Tc > 0 else np.inf
    return sigma


#  scenario -> tuple of axis indices (0=x, 1=y, 2=z) that are PHYSICALLY
#  MEANINGFUL to score kinematic plausibility over. None (every scenario
#  not listed here) means all 3, unchanged from before this existed.
#
#  ramp_descent / ramp_descent_high_friction (2026-08, AGENT.md M2.6):
#  this scene's own motion is, BY DESIGN, confined to the ramp's own X-Z
#  plane -- there is no real lateral (Y) dynamics anywhere in the scene
#  (scenes/ramp_descent*.xml has no lateral force/perturbation at all).
#  Every REFERENCE's own true Y stays within a few mm of 0 throughout, so
#  the reference ensemble's own natural Y-spread is tiny -- correctly so,
#  it reflects genuinely near-zero real variability. The problem this
#  causes is specific to L1 (video-measured) candidates, not real physics:
#  Phi's own reconstruction noise (an ALREADY-ACCEPTED ~0.10-0.16m floor
#  on this scene, matching the already-published ~0.11-0.13m figure) is
#  perfectly fine relative to X/Z's own much larger natural variance, but
#  becomes HUGE in units of Y's own near-zero natural variance -- enough
#  to spuriously cross a threshold calibrated assuming only real physics
#  variance, not instrument noise, would ever appear there. Confirmed
#  directly: GATE 2's own null (defect-free) population fired 10/10 via
#  this exact mechanism before this fix -- real, correct physics, scored
#  through the SAME video pipeline a real model's output goes through,
#  already failing 100% of the time for a reason with nothing to do with
#  whether the physics was actually right. Excluding Y here doesn't lower
#  a bar to make a defect-free population pass (AGENT.md 3.3/4) -- Y was
#  never a physically meaningful axis for THIS scenario's own defects to
#  begin with; every real defect this scenario's own mutants plant
#  (velocity_freeze, wrong_gravity, jitter -- AGENT.md M2.5/M2.6) is fully
#  expressed in X/Z, none of them rely on Y at all.
SCENARIO_KINEMATIC_AXES = {
    "ramp_descent": (0, 2),
    "ramp_descent_high_friction": (0, 2),
    # soft_drop (AGENT.md M9, 2026-09-14): velocity_x_only perturbation of
    # a dropped body -- x carries the ensemble's real spread, z the drop /
    # bounce (identical across references up to contact noise, so it runs
    # on the 1cm floor: that is what makes wrong_damping detectable at
    # all), y is never perturbed and is excluded for the same reason as
    # collision's.
    "soft_drop": (0, 2),
    # soft_ramp (2026-09-14): ramp_descent's own restriction -- the incline
    # is in the x-z plane; y carries only contact noise.
    "soft_ramp": (0, 2),
    # collision (2026-08): the SAME mechanism as ramp_descent's own Y
    # exclusion above, on Y AND Z instead of just Y -- collision.yaml's
    # own perturb_mode=velocity_x_only perturbs ONLY initial x-velocity,
    # so the M=30 reference ensemble's own Y/Z spread is exactly zero at
    # every frame (measured directly: ref_y_std=0.000000, ref_z_std on
    # the order of 1e-4-1e-3, both essentially machine-precision noise,
    # not real physical variability) by construction, not incidentally.
    # Confirmed directly: GATE 2's own null population fired 10/10 via
    # this exact mechanism (mean reconstruction error only ~0.04-0.06m,
    # already within this project's own accepted floor, made irrelevant
    # by dividing by a ~0 reference std). A real, separate reconstruction
    # bug (mujoco_renderer.py's own stereo-camera cam_pos, see that
    # module's own comment) was found and fixed alongside this and
    # measurably reduced the noise (0.061m -> 0.041m) but could not have
    # fixed this on its own -- ANY nonzero noise against an exactly-zero
    # reference std still fires. X is the only axis this scenario's own
    # defects (velocity_freeze, jitter -- collision's own wrong_gravity is
    # excluded entirely, see NEAR_FLAT_SCENARIOS) are expressed in; Y/Z
    # were never meaningful here to begin with (this scenario is confined
    # to the x-axis by the SAME perturb_mode that makes their reference
    # std zero).
    "collision": (0,),
    # occlusion_reemergence + block_stack (2026-09-12): the SAME mechanism
    # as collision, found the same way -- both manifests perturb only
    # along x (velocity_x_only / velocity_x_only_obj0), so the reference
    # Y/Z spread is exactly zero by construction, and GATE 2's first run
    # fired R3 on 9/10 null (defect-free) instances of each with mean
    # reconstruction error only 0.033m / 0.040m. Not extended to the
    # original occlusion_corridor (also velocity_x_only) here: its own
    # GATE 2 history was measured without this restriction and its
    # published populations were scored without it; changing it is a
    # separate, deliberate recalibration, not a side effect of adding
    # two new scenes.
    "occlusion_reemergence": (0,),
    "occlusion_reemergence_domA": (0,),   # M10 render-domain variants: same physics, same restriction
    "occlusion_reemergence_domB": (0,),
    "block_stack": (0,),
}


def kinematic_axes_for(scenario_name):
    """None (default: every axis) for any scenario not explicitly listed
    in SCENARIO_KINEMATIC_AXES -- a strict no-op for occlusion_corridor,
    projectile, and any future scenario, unless and until it's added here
    with its own measured justification, the same way ramp_descent's own
    entry was."""
    return SCENARIO_KINEMATIC_AXES.get(scenario_name)


def sigma_kinematic(traj, others, obj=0, axes=None):
    """Candidate object's distance from the reference centroid, in units of
    the reference ensemble's own per-axis spread at that instant.

    Each axis is normalised separately (diagonal-covariance Mahalanobis)
    before combining, not combined-then-normalised: this scene's dominant
    variability is along the roll axis (x), which grows to ~80x the
    vertical (z) spread once the ball is on flat ground. An isotropic
    combined-norm distance divided by a combined-norm spread lets that one
    high-variance axis swamp the ratio, silently hiding defects confined to
    a low-variance axis (e.g. wrong_gravity, which only perturbs z).

    axes: optional tuple of axis indices (0=x, 1=y, 2=z) to restrict the
    per-axis normalization/combination to -- None (default) uses all 3,
    identical to this function's own behavior before this parameter
    existed. Callers should get this from `kinematic_axes_for(scenario_
    name)`, not invent a value locally -- see that function's own
    docstring for why some scenario needs restricting at all (a real,
    measured finding, not a guess) and SCENARIO_KINEMATIC_AXES for the
    one place that mapping is allowed to live.

    NaN-vs-inf distinction (AGENT.md defect #13), load-bearing for whole-
    path threshold calibration -- do not collapse the two:
      - candidate genuinely gone (present=False, pos=NaN, e.g. the `vanish`
        mutant): sigma = +inf. This IS "maximally bad" -- it's what makes
        vanish detectable at all (GATE 1b), and it's a deliberate,
        permanent corruption, not something that resolves.
      - candidate temporarily unmeasured but still believed to exist
        (present=True, pos=NaN -- a live re-identification gap on a
        Phi-reconstructed Trajectory, or any other legitimate "can't see
        it right now" state): sigma = NaN, meaning genuinely UNDEFINED,
        not "worst possible." Kinematic plausibility of an object nobody
        can currently see isn't assessable -- treating it as inf falsely
        manufactures a maximal violation out of ordinary occlusion, and
        during a scenario where a large fraction of the reference ensemble
        is simultaneously occluded (occlusion_corridor's own by-design
        "stops behind the wall forever" outcome, ~2/3 of realizations),
        that false signal is common enough to blow up whole-path
        calibration to NaN (measured directly -- see AGENT.md defect #13).
      NaN is deliberately NOT collapsed to inf here; downstream consumers
      (thresholds.py's path_max reduction, events.py's `sigma > theta`
      crossing test) already treat NaN as "this instant contributes
      nothing" correctly on their own -- see thresholds.py for the one
      place that previously undid this distinction and has been fixed to
      match.
    """
    Tc = _common_T(traj, others)
    cand = traj.pos[:Tc, obj]                              # (Tc, 3)
    ref = np.stack([o.pos[:Tc, obj] for o in others])       # (M, Tc, 3)

    if axes is not None:
        # Restrict BEFORE computing centroid/spread/deviation, not after --
        # everything downstream (missingness check included) then operates
        # purely on the physically-meaningful axes, exactly as if the
        # excluded axis were never part of this Trajectory at all.
        cand = cand[:, axes]
        ref = ref[:, :, axes]

    centroid = np.nanmean(ref, axis=0)
    # 1cm floor, not 1e-6: an axis genuinely collapses to ~0 variance once
    # the ball comes to rest (all references converge to the same resting
    # height), and a micron-scale floor there just turns float noise into
    # huge spurious sigma spikes that blow out the whole-path calibration
    # in estimate_threshold (one such spike among the M LOO curves inflates
    # the single global multiplier for every bin, not just that one).
    spread = np.maximum(np.nanstd(ref, axis=0), 0.01)       # (Tc, len(axes) or 3), per-axis

    per_axis_dev = (cand - centroid) / spread                # (Tc, len(axes) or 3)
    dist = np.linalg.norm(per_axis_dev, axis=-1)

    cand_missing = np.isnan(cand).any(axis=-1)                          # (Tc,)
    cand_present = traj.present[:Tc, obj] if traj.present is not None else np.ones(Tc, bool)
    dist = np.where(np.isnan(dist) & cand_missing & ~cand_present, np.inf, dist)

    sigma = np.full(traj.T, np.nan)
    sigma[:Tc] = dist
    if Tc < traj.T:
        sigma[Tc:] = sigma[Tc - 1] if Tc > 0 else np.inf
    return sigma


def sigma_shape(traj, others, obj=0):
    """R7 shape -- the soft-body MATERIAL channel (AGENT.md M9): Mahalanobis
    distance of the candidate's SHAPE descriptors (Trajectory.shape,
    vitals.physics.softbody.SHAPE_DESCRIPTORS: silhouette area in px and
    principal-axis ratio) from the reference band's, per instant. Same
    form as the kinematic channel, same floor logic (per-descriptor floor
    = 1% of the reference mean area / 0.01 in ratio), same NaN conventions.
    NaN throughout when the candidate or references carry no shape state
    (every rigid scenario) -- the channel simply never fires there."""
    if traj.shape is None or any(o.shape is None for o in others):
        return np.full(traj.T, np.nan)
    Tc = _common_T(traj, others)
    # SCALE-FREE descriptors only (index 1 onward: axis ratio). Size (area)
    # belongs to R6 -- measured directly on the first GATE 1 run
    # (2026-09-14): with area inside R7 too, volume_leak fired R7 at
    # 1.07s instead of R6, because R7's tighter area floor crossed one
    # frame before R6's ratio statistic and precedence never got a tie
    # to resolve. One quantity, one channel.
    cand = traj.shape[:Tc, obj, 1:]                               # (Tc, S-1)
    ref = np.stack([o.shape[:Tc, obj, 1:] for o in others])       # (M, Tc, S-1)
    centroid = np.nanmean(ref, axis=0)
    floor = 0.005                                                  # half a percent of axis ratio
    spread = np.maximum(np.nanstd(ref, axis=0), floor)
    dist = np.linalg.norm((cand - centroid) / spread, axis=-1)
    sigma = np.full(traj.T, np.nan)
    sigma[:Tc] = dist
    if Tc < traj.T:
        sigma[Tc:] = sigma[Tc - 1] if Tc > 0 else np.nan
    return sigma


def sigma_conservation(traj, others, obj=0):
    """R6 (AGENT.md M9): silhouette-area CONSERVATION -- the soft-body
    analogue of interpenetration, a constraint the entity must keep to
    stay the same entity ("objects that squash without bulging, or bulge
    without squashing"). Statistic: the candidate's area ratio to its own
    frame-0 area, A(t)/A(0), compared with the reference band's ratio at
    the same instant, normalized by the band's spread with a 2% floor
    (area is a projected quantity; ordinary squash-and-bulge moves it by
    a few percent, which the band absorbs -- a leak or a puff does not).
    One-sided in EFFECT only through calibration, like R2: deviation in
    either direction counts, because losing area and gaining it are both
    conservation failures. NaN wherever undefined (no shape state, empty
    silhouette, or A(0) undefined)."""
    if traj.shape is None or any(o.shape is None for o in others):
        return np.full(traj.T, np.nan)
    Tc = _common_T(traj, others)
    a0 = traj.shape[0, obj, 0]
    cand = traj.shape[:Tc, obj, 0] / a0 if np.isfinite(a0) and a0 > 0 else np.full(Tc, np.nan)
    ref = np.stack([o.shape[:Tc, obj, 0] / o.shape[0, obj, 0] for o in others])   # (M, Tc)
    mean = np.nanmean(ref, axis=0)
    spread = np.maximum(np.nanstd(ref, axis=0), 0.02)
    sigma = np.full(traj.T, np.nan)
    sigma[:Tc] = np.abs(cand - mean) / spread
    if Tc < traj.T:
        sigma[Tc:] = sigma[Tc - 1] if Tc > 0 else np.nan
    return sigma


def sigma_interpenetration(traj, others, obj=0, target=1):
    """reference-typical separation / actual separation, between two
    tracked objects -- a RATIO, not a difference. ~1.0 when the candidate's
    own separation matches the reference ensemble's typical separation at
    that instant (nothing wrong); grows large as the two objects get
    closer together than the ensemble ever does (interpenetration); stays
    small (well below 1) when they drift FARTHER apart than usual, which
    is not a P3 violation and must never be flagged -- one-sided in EFFECT
    (a small ratio can't cross a threshold calibrated near/above 1.0)
    rather than by an explicit clip.

    Deliberately NOT a z-scored difference ((ref_mean-actual)/ref_std, the
    first version tried here): that quantity is ~0 for a "normal" LOO
    reference by construction (a difference from the mean, by definition),
    which sounds right but is fatal to estimate_threshold's own shared
    per-bin-median normalization (thresholds.py's `_bin_scale`) -- roughly
    half of any bin's M LOO curves land ABOVE the mean (difference clips
    to exactly 0, since only "closer than typical" should count) and half
    land below (a real, unclipped positive value), so the per-bin MEDIAN
    across those M values sits right at that 50/50 boundary and floors to
    thresholds.py's 1e-6 "no signal" fallback unpredictably often. When it
    does, a handful of OTHER references' genuinely modest deviations at
    that same bin get divided by ~1e-6 and blow the calibrated threshold
    up by 5+ orders of magnitude (confirmed directly: theta reached
    ~15,800 against a real defect's own natural value of ~1). R1/R3 never
    hit this failure mode not because they're immune to it, but because
    their own "near-zero" bins are LITERALLY always exactly zero in a
    clean reference ensemble (an object's presence count never fluctuates
    by chance; a position's Mahalanobis NORM across 3 axes is essentially
    never exactly 0) -- there's no 50/50 boundary for their median to land
    on unpredictably. A RATIO sidesteps this entirely: its typical value
    is a stable ~1.0 (never an atom at 0, never a 50/50 split), which is
    exactly the kind of well-behaved, non-degenerate scale `_bin_scale`'s
    median-based normalizer was already built to handle.

    NaN, never inf, when either object is missing: an existence failure
    (either object absent) is already R1/sigma_existence's job (AGENT.md
    3.6 -- one necessary property per detector). Manufacturing a SECOND
    finding (R2) out of the same missing-object evidence sigma_existence
    already covers would double-count one defect as two, and PRECEDENCE
    (detect/events.py) already ranks R1 ahead of R2 so R1 wins any
    simultaneous crossing regardless -- NaN here just avoids inventing a
    spurious R2 signal from evidence that already has an owner.
    """
    Tc = _common_T(traj, others)
    cand_obj, cand_tgt = traj.pos[:Tc, obj], traj.pos[:Tc, target]     # (Tc, 3) each
    cand_dist = np.linalg.norm(cand_obj - cand_tgt, axis=-1)           # (Tc,)

    ref_dist = np.stack([
        np.linalg.norm(o.pos[:Tc, obj] - o.pos[:Tc, target], axis=-1)
        for o in others
    ])                                                                  # (M, Tc)
    ref_mean = np.nanmean(ref_dist, axis=0)

    sigma_defined = ref_mean / np.maximum(cand_dist, 0.01)               # 1cm floor -- avoids a literal divide-by-zero at true overlap

    obj_present = traj.present[:Tc, obj] if traj.present is not None else np.ones(Tc, bool)
    tgt_present = traj.present[:Tc, target] if traj.present is not None else np.ones(Tc, bool)
    sigma_defined = np.where(obj_present & tgt_present, sigma_defined, np.nan)

    sigma = np.full(traj.T, np.nan)
    sigma[:Tc] = sigma_defined
    if Tc < traj.T:
        sigma[Tc:] = sigma[Tc - 1] if Tc > 0 else np.nan
    return sigma
