"""L0 demo: the whole statistical core, state-space only, no video.

Run this first. It answers the week-2 gate: does the null mutant terminate at
rate alpha, and do planted defects fire on the RIGHT risk at the RIGHT time?

Scenario is manifest-driven, not hardcoded -- each manifest is a fully
isolated, pre-registered experiment testing ONE necessary property
(AGENT.md 3.6). There is no combined scenario and there should never be one:
a model could nail incline kinematics and still hallucinate through an
occlusion, or vice versa, and a hybrid scene would make a failure ambiguous
about which property broke.

    python scripts/run_l0_demo.py                                  # ramp_descent, synthetic
    VITALS_MANIFEST=configs/manifests/occlusion_corridor.yaml \\
        VITALS_BACKEND=mujoco python scripts/run_l0_demo.py         # occlusion_corridor, mujoco
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import functools
import json
import os
import numpy as np
import yaml
from vitals.types import EpisodeSpec
from vitals.physics import make_backend
from vitals.mutants import library as mut
from vitals.detect.statistics import sigma_existence, sigma_kinematic, sigma_interpenetration, kinematic_axes_for
from vitals.detect.thresholds import estimate_threshold
from vitals.detect.events import extract_event
from vitals.stats.survival import (validity_interval, termination_profile,
                                   bootstrap_vi, kaplan_meier)

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND = os.environ.get("VITALS_BACKEND", "synthetic")
MANIFEST_PATH = os.environ.get("VITALS_MANIFEST", "configs/manifests/ramp_descent.yaml")

manifest = yaml.safe_load((ROOT / MANIFEST_PATH).read_text())
SPEC = EpisodeSpec(name=manifest["name"], scene=manifest["scene"],
                    target_property=manifest["target_property"], band=manifest["band"],
                    lam=manifest["lam"], n_reference=manifest["n_reference"],
                    horizon_s=manifest["horizon_s"], fps=manifest["fps"],
                    seed=manifest.get("seed", 0),
                    perturb_mode=manifest.get("perturb_mode", "full"))
M, LAM, ALPHA = SPEC.n_reference, SPEC.lam, 0.01

# scene= doubles as the synthetic-backend scenario selector (see physics/__init__.py)
roll = make_backend(BACKEND, scene=SPEC.scene)

# R4 (sigma_interpenetration) is added conditionally in main(), once the
# reference ensemble's own K is known -- every existing K=1 scenario must
# stay byte-identical to before this existed, not silently gain a 3rd,
# always-undefined (NaN) risk channel.
#
# R5's own axes restriction (AGENT.md M2.6) is bound ONCE here, from
# kinematic_axes_for(SPEC.name) -- None (every scenario but ramp_descent*)
# is a strict no-op, identical to calling sigma_kinematic directly. Bound
# at STATS-construction time, not per-call, so GATE 1a/1b's OWN threshold
# calibration and every candidate scored against it use the IDENTICAL
# axes-restricted statistic -- that consistency (one theta, computed once,
# applied unchanged to every L0 AND L1 candidate) is the whole reason this
# lives in STATS's own construction rather than being threaded through
# each call site separately.
STATS = {"R2": sigma_existence,
         "R5": functools.partial(sigma_kinematic, axes=kinematic_axes_for(SPEC.name))}


def build_reference(spec, M, seed0=1000):
    return [roll(spec, seed0 + i) for i in range(M)]


def evaluate(cand, refs, thetas, dt, t_max):
    sigmas = {k: fn(cand, refs) for k, fn in STATS.items()}
    return extract_event(sigmas, thetas, dt, t_max)


#  Scenarios where the tracked object stays close to the floor the whole
#  clip (minimal vertical excursion) -- wrong_gravity corrupts VERTICAL
#  motion specifically, so it's a near no-op on any of these, the same
#  symptom independently confirmed for TWO differently-named scenario
#  families now: occlusion_corridor* (AGENT.md M5.5/M5.7's own GATE 1b
#  fix) and collision (2026-08, Phase 1 scenario diversity -- both balls
#  stay at z~0.15 throughout, confirmed empirically before assuming this
#  generalizes). Prefix-matched (not exact-name), same reasoning as
#  occlusion_corridor* itself: a future near-flat scenario variant should
#  inherit this without another silent GATE 1b bug.
NEAR_FLAT_SCENARIOS = ("occlusion_corridor", "collision")


def build_demo_cases(base, horizon_s):
    """GATE 1b's demonstration mutants, chosen per scenario -- not the same
    five everywhere. wrong_gravity only means something where there's
    vertical motion to corrupt (ramp_descent, projectile); every
    NEAR_FLAT_SCENARIOS variant (occlusion_corridor's own base/distractor/
    moving-occluder, and collision -- all "object barely leaves the
    floor" geometry, only the scene dressing/property differs) gets
    `duplicate` instead, still an R2 mutant but the "erroneous extra
    object" side of existence rather than "object vanished". Matched by
    prefix, NOT exact name -- an earlier version checked `SPEC.name ==
    "occlusion_corridor"` exactly, which silently fell through to
    wrong_gravity (a no-op on this near-flat geometry) for occlusion_
    corridor_distractor/_moving, a real GATE 1b bug found while
    calibrating those two scenes (AGENT.md M5.5/M5.7).

    P3 manifests get their OWN case list (null + teleport only), not the
    R2/R5 set below -- a real bug, found directly (2026-08): including
    velocity_freeze here for occlusion_corridor_interpenetration made it
    fire R4 instead of its own expected R5, because mutating the ball's
    OWN kinematics incidentally changes its distance to the untouched
    decoy too, and PRECEDENCE ranks R4 ahead of R5. That's cross-talk
    between two DIFFERENT properties' demo cases sharing one manifest,
    which is exactly what 3.6/3.12's "one manifest tests one
    pre-registered property" already rules out elsewhere in this
    function (the occlusion_corridor* prefix-match branch below) --
    applied here too, not a new principle."""
    t_mid = round(horizon_s * 0.25, 2)
    if SPEC.target_property == "P3":
        return [("null", mut.null(base)), ("teleport", mut.teleport(base, t_star=t_mid))]
    cases = [
        ("null", mut.null(base)),
        ("vanish", mut.vanish(base, t_star=t_mid)),
        ("velocity_freeze", mut.velocity_freeze(base, t_star=t_mid)),
    ]
    if SPEC.name.startswith(NEAR_FLAT_SCENARIOS):
        cases.append(("duplicate", mut.duplicate(base, t_star=t_mid)))
    else:
        # projectile needs a stronger wrong_gravity distortion than
        # ramp_descent's own default (factor=0.6, severity=0.4) --
        # confirmed directly (AGENT.md M4.5's GATE 1b calibration writeup):
        # even after lam was cut to fix R5's threshold inflation, this
        # scene's R5 threshold (~8.2) is still high enough that factor=0.6
        # stays censored; factor=0.4 (severity=0.6) fires reliably.
        # ramp_descent's own default is untouched.
        wg_factor = 0.4 if SPEC.name == "projectile" else 0.6
        cases.append(("wrong_gravity", mut.wrong_gravity(base, factor=wg_factor)))
    # same story for jitter -- every NEAR_FLAT_SCENARIOS variant's own R5
    # threshold is low enough (~2.3-2.7) that sigma=0.05 fires reliably, but
    # ramp_descent's and projectile's own (~6.5, ~8.2) need sigma=0.15 to
    # fire at all -- confirmed directly for BOTH, not assumed from one:
    # sigma=0.05 stayed censored on ramp_descent too once actually checked
    # (a real, pre-existing gap, found incidentally while calibrating
    # projectile -- not introduced by it). collision fired fine at 0.15
    # ONLY while R5 was still scored on all 3 axes -- once collision was
    # restricted to X-only (SCENARIO_KINEMATIC_AXES, 2026-08, fixing the
    # null false-positive documented there) jitter's own isotropic noise
    # was mostly being caught via its Y/Z components against those axes'
    # near-zero reference std, the SAME fragile mechanism the restriction
    # exists to remove -- not a real X-axis detection. X-only jitter also
    # needs a SUSTAINED crossing (events.py's own pi=0.3s persistence,
    # ~9 consecutive frames @30fps) that per-frame IID noise on a single
    # axis doesn't reliably produce at small sigma -- confirmed directly
    # by sweeping sigma against theta on 3 held-out bases (seeds 9000-9002,
    # same convention as every other per-scenario mutant tuning in this
    # file): 0.15/0.4 both censored (longest sustained run 3-6 frames,
    # short of the 9 needed); 0.8 fires reliably on all 3 (longest run
    # 13-15 frames).
    jitter_sigma = 0.05 if SPEC.name.startswith("occlusion_corridor") else (
        0.8 if SPEC.name == "collision" else 0.15)
    cases.append((f"jitter sig={jitter_sigma}", mut.jitter(base, sigma=jitter_sigma)))
    return cases


def main():
    global STATS
    refs = build_reference(SPEC, M)
    dt, t_max = refs[0].dt, float(refs[0].t[-1])

    # P3-lite (2026-08): gated on target_property == "P3", NOT on K -- a
    # real bug, found directly by this change's own verification pass:
    # occlusion_corridor_distractor.yaml is K=2 (ball+decoy) but is
    # pre-registered as a P2 specificity test, not P3. Gating on K alone
    # silently added R4 to that manifest too, which changed its GATE 1b
    # outcome (velocity_freeze started firing R4 instead of R5 -- the
    # ball's own velocity_freeze mutation incidentally changes its
    # distance to the untouched decoy enough to cross R4's threshold, and
    # PRECEDENCE ranks R4 ahead of R5) and pushed its GATE 1a false-
    # termination rate from 0.05 to 0.067. Same 3.6/3.12 principle already
    # applied to build_demo_cases below: one manifest, one pre-registered
    # property, decided in advance -- a scene's own geometry (K) must
    # never silently expand what a manifest scores.
    if SPEC.target_property == "P3":
        assert refs[0].K >= 2, (
            f"{SPEC.name!r} is registered target_property: P3 but its scene has K={refs[0].K} "
            f"objects -- sigma_interpenetration needs obj=0/target=1 both real. Fix the manifest "
            f"or the scene, not this check.")
        STATS = dict(STATS, R4=sigma_interpenetration)

    # GATE 1a's own false-termination check is a rate on the UNION of every
    # active risk channel (extract_event fires if ANY sigma crosses its own
    # theta), so adding a 3rd channel gives real physics one more chance to
    # cross SOME threshold. First guess here was a Bonferroni-style alpha
    # split (alpha/n_channels) -- tried directly, then MEASURED, not
    # assumed: at M=30 references, estimate_threshold's own calibrated
    # value is completely FLAT from alpha=0.01 down to alpha=0.001 (checked
    # directly -- identical R4/R5 medians at every alpha in that range).
    # The whole-path reduction (each LOO curve collapsed to one worst-case
    # scalar before taking the cross-reference quantile, thresholds.py's
    # own design) means alpha this small is already selecting the single
    # MAXIMUM of only 30 scalars regardless of exactly how small alpha
    # gets -- there's no room left to tighten via alpha alone at this M.
    # The lever that actually works, confirmed by directly sweeping it
    # (M=30 -> fp=0.067, M=60 -> 0.033, M=100 -> 0.000): more reference
    # rollouts, which is what M is FOR -- a small M's own "worst case seen
    # so far" is an unreliable calibration basis once genuinely-clean
    # physics has enough natural variability (this scene's ball-vs-decoy
    # geometry does) to occasionally exceed anything in a 30-rollout
    # sample. Fixed in the manifest itself (n_reference: 100 for THIS
    # scenario only, configs/manifests/occlusion_corridor_interpenetration.
    # yaml) rather than here, so every other manifest's own M=30 stays
    # untouched.
    thetas = {k: estimate_threshold(refs, fn, alpha=ALPHA)
              for k, fn in STATS.items()}

    print(f"scenario={SPEC.name}  target_property={SPEC.target_property}  backend={BACKEND}")
    print(f"reference ensemble M={M}  lambda={LAM}  T_max={t_max:.1f}s")
    print(f"thresholds  R2 median={np.median(thetas['R2']):.3f}"
          f"  R5 median={np.median(thetas['R5']):.3f}"
          + (f"  R4 median={np.nanmedian(thetas['R4']):.3f}" if "R4" in thetas else "") + "\n")

    # --- Gate: instrument false-positive floor -----------------------------
    held_out = build_reference(SPEC, 60, seed0=9000)
    ev = [evaluate(h, refs, thetas, dt, t_max) for h in held_out]
    fp = sum(1 for e in ev if not e.censored) / len(ev)
    print(f"GATE 1a  null false-termination rate = {fp:.3f}   (target ~{ALPHA})")
    print("         " + ("PASS" if fp <= 5 * ALPHA else "FAIL -- detector or sigma is wrong") + "\n")

    # --- Gate: sensitivity, specificity, event-time bias -------------------
    base = held_out[0]
    cases = build_demo_cases(base, SPEC.horizon_s)
    print("GATE 1b  detector sensitivity / specificity")
    print(f"  {'mutant':<18}{'expect':<8}{'fired':<8}{'t*':>6}{'t_det':>8}{'bias':>8}")
    gate_1b_results = []
    for label, m in cases:
        e = evaluate(m.traj, refs, thetas, dt, t_max)
        fired = e.risk or "-"
        ts = f"{m.t_star:.2f}" if m.t_star is not None else "  -"
        td = f"{e.time:.2f}" if not e.censored else "cens"
        bias = (f"{e.time - m.t_star:+.2f}"
                if (m.t_star is not None and not e.censored) else "   -")
        ok = "OK " if fired == (m.risk_expected or "-") else "XX "
        print(f"  {ok}{label:<15}{str(m.risk_expected or '-'):<8}{fired:<8}{ts:>6}{td:>8}{bias:>8}")
        gate_1b_results.append(dict(label=label, risk_expected=m.risk_expected, t_star=m.t_star,
                                     risk_fired=e.risk, t_detected=None if e.censored else e.time,
                                     censored=e.censored))

    # --- Survival curve over a population of mutants -----------------------
    # P3-lite (2026-08): a P3-registered manifest gets its OWN 2-way
    # null/teleport mix (50/50), not the generic 4-way R2/R5 set -- same
    # fix, same reason, as build_demo_cases above: vanish/velocity_freeze
    # are OTHER properties' mutants, and including them here risks the
    # identical cross-talk (a ball-only R5 mutation incidentally tripping
    # R4 via its changed distance to the untouched decoy) contaminating
    # this manifest's own survival curve, not just its GATE 1b printout.
    if SPEC.target_property == "P3":
        mutant_kinds, mutant_probs = ["null", "teleport"], [0.5, 0.5]
    else:
        mutant_kinds, mutant_probs = ["null", "vanish", "velocity_freeze", "jitter"], [0.25, 0.25, 0.25, 0.25]
    pop = []
    rng = np.random.default_rng(7)
    t_lo, t_hi = SPEC.horizon_s * 0.15, SPEC.horizon_s * 0.7
    for i in range(80):
        b = roll(SPEC, 20000 + i)
        kind = rng.choice(mutant_kinds, p=mutant_probs)
        t_star = float(rng.uniform(t_lo, t_hi))
        m = (mut.null(b) if kind == "null" else
             mut.vanish(b, t_star=t_star) if kind == "vanish" else
             mut.velocity_freeze(b, t_star=t_star) if kind == "velocity_freeze" else
             mut.teleport(b, t_star=t_star) if kind == "teleport" else
             mut.jitter(b, sigma=0.04))
        pop.append(evaluate(m.traj, refs, thetas, dt, t_max))

    vi50 = validity_interval(pop, 0.5)
    lo, hi = bootstrap_vi(pop, 0.5, n_boot=400)
    prof = termination_profile(pop)
    cens = sum(e.censored for e in pop) / len(pop)

    print(f"\nSURVIVAL  (n={len(pop)} mixed-mutant rollouts)")
    print(f"  VI_50            = {vi50:.2f}s   95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  censoring        = {cens:.2f}")
    print(f"  termination profile = "
          + ", ".join(f"{k}:{v:.2f}" for k, v in prof.items()))

    grid, surv = kaplan_meier(pop)
    print("\n  S(t):  " + "  ".join(
        f"{t:.1f}s={s:.2f}" for t, s in zip(grid[::max(1, len(grid)//6)],
                                            surv[::max(1, len(surv)//6)])))

    # Persist the FULL result -- everything above only ever printed to the
    # console, downsampled even there (the survival curve print shows ~6
    # points; the full grid/surv arrays were computed then discarded). This
    # is the same data GATE 1a/1b/the survival analysis already produce,
    # just actually kept somewhere accessible after the process exits --
    # not a new computation, not a new "scorecard" concept (this project's
    # own 3.11 rules out a single aggregate scalar; a survival curve +
    # termination profile is exactly the non-scalar report that's supposed
    # to stand in for one).
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"l0_demo_{SPEC.name}_{BACKEND}.json"
    result = dict(
        scenario=SPEC.name, target_property=SPEC.target_property, backend=BACKEND,
        M=M, lam=LAM, t_max=t_max,
        # General over whatever's in STATS (R2/R5 always, R4 added only for
        # K>=2 scenarios) rather than hardcoding two keys -- otherwise a
        # P3 manifest's own calibrated R4 threshold would silently never
        # reach the persisted record. nanmedian since R4 is legitimately
        # NaN whenever either tracked object is absent (statistics.py's own
        # sigma_interpenetration docstring).
        thresholds_median={k: float(np.nanmedian(v)) for k, v in thetas.items()},
        gate_1a=dict(false_termination_rate=fp, target_alpha=ALPHA, passed=bool(fp <= 5 * ALPHA)),
        gate_1b=gate_1b_results,
        survival=dict(
            n=len(pop), vi50=vi50, vi50_ci95=[lo, hi], censoring_rate=cens,
            termination_profile=prof,
            kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),  # FULL curve, not downsampled
            # Raw per-episode events, not just the derived VI/profile/curve
            # above -- lets bootstrap_vi_delta (or any other re-analysis,
            # e.g. comparing this population against a DIFFERENT scenario's
            # or a future real model's) run later WITHOUT re-simulating.
            # Saved for exactly this reason (2026-08): "how do I assess
            # significance" needs the raw population, not just its summary
            # statistics, and re-deriving it means re-running physics.
            events=[dict(time=e.time, risk=e.risk, censored=e.censored) for e in pop],
        ),
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nfull result (including the complete, non-downsampled survival curve) -> {out_path}")


if __name__ == "__main__":
    main()
