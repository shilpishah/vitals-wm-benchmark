"""First sustained threshold crossing, with precedence.

A crossing must hold for `pi` seconds (the persistence window) before it
counts -- a single noisy frame over threshold is not a failure. When more
than one risk crosses near the same instant, the precedence rule (AGENT.md
3.6, derived from the P1-P6 dependency order) decides which one is reported:
a violation at a lower level is undefined once a higher level has already
broken, so the higher-precedence risk wins, not the numerically-earliest one.

v1 only ever registers R1 and R3 (README scope); R2 (interpenetration)
was added 2026-09 for P3-registered scenarios. R5 (frame, scoped but
deprioritized) and R4 would join this SAME list if either R5 or the
original "identity" reading of R4 were ever built -- but R4 was
repurposed (2026-09, AGENT.md's own "Future measurement gaps" writeup)
to long-horizon/cumulative consistency, which is deliberately NOT part
of this precedence race at all (see that writeup for why: it competes
for evidence the other channels already own, and collapsing it into the
same first-crossing-wins resolution would suppress it behind an early
R1/R2/R3 hit instead of reporting both). So `"R4"` is intentionally
ABSENT from PRECEDENCE below, not an oversight -- R4 is scored and
reported as an independent, parallel diagnostic, never inserted into
this list's own mutual-exclusion ranking.
"""
from __future__ import annotations
import numpy as np
from ..types import Event

# Order here is PRECEDENCE (AGENT.md 3.6), not channel number. Channels were
# renumbered 2026-09-11 so the fundamental four read in check order --
# R1 existence, R2 interpenetration, R3 kinematic, R4 long-horizon -- with
# the pixel-scoped, unwired frame-invariance check as R5 (highest
# precedence if it were ever wired; it sits first for that reason).
# R6 conservation (AGENT.md M9, soft bodies, 2026-09-13) sits at the P2/P3
# level -- "about the entity remaining the same entity, not about its
# trajectory" -- so it ranks after existence and before interpenetration
# and kinematics. R7 shape (the soft body's shape descriptors vs. the
# reference band -- the material channel) sits LAST: when a wrong material
# and a wrong trajectory cross together, the trajectory is the primary
# failure, and a shape-only crossing is then unambiguously "material" --
# the attribution test (M9 decision 4) reads off this ordering. Numbers
# kept out of order deliberately (renumbering is the user's call); order
# here is the contract.
PRECEDENCE = ["R5", "R1", "R6", "R2", "R3", "R7"]


def _rank(risk):
    return PRECEDENCE.index(risk) if risk in PRECEDENCE else len(PRECEDENCE)


def _first_sustained_crossing(sigma, theta, dt, pi):
    exceed = sigma > theta
    need = max(1, int(round(pi / dt)))
    run = 0
    for i, e in enumerate(exceed):
        run = run + 1 if e else 0
        if run >= need:
            return i - need + 1
    return None


def extract_event(sigmas, thetas, dt, t_max, pi=0.3):
    crossings = {}
    for risk, sigma in sigmas.items():
        idx = _first_sustained_crossing(sigma, thetas[risk], dt, pi)
        if idx is not None:
            crossings[risk] = idx * dt

    if not crossings:
        return Event.censor(t_max)

    earliest = min(crossings.values())
    tied = [r for r, t in crossings.items() if t <= earliest + dt]
    winner = min(tied, key=_rank)
    return Event(time=crossings[winner], risk=winner, censored=False)
