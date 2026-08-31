"""First sustained threshold crossing, with precedence.

A crossing must hold for `pi` seconds (the persistence window) before it
counts -- a single noisy frame over threshold is not a failure. When more
than one risk crosses near the same instant, the precedence rule (AGENT.md
3.6, derived from the P1-P6 dependency order) decides which one is reported:
a violation at a lower level is undefined once a higher level has already
broken, so the higher-precedence risk wins, not the numerically-earliest one.

v1 only ever registers R2 and R5 (README scope), so in practice this only
ever resolves an R2-vs-R5 tie -- but the full order is implemented rather
than hardcoding the pair, so adding R1/R3/R4 later costs nothing here.
"""
from __future__ import annotations
import numpy as np
from ..types import Event

PRECEDENCE = ["R1", "R2", "R3", "R4", "R5"]


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
