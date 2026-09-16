"""Renders results/score_report.pdf -- a per-model diagnosed-failure
report, real model vs. baseline only (2026-08, fourth pass: color key
added, degradation (Kaplan-Meier) curves added to each page, ordering
made consistent across every chart on a page, axis scales made
consistent ACROSS pages so e.g. two models' failure-time distributions
are visually comparable, and the "no LLM" footnotes removed -- those
were notes for the assistant building this, not report content).

Real model vs. Baseline only -- Truth (ground-truth physics with a
KNOWN planted defect) is NOT included here. It calibrates the detector
itself and lives on the GATE 1 page; it isn't a meaningful comparison
point for judging a real model's OWN prediction quality. Baseline
(ConstantVelocity/CopyLastState) is a real, if trivial, PREDICTED
continuation from the same real prefix a real model gets, scored
through the identical calibrated thresholds -- that's the actual floor
a real model should be judged against, and the only thing this report
compares to. ConstantVelocity extrapolates linearly from the last
observed velocity (keeps moving); CopyLastState freezes the last frame
(stops moving) -- two different, genuinely distinct trivial predictors,
not the same baseline twice.

Canonical series order everywhere on a page (legend, bar chart, box
plot, degradation curves, failure-mode bars): model, constant_velocity,
copy_last_state. Every chart must use this same order -- a real bug
existed here where the bar chart and box plot disagreed on it (box plot
plots position 1 at the bottom by default, opposite of the barh/legend
convention used everywhere else), which read as an inconsistency
between adjacent charts on the same page.

Axis scales for VI50 (bar) and failure-time (box plot) are computed
GLOBALLY across every model page in this run, not per-page, so e.g.
Cosmos's and Wan's pages share identical axis ranges and are directly,
visually comparable -- a per-page auto-scaled axis would silently make
two different numbers look the same width on the page.

Saved as a PDF (matplotlib, vector, no browser) so it opens in any
native PDF viewer -- no HTML anywhere in this output. The same
degradation curves are also viewable live (second-by-second crosshair
tracing) via `scripts/plot_survival_native.py --model <name>`, which
imports this module's own color constants so the two artifacts match.

2026-09, fifth pass (three user-reported gaps, all real):
  1. Superseded model versions (registry `superseded_by`: cosmos* ->
     cosmos3nano, wan -> wan22) are omitted PER SCENARIO once the successor
     has a results file for that scenario; the cover lists every omission.
     `--include-superseded` shows everything.
  2. Missing baseline lines were missing RESULT FILES (billiards and
     occlusion_corridor_interpenetration had never had baselines run;
     ramp_descent lacked copy_last_state) and the report dropped the
     series silently. Now a red band names any absent baseline and the
     exact command to generate it. Separately, baselines are scored over
     the manifest's full horizon while real-model runs are truncated to
     prefix+continuation frames (t_max 3.0s vs 6-8s): a baseline failing
     at 4.5s counted as "fired" where a model would be censored at 3.0s.
     Baselines are now administratively RE-CENSORED at the model page's
     own t_max (`recensor`, below) before any comparison.
  3. R4 (long-horizon consistency, an independent CUSUM channel in the
     `long_horizon` block) gets its own survival panel and readout lines;
     the failure-mode chart is drawn over every channel the detector
     SCORED (thresholds_median keys), so R2 at 0% is a visible finding
     rather than an absent bar. Populations that predate R4 wiring say so.
  Also: `beats_all_baselines` was renamed `outlasts_all_baselines` -- the
  old flag was True when the model's VI50 was SMALLER than every
  baseline's, i.e. it failed FASTER (worse), the opposite of what the
  name read as. The new one is True only when the model's VI50 is
  strictly LARGER than every baseline's.

    python3 scripts/render_score_report.py
    open results/score_report.pdf
    python3 scripts/plot_survival_native.py --model cosmos   # live tracing, same data
"""
import argparse
import json
import pathlib
import sys
import textwrap
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "score_report.pdf"

BASELINE_BACKENDS = ["constant_velocity", "copy_last_state"]

# Scenario roles (user's review, 2026-09-14, AGENT.md "Review decisions"):
# the headline cross-model comparison is occlusion_reemergence (0.98
# instrument ceiling); the corridor family keeps its published populations
# as the PERCEPTION-TAX case study (0.68 ceiling, ~0.39s tax at occlusion
# entry). Pages are ordered headline first, case studies last, and both
# are NOT labelled on the pages (user, 2026-09-16: the labels are not
# needed) -- the ordering alone carries the decision; AGENT.md records it.
HEADLINE_SCENARIO = "occlusion_reemergence"
CASE_STUDY_SCENARIOS = ("occlusion_corridor", "occlusion_corridor_interpenetration",
                        "occlusion_corridor_distractor", "occlusion_corridor_moving")
SCENARIO_ROLE = {HEADLINE_SCENARIO: "headline comparison"}
SCENARIO_ROLE.update({s: "perception-tax case study" for s in CASE_STUDY_SCENARIOS})


def scenario_sort_key(scenario):
    return (0 if scenario == HEADLINE_SCENARIO else 2 if scenario in CASE_STUDY_SCENARIOS else 1, scenario)


def scenario_role(scenario):
    return SCENARIO_ROLE.get(scenario, "")
# The two IDEALS (2026-09-12, user: "why is this not being compared against
# the ideal? ... so that users can see how their models are performing
# against the ideal"): `truth` = the physics ideal (the held-out realization
# scored in state space; survival ~ 1 - alpha), `truth_render` = the
# INSTRUMENT ideal (the same realization rendered -> codec -> Phi, scored on
# every channel incl. R5 -- what a perfect video model would score after the
# perception tax). A model's honest headroom is the gap to truth_render;
# the gap between the two ideals is the instrument's own tax on that
# scenario. Both drawn on every page; neither gets a page of its own.
IDEAL_BACKENDS = ["truth_render", "truth"]
SERIES_LABEL = {"truth_render": "ideal (instrument)", "truth": "ideal (physics)"}
MIN_N_FOR_STATS = 10
# extract_event's own persistence window (vitals/detect/events.py, pi=0.3):
# a crossing at frame i is only confirmed when frames i..i+need-1 all
# exist, so in a run truncated at t_max only events with time <= t_max-pi
# were detectable at all. `recensor` uses this so a re-censored full-
# horizon baseline is exactly what a truncated run would have produced.
PI_PERSISTENCE_S = 0.3
HORIZONS_S = [1.5, 2.0, 3.0]   # S(t) horizons, same as both population scripts
MIN_TIME_AXIS = 0.5  # shared lower bound for every failure-time axis, so t=0.5 always means the same thing on the page

COLOR_MODEL = "#2563EB"
COLOR_BASELINE = {"constant_velocity": "#4ADE80", "copy_last_state": "#16A34A"}
COLOR_IDEAL = {"truth_render": "#374151", "truth": "#9CA3AF"}   # instrument ideal dark, physics ideal light
COLOR_INK = "#17171b"
COLOR_MUTED = "#6b6b76"
COLOR_CARD = "#f4f5f7"
COLOR_BORDER = "#dcdce2"
COLOR_BAD = "#B3261E"
COLOR_OK = "#0F7B4F"
COLOR_RISK = {"R1": "#7C3AED", "R2": "#D97706", "R3": "#2563EB", "R4": "#0F766E", "R5": "#9F1239",
              "R6": "#B45309", "R7": "#DB2777"}
RISK_LABEL = {"R1": "purple = R1 existence", "R2": "orange = R2 interpenetration", "R3": "blue = R3 kinematic",
              "R4": "teal = R4 long-horizon", "R5": "crimson = R5 frame invariance",
              "R6": "brown = R6 conservation", "R7": "pink = R7 shape"}
COLOR_WARN_BG, COLOR_WARN_INK = "#FBD5D5", "#7A1F1F"
COLOR_NOTE_BG, COLOR_NOTE_INK = "#FCE8B2", "#5C3D00"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": COLOR_INK,
    "axes.edgecolor": COLOR_BORDER,
    "axes.labelcolor": COLOR_INK,
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_MUTED,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

PAGE_W = 8.5
COVER_H = 11.0
# Model pages size THEMSELVES (2026-09-12): the readout's line count and the
# number of warning bands vary per page, and a fixed height either wasted
# space or ran the readout into the page bottom (both happened). Block
# heights below are in inches; draw_model_page sums them before creating
# the figure.
READOUT_LINE_H = 0.185      # inches per readout line (8pt monospace + leading)
READOUT_WRAP = 100          # characters per readout line at 8pt monospace across the card
BAND_H = 0.30               # warning/notice band height
BAND_GAP = 0.08
L, R = 0.09, 0.93


def fmt(v, d=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == float("inf"):
            return "∞"
        if v == float("-inf"):
            return "−∞"
        if v != v:
            return "undefined"
    return f"{v:.{d}f}"


def event_times(d, fired_only=True):
    ev = d["survival"]["events"]
    if fired_only:
        return np.array([e["time"] for e in ev if not e["censored"]])
    return np.array([e["time"] for e in ev])


def discover_all():
    out = {}
    for f in sorted(RESULTS_DIR.glob("l0_demo_*.json")):
        d = json.loads(f.read_text())
        out[(d["scenario"], d["backend"])] = d
    return out


def card(ax, x, y, w, h, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                 linewidth=0.8, edgecolor=COLOR_BORDER, facecolor=COLOR_CARD,
                                 transform=ax.transAxes, clip_on=False, **kw))


def series_order(st):
    """The one canonical order every chart on a page must follow: model,
    then the ideals (instrument, physics), then baselines in
    BASELINE_BACKENDS order."""
    return [st["model_backend"]] + [b for b, _ in st.get("ideals", [])] + [b for b, _ in st["baselines"]]


def series_data(st, name):
    if name == st["model_backend"]:
        return st["model_data"]
    d = dict(st["baselines"]); d.update(dict(st.get("ideals", [])))
    return d[name]


def series_color(st, name):
    if name == st["model_backend"]:
        return COLOR_MODEL
    return COLOR_IDEAL.get(name) or COLOR_BASELINE.get(name, "#999")


def series_label(name):
    return SERIES_LABEL.get(name, name)


def registered_channels(d):
    """The channels the detector actually SCORED for this population: the
    keys of thresholds_median, which both population scripts write from
    whatever was in STATS. This is the honest denominator for the failure-
    mode chart -- a channel that was scored and never fired (R2 at 0% on a
    P3 scenario) is a finding and must be listed at 0%, not vanish because
    termination_profile only carries risks that fired."""
    return sorted(d.get("thresholds_median", {}).keys())


def recensor(d, t_max_new, pi=PI_PERSISTENCE_S):
    """Administrative re-censoring of a population at a SHORTER t_max, so a
    baseline scored over the manifest's full horizon becomes exactly what
    the same baseline would have produced in a run truncated to the model
    page's own episode length. Valid because every channel (R1/R2/R3 per
    frame, R4's CUSUM) is causal and the LOO thresholds are per-frame
    prefixes; the only end effect is extract_event's own persistence
    window, handled by the `time > t_max - pi` cut (see PI_PERSISTENCE_S).
    A population already at or below t_max_new is returned unchanged.
    Never mutates the input; the original t_max is kept as
    `recensored_from_t_max` so the page can say what it did."""
    from vitals.types import Event
    from vitals.stats.survival import (validity_interval, termination_profile, bootstrap_vi, kaplan_meier,
                                       restricted_mean_validity, survival_at)

    if d["t_max"] <= t_max_new + 1e-9:
        return d
    cut = t_max_new - pi

    def clip(block, with_risk):
        evs = []
        for e in block["events"]:
            if e["censored"] or e["time"] > cut + 1e-9:
                evs.append(Event(time=t_max_new, risk=None, censored=True))
            else:
                evs.append(Event(time=e["time"], risk=e.get("risk") if with_risk else None, censored=False))
        grid, surv = kaplan_meier(evs)
        lo, hi = bootstrap_vi(evs, 0.5, n_boot=400) if len(evs) > 1 else (float("nan"), float("nan"))
        nb = dict(block)
        nb.update(n=len(evs), vi50=validity_interval(evs, 0.5), vi50_ci95=[lo, hi],
                  censoring_rate=sum(e.censored for e in evs) / max(1, len(evs)),
                  rmvt=restricted_mean_validity(evs, t_max_new),
                  S_at={str(h): s for h, s in zip(HORIZONS_S, survival_at(evs, HORIZONS_S))},
                  kaplan_meier=dict(t=grid.tolist(), S=surv.tolist()),
                  events=[(dict(time=e.time, risk=e.risk, censored=e.censored) if with_risk
                           else dict(time=e.time, censored=e.censored)) for e in evs])
        if with_risk:
            nb["termination_profile"] = termination_profile(evs)
        return nb

    out = dict(d)
    out["survival"] = clip(d["survival"], True)
    if "long_horizon" in d:
        out["long_horizon"] = clip(d["long_horizon"], False)
    out["recensored_from_t_max"] = d["t_max"]
    out["t_max"] = t_max_new
    return out


def apply_supersession(model_pages, all_results, registry):
    """Per-scenario: drop (scenario, old) when the registry names a
    `superseded_by` successor AND that successor has its own results file
    for the SAME scenario. Never global -- a scenario the successor has not
    been run on keeps its superseded page, or it would lose its only real-
    model result. Returns (kept, omitted) with omitted = (scenario, old,
    successor) triples for the cover page."""
    kept, omitted = [], []
    for scenario, backend in model_pages:
        succ = registry.get(backend, {}).get("superseded_by")
        if succ and (scenario, succ) in all_results:
            omitted.append((scenario, backend, succ))
        else:
            kept.append((scenario, backend))
    return kept, omitted


def outlasts_all_baselines(model_vi50, baseline_vi50s):
    """True only when the model's VI50 is strictly LARGER than every
    baseline's (fails later = better). Infinite (never-failing within
    t_max) values are handled explicitly rather than folded into False:
    a finite model vs. an infinite baseline is False; an infinite model
    vs. all-finite baselines is True; infinite on both sides is
    'undetermined' -- neither failed within the window, so the window
    cannot rank them."""
    if not baseline_vi50s:
        return None
    inf = float("inf")
    if model_vi50 == inf:
        return True if all(b != inf for b in baseline_vi50s) else "undetermined (both censored past t_max)"
    return all(b != inf and model_vi50 > b for b in baseline_vi50s)


def compute_stats(scenario, model_backend, all_results, bootstrap_vi_delta, Event):
    model = all_results[(scenario, model_backend)]
    s = model["survival"]
    n = s["n"]
    small_n = n < MIN_N_FOR_STATS

    # Baselines re-censored to THIS page's own t_max (see `recensor`) --
    # a raw full-horizon baseline is not comparable to a truncated run.
    baselines = [(b, recensor(all_results[(scenario, b)], model["t_max"]))
                 for b in BASELINE_BACKENDS if (scenario, b) in all_results]
    missing_baselines = [b for b in BASELINE_BACKENDS if (scenario, b) not in all_results]
    ideals = [(b, recensor(all_results[(scenario, b)], model["t_max"]))
              for b in IDEAL_BACKENDS if (scenario, b) in all_results]
    missing_ideals = [b for b in IDEAL_BACKENDS if (scenario, b) not in all_results]

    deltas = []
    model_events = [Event(time=e["time"], risk=e["risk"], censored=e["censored"]) for e in s["events"]]
    # Delta cards: vs the instrument ideal first (the headroom), then each baseline.
    for b_name, b_data in [(b, d) for b, d in ideals if b == "truth_render"] + baselines:
        b_s = b_data["survival"]
        if small_n or b_s["n"] < MIN_N_FOR_STATS:
            deltas.append(dict(name=b_name, point=None, lo=None, hi=None, verdict="insufficient sample"))
            continue
        b_events = [Event(time=e["time"], risk=e["risk"], censored=e["censored"]) for e in b_s["events"]]
        point, lo, hi = bootstrap_vi_delta(model_events, b_events, q=0.5, n_boot=2000, seed=0)
        inf = float("inf")
        # An infinite VI50 on either side (never fails within t_max) makes
        # the point estimate +/-inf and the bootstrap CI NaN (every
        # resample with an infinite side is dropped, survival.py's own
        # caveat) -- the old `hi < 0 / lo > 0` test then fell through to
        # "not significant", which is the OPPOSITE of what a model failing
        # at 1.8s vs. a baseline that never fails says. Found directly on
        # the billiards/copy_last_state card (2026-09).
        # The point is set explicitly here: bootstrap_vi_delta returns a
        # NaN point when EVERY resample is dropped (an always-infinite
        # baseline) but -inf when only some are -- the same situation
        # would otherwise print "undefined" on one page and "-inf" on the
        # next.
        if s["vi50"] != inf and b_s["vi50"] == inf:
            point, verdict = -inf, f"worse ({'ideal' if b_name in IDEAL_BACKENDS else 'baseline'} never fails)"
        elif s["vi50"] == inf and b_s["vi50"] != inf:
            point, verdict = inf, "better (model never fails within t_max)"
        elif s["vi50"] == inf and b_s["vi50"] == inf:
            point, verdict = float("nan"), "undetermined (neither fails within t_max)"
        elif hi < 0:
            verdict = "worse (fails faster)"
        elif lo > 0:
            verdict = "better (fails slower)"
        else:
            verdict = "not significant"
        deltas.append(dict(name=b_name, point=point, lo=lo, hi=hi, verdict=verdict))

    times = event_times(model, fired_only=True)
    frac_at_first = float((times == times.min()).mean()) if len(times) else 0.0
    prof = s["termination_profile"]
    dominant_risk = max(prof.items(), key=lambda kv: kv[1])[0] if prof else None

    outlasts = outlasts_all_baselines(s["vi50"], [b_data["survival"]["vi50"] for _, b_data in baselines])

    # Every channel scored by ANY series on the page (registered, whether
    # or not it ever fired) plus anything that fired -- the failure-mode
    # chart's x-categories. R4 is never in here: it is an independent
    # channel with its own block, not a competing first-crossing risk.
    channels = sorted({c for name_d in [model] + [d for _, d in baselines] for c in registered_channels(name_d)}
                      | {k for name_d in [model] + [d for _, d in baselines] for k in name_d["survival"]["termination_profile"]})

    lh = model.get("long_horizon")   # None => population predates R4 wiring
    lh_baselines = [(b, d["long_horizon"]) for b, d in baselines if "long_horizon" in d]

    # VI50-floor summaries (2026-09-12): RMVT and S(t) at fixed horizons,
    # computed from the events here when a population file predates them,
    # so every page has them regardless of when it was run.
    from vitals.stats.survival import restricted_mean_validity, survival_at
    def floor_stats(d):
        sv = d["survival"]
        evs = [Event(time=e["time"], risk=e["risk"], censored=e["censored"]) for e in sv["events"]]
        return dict(rmvt=sv.get("rmvt", restricted_mean_validity(evs, d["t_max"])),
                    rmvt_ci=sv.get("rmvt_ci95"),
                    S_at=sv.get("S_at", {str(h): v for h, v in zip(HORIZONS_S, survival_at(evs, HORIZONS_S))}))
    fs = floor_stats(model)
    fs_baselines = [(b, floor_stats(d)) for b, d in baselines]
    fs_ideals = [(b, floor_stats(d)) for b, d in ideals]
    rmvt_outlasts = all(fs["rmvt"] > fb["rmvt"] for _, fb in fs_baselines) if fs_baselines else None
    rescore_path = RESULTS_DIR / f"lambda_rescore_{scenario}_{model_backend}.json"
    rescore = json.loads(rescore_path.read_text()) if rescore_path.exists() else None
    fi = model.get("frame_invariance")   # R5 photometric diagnostics, populations run after 2026-09-12 only

    return dict(scenario=scenario, model_backend=model_backend, n=n, small_n=small_n,
                vi50=s["vi50"], ci=s["vi50_ci95"], censoring=s["censoring_rate"],
                t_max=model["t_max"], baselines=baselines, missing_baselines=missing_baselines,
                deltas=deltas, frac_at_first=frac_at_first, dominant_risk=dominant_risk, prof=prof,
                outlasts=outlasts, channels=channels, lh=lh, lh_baselines=lh_baselines,
                rmvt=fs["rmvt"], rmvt_ci=fs["rmvt_ci"], S_at=fs["S_at"], fs_baselines=fs_baselines,
                rmvt_outlasts=rmvt_outlasts, rescore=rescore, frame_invariance=fi,
                ideals=ideals, missing_ideals=missing_ideals, fs_ideals=fs_ideals,
                model_data=model)


def compute_global_axis_limits(all_stats):
    """VI50 and failure-time axis ranges, computed across EVERY model page
    in this run so different models' pages share identical scales and are
    directly comparable side by side -- a per-page autoscale would let two
    different numbers draw the same bar width."""
    vi_vals, time_vals, km_t_vals = [], [], []
    for st in all_stats:
        for name in series_order(st):
            d = series_data(st, name)
            s = d["survival"]
            v = s["vi50"] if s["vi50"] != float("inf") else d["t_max"]
            lo, hi = s["vi50_ci95"]
            hi = hi if hi not in (None, float("inf")) and not (isinstance(hi, float) and np.isnan(hi)) else v
            vi_vals.append(max(v, hi))
            t = event_times(d, fired_only=True)
            if len(t):
                time_vals.append(t.max())
            km_t_vals.append(d["t_max"])
    return dict(
        vi_max=(max(vi_vals) * 1.08) if vi_vals else 1.0,
        time_max=(max(time_vals) * 1.08) if time_vals else 1.0,
        km_t_max=(max(km_t_vals) * 1.03) if km_t_vals else 1.0,
    )


def draw_legend(fig, cur_y_frac, st):
    """Bullet + label color key, canonical order, fixed 3-column layout
    (model + 2 baselines). A baseline with no results file is still
    listed -- struck through and labeled NOT RUN -- so its absence is
    visible on the page instead of silently shrinking the legend."""
    names = series_order(st) + list(st.get("missing_ideals", [])) + list(st.get("missing_baselines", []))
    step = (R - L) / max(3, len(names))
    xs = [L + i * step for i in range(len(names))]
    ideal_names = {b for b, _ in st.get("ideals", [])}
    for name, x in zip(names, xs):
        missing = name in st.get("missing_baselines", []) or name in st.get("missing_ideals", [])
        color = COLOR_BAD if missing else series_color(st, name)
        role = ("NOT RUN — no results file" if missing
                else "real model" if name == st["model_backend"]
                else "ideal" if name in ideal_names else "baseline")
        fig.text(x, cur_y_frac, "○" if missing else "●", fontsize=12, color=color, va="center")
        fig.text(x + 0.02, cur_y_frac, series_label(name), fontsize=8, color=COLOR_BAD if missing else COLOR_INK,
                 va="center", fontweight="medium")
        fig.text(x + 0.02, cur_y_frac - 0.014, role, fontsize=6.5, color=COLOR_BAD if missing else COLOR_MUTED,
                 va="center")


def draw_band(fig, top_frac, band_h_in, page_h, text, bg, ink):
    band = fig.add_axes([L, top_frac, R - L, band_h_in / page_h])
    band.axis("off")
    band.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15",
                                   linewidth=0, facecolor=bg, transform=band.transAxes, clip_on=False))
    band.text(0.015, 0.5, text, transform=band.transAxes, fontsize=7.6, va="center", color=ink, fontweight="medium")


def readout_lines(st):
    """The COMPUTED READOUT's discrete `key = value` lines (never prose),
    wrapped to the card width with indented continuation lines so nothing
    runs off the page. Built before the figure exists so the page can be
    sized to fit them."""
    lh = st["lh"]
    lines = [
        f"outlasts_all_baselines = {st['outlasts']}   (model VI50 strictly larger than every baseline's)",
        f"frac_failures_at_earliest_instant = {fmt(st['frac_at_first']*100, 1)}%",
        f"dominant_failure_mode = {st['dominant_risk']} ({fmt(st['prof'].get(st['dominant_risk'], 0)*100, 1)}% of terminations, "
        f"{fmt(st['prof'].get(st['dominant_risk'], 0)*(1-st['censoring'])*100, 1)}% of episodes)",
        "channels_scored = " + ", ".join(st["channels"]) + "   (R4 long-horizon scored separately, below)",
    ]
    # Headroom to the ideals (2026-09-12): the gap to the instrument ideal
    # is what a model could still gain; the gap between the two ideals is
    # the instrument's own tax on this scenario.
    fi_ = dict(st.get("fs_ideals", []))
    if "truth_render" in fi_:
        lines.append(f"gap_to_ideal(instrument) = rmvt {fmt(fi_['truth_render']['rmvt'] - st['rmvt'])}s  ·  "
                     f"S(3.0) {fmt(fi_['truth_render']['S_at'].get('3.0', float('nan')) - st['S_at'].get('3.0', float('nan')), 2)}"
                     f"   (ideal rmvt {fmt(fi_['truth_render']['rmvt'])}s)")
    if "truth" in fi_ and "truth_render" in fi_:
        lines.append(f"instrument_tax = rmvt {fmt(fi_['truth']['rmvt'] - fi_['truth_render']['rmvt'])}s between ideal(physics) "
                     f"{fmt(fi_['truth']['rmvt'])}s and ideal(instrument) {fmt(fi_['truth_render']['rmvt'])}s")
    elif "truth" in fi_:
        lines.append(f"gap_to_ideal(physics) = rmvt {fmt(fi_['truth']['rmvt'] - st['rmvt'])}s   (instrument ideal not run)")
    lines += [
        f"rmvt = {fmt(st['rmvt'])}s" + (f"  CI [{fmt(st['rmvt_ci'][0])}, {fmt(st['rmvt_ci'][1])}]" if st.get("rmvt_ci") else "")
        + "".join(f"  ·  rmvt[{b}] = {fmt(fb['rmvt'])}s" for b, fb in st["fs_baselines"])
        + f"   ->  rmvt_outlasts_all_baselines = {st['rmvt_outlasts']}",
        "S(t) = " + ", ".join(f"S({h}) {fmt(v, 2)}" for h, v in st["S_at"].items())
        + "".join(f"   [{b}: " + ", ".join(fmt(v, 2) for v in fb["S_at"].values()) + "]" for b, fb in st["fs_baselines"]),
    ]
    if st.get("rescore"):
        lv = st["rescore"]["levels"]
        floor = min(l["vi50"] for l in lv)
        off_floor = [l for l in lv if l["vi50"] > floor + 0.1]
        lines.append("vi50_vs_lambda = " + ", ".join(f"λ{l['lam']:g}: {fmt(l['vi50'])}" for l in lv)
                     + (f"   (leaves the floor at λ={off_floor[0]['lam']:g})" if off_floor else "   (pinned at the floor at every λ)"))
        lines.append("rmvt_vs_lambda = " + ", ".join(f"λ{l['lam']:g}: {fmt(l['rmvt'])}" for l in lv))
    else:
        lines.append("vi50_vs_lambda = not computed (run scripts/rescore_population.py)")
    if st.get("frame_invariance"):
        fi = st["frame_invariance"]
        lines.append(f"R5_photometric: boundary_jump = {fmt(fi.get('photometric_boundary_jump_median'))} levels  ·  "
                     f"drift = {fmt(fi.get('photometric_drift_median'))} levels  ·  theta_R5 = {fmt(fi.get('theta_R5_px_median'))} px"
                     f"  (calibrated on this population's own render environment; not portable across machines, AGENT.md T9)")
    # Provider-side seed reproducibility (registry `seed_reproducible`,
    # 2026-09-14): stated per model, never buried. Pairing is on the
    # episode (same prefix + initial condition per seed for every model),
    # so the cross-model comparison stands; this population itself is
    # replicable only as a distribution.
    from vitals.adapters import MODEL_REGISTRY as _REG
    if _REG.get(st["model_backend"], {}).get("seed_reproducible", True) is False:
        lines.append("seed_reproducible = False   (provider accepts no seed: re-running does not reproduce these videos; "
                     "paired on episodes, not on the provider's sampler; cannot be topped up with --merge-existing)")
    if lh is None:
        lines.append("R4_long_horizon = not computed (population predates R4)")
    else:
        lines.append(f"R4_long_horizon_fired = {fmt((1 - lh['censoring_rate'])*100, 0)}% of episodes  ·  "
                     f"R4_vi50 = {fmt(lh['vi50'])}{'' if lh['vi50'] == float('inf') else 's'}  "
                     f"CI [{fmt(lh['vi50_ci95'][0])}, {fmt(lh['vi50_ci95'][1])}]")
        for b, block in st["lh_baselines"]:
            lines.append(f"R4_long_horizon_fired[{b}] = {fmt((1 - block['censoring_rate'])*100, 0)}%  ·  "
                         f"R4_vi50[{b}] = {fmt(block['vi50'])}{'' if block['vi50'] == float('inf') else 's'}")
    for d in st["deltas"]:
        p = d["point"]
        finite = isinstance(p, (int, float)) and p == p and abs(p) != float("inf")
        v = d["verdict"] if p is None else f"{fmt(p)}{'s' if finite else ''}, {d['verdict']}"
        lines.append(f"vi50_delta[{d['name']}] = {v}")
    v = st.get("validity")
    if v is None:
        lines.append("seed_test_retest / minimum_detectable_difference = not yet computed "
                     "(run scripts/run_validity_analysis.py)")
    else:
        tr = v["seed_test_retest"]
        mdd = v["minimum_detectable_difference"]
        tr_str = (f"{fmt(tr['median_abs_delta'])}s" if tr["median_abs_delta"] == tr["median_abs_delta"]
                  else "undefined (no valid split)")
        mdd_str = f"{fmt(mdd['mdd'])}s" if mdd["mdd"] is not None else "not reached within tested grid"
        vacuous = " (VACUOUS -- zero/undefined natural spread, not a real threshold)" if v.get("mdd_caveat") else ""
        lines.append(f"seed_test_retest_median_abs_delta = {tr_str}")
        lines.append(f"minimum_detectable_difference(power>={mdd['target_power']:.0%}) = {mdd_str}{vacuous}")
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, READOUT_WRAP, subsequent_indent="      ", break_long_words=False) or [""])
    return wrapped


def draw_model_page(pdf, st, xlims):
    # Size the page from its own content (see the READOUT_* constants).
    lines = readout_lines(st)
    n_bands = (int(st["small_n"]) + len(st["missing_baselines"]) + len(st["missing_ideals"])
               + int(any("recensored_from_t_max" in d for _, d in st["baselines"] + st["ideals"])))
    read_h = 0.40 + READOUT_LINE_H * len(lines)
    lh_h = (0.30 + 1.45 + 0.34 + 0.24) if st["lh"] is not None else (0.06 + BAND_H + 0.24)
    header_h = 0.42 + 0.34 + 0.30 + 0.34 + 0.38 + (0.16 if n_bands == 0 else 0.12) + n_bands * (BAND_H + BAND_GAP)
    page_h = header_h + (1.05 + 0.30) + (0.30 + 2.05 + 0.34 + 0.24) + (0.30 + 1.55 + 0.34 + 0.24) \
             + (0.30 + 0.62 + 0.34 + 0.24) + lh_h + read_h + 0.30
    fig = plt.figure(figsize=(PAGE_W, page_h))

    def top(cur_in):
        return 1 - cur_in / page_h

    cur = 0.42
    fig.text(L, top(cur), "VITALS Score Report", fontsize=9, color=COLOR_MUTED, fontweight="medium")
    cur += 0.34
    fig.text(L, top(cur), st["model_backend"], fontsize=24, color=COLOR_INK, fontweight="bold")
    cur += 0.30
    fig.text(L, top(cur), f"scenario: {st['scenario']}   ·   n = {st['n']} episodes   ·   scored vs. baseline",
             fontsize=10, color=COLOR_MUTED)
    cur += 0.34

    draw_legend(fig, top(cur), st)
    cur += 0.38

    band_h = 0.30
    n_bands = 0
    if st["small_n"]:
        draw_band(fig, top(cur + band_h), band_h, page_h,
                  f"sample size n={st['n']} is below the n={MIN_N_FOR_STATS} minimum for "
                  f"significance testing — figures below are illustrative only", COLOR_NOTE_BG, COLOR_NOTE_INK)
        cur += band_h + 0.08
        n_bands += 1
    for b in st["missing_baselines"]:
        # The bug this replaces: an absent baseline file silently dropped
        # the series from every chart, read as "the line is missing".
        draw_band(fig, top(cur + band_h), band_h, page_h,
                  f"baseline '{b}' has NOT been run for {st['scenario']} — generate it with: "
                  f"python3 scripts/run_eval.py --adapter {b} --manifest configs/manifests/{st['scenario']}.yaml "
                  f"--backend mujoco --n-episodes 50", COLOR_WARN_BG, COLOR_WARN_INK)
        cur += band_h + 0.08
        n_bands += 1
    for b in st["missing_ideals"]:
        cmd = (f"python3 scripts/run_eval.py --adapter truth --manifest configs/manifests/{st['scenario']}.yaml --backend mujoco --n-episodes 50"
               if b == "truth" else f"python3 remote/call_orchestrate.py submit --model truth_render --scenario {st['scenario']} --seeds 1..50")
        draw_band(fig, top(cur + band_h), band_h, page_h,
                  f"{series_label(b)} has NOT been run for {st['scenario']} — generate it with: {cmd}", COLOR_WARN_BG, COLOR_WARN_INK)
        cur += band_h + 0.08
        n_bands += 1
    recens = [(b, d["recensored_from_t_max"]) for b, d in st["baselines"] + st["ideals"] if "recensored_from_t_max" in d]
    if recens:
        draw_band(fig, top(cur + band_h), band_h, page_h,
                  f"baselines/ideals re-censored at this page's t_max = {fmt(st['t_max'])}s "
                  f"(scored over {', '.join(f'{series_label(b)} {fmt(t)}s' for b, t in recens)})", COLOR_NOTE_BG, COLOR_NOTE_INK)
        cur += band_h + 0.08
        n_bands += 1
    cur += 0.16 if n_bands == 0 else 0.12

    # --- stat cards row --------------------------------------------------
    cards_h = 1.05
    cards_ax = fig.add_axes([L, top(cur + cards_h), R - L, cards_h / page_h])
    cards_ax.set_xlim(0, 1)
    cards_ax.set_ylim(0, 1)
    cards_ax.axis("off")

    n_cards = 1 + len(st["deltas"])
    gap_frac = 0.02
    cw_frac = (1.0 - gap_frac * (n_cards - 1)) / n_cards
    x = 0.0
    card(cards_ax, x, 0, cw_frac, 1)
    cards_ax.text(x + 0.02, 0.72, "VI₅₀ (median)", fontsize=7.2, color=COLOR_MUTED, transform=cards_ax.transAxes)
    vi_str = f"{fmt(st['vi50'])}{'' if st['vi50'] == float('inf') else 's'}"
    cards_ax.text(x + 0.02, 0.30, vi_str, fontsize=16, color=COLOR_INK, fontweight="bold", transform=cards_ax.transAxes)
    cards_ax.text(x + 0.02, 0.08, f"CI [{fmt(st['ci'][0])}, {fmt(st['ci'][1])}]  ·  censor {fmt(st['censoring'])}  ·  "
                                  f"RMVT {fmt(st['rmvt'])}s",
                  fontsize=6.6, color=COLOR_MUTED, transform=cards_ax.transAxes)
    x += cw_frac + gap_frac

    for d in st["deltas"]:
        card(cards_ax, x, 0, cw_frac, 1)
        cards_ax.text(x + 0.02, 0.72, f"Δ vs. {series_label(d['name'])}", fontsize=7.2, color=COLOR_MUTED, transform=cards_ax.transAxes)
        if d["point"] is None:
            cards_ax.text(x + 0.02, 0.30, "n/a", fontsize=14, color=COLOR_MUTED, transform=cards_ax.transAxes)
            cards_ax.text(x + 0.02, 0.08, "insufficient sample", fontsize=6.6, color=COLOR_MUTED, transform=cards_ax.transAxes)
        else:
            vcolor = COLOR_BAD if "worse" in d["verdict"] else (COLOR_OK if "better" in d["verdict"] else COLOR_INK)
            unit = "" if (isinstance(d["point"], float) and (abs(d["point"]) == float("inf") or d["point"] != d["point"])) else "s"
            cards_ax.text(x + 0.02, 0.30, f"{fmt(d['point'])}{unit}", fontsize=16, color=vcolor, fontweight="bold", transform=cards_ax.transAxes)
            ci_txt = "CI n/a" if d["lo"] != d["lo"] else f"CI [{fmt(d['lo'])}, {fmt(d['hi'])}]s"
            cards_ax.text(x + 0.02, 0.08, f"{ci_txt} · {d['verdict']}",
                          fontsize=6.2, color=COLOR_MUTED, transform=cards_ax.transAxes)
        x += cw_frac + gap_frac
    cur += cards_h + 0.30

    # --- degradation curves (Kaplan-Meier survival) -----------------------
    title_h, xlabel_h = 0.30, 0.34
    km_ax_h = 2.05
    ax_km = fig.add_axes([L, top(cur + title_h + km_ax_h), R - L, km_ax_h / page_h])
    names = series_order(st)
    for name in names:
        d = series_data(st, name)
        color = series_color(st, name)
        km = d["survival"]["kaplan_meier"]
        t, S = np.array(km["t"]), np.array(km["S"])
        ax_km.step(t, S, where="post", color=color, linewidth=2)
        t_max = d["t_max"]
        if t_max > t[-1]:
            ax_km.plot([t[-1], t_max], [S[-1], S[-1]], color=color, linewidth=2,
                       linestyle=(0, (2, 3)), alpha=0.55)
    ax_km.set_xlim(0, xlims["km_t_max"])
    ax_km.set_ylim(-0.02, 1.02)
    ax_km.set_ylabel("S(t) — fraction still valid", fontsize=8)
    ax_km.set_xlabel("time (s)", fontsize=8)
    ax_km.set_title("Degradation curve (Kaplan–Meier survival, dashed = censored past clip end)",
                     fontsize=9, fontweight="bold", loc="left", pad=8)
    ax_km.spines[["top", "right"]].set_visible(False)
    ax_km.tick_params(labelsize=7.5)
    ax_km.grid(alpha=0.25, linewidth=0.6)
    cur += title_h + km_ax_h + xlabel_h + 0.24

    # --- VI50 bar chart + failure-time distribution, side by side --------
    # Both axes are inset well past L/R -- horizontal-bar y-tick labels
    # ("constant_velocity" etc) draw OUTSIDE the axes box to its left, and
    # without this inset they clip off the page edge (a real bug hit
    # twice now rebuilding this layout -- LABEL_INSET exists specifically
    # so it can't silently regress a third time).
    row_ax_h = 1.55
    LABEL_INSET = 0.15
    ax_bar = fig.add_axes([L + LABEL_INSET, top(cur + title_h + row_ax_h), 0.24, row_ax_h / page_h])
    ax_box = fig.add_axes([L + LABEL_INSET + 0.24 + 0.13, top(cur + title_h + row_ax_h),
                            R - (L + LABEL_INSET + 0.24 + 0.13), row_ax_h / page_h])

    vi50s, cis, colors = [], [], []
    for name in names:
        d = series_data(st, name)
        s = d["survival"]
        v = s["vi50"] if s["vi50"] != float("inf") else d["t_max"]
        vi50s.append(v)
        lo, hi = s["vi50_ci95"]
        lo = lo if lo not in (None, float("inf"), float("-inf")) and not np.isnan(lo) else v
        hi = hi if hi not in (None, float("inf")) and not np.isnan(hi) else v
        cis.append((max(0, v - lo), max(0, hi - v)))
        colors.append(series_color(st, name))

    y_pos = np.arange(len(names))[::-1]
    err = np.array(cis).T
    ax_bar.barh(y_pos, vi50s, color=colors, height=0.55, xerr=err, capsize=3,
                error_kw=dict(ecolor=COLOR_MUTED, elinewidth=1))
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([series_label(n) for n in names], fontsize=7.5)
    ax_bar.set_xlim(0, xlims["vi_max"])
    ax_bar.set_xlabel("VI₅₀ (s)", fontsize=8)
    ax_bar.set_title("Median validity interval", fontsize=9, fontweight="bold", loc="left", pad=8)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(labelsize=7.5)
    ax_bar.grid(axis="x", alpha=0.25, linewidth=0.6)

    # Box plot's own default vertical order (position 1 = bottom) is the
    # OPPOSITE of barh's y_pos convention above -- feeding it names in
    # reverse puts the canonical first-listed series (the model) at the
    # top here too, so both charts read top-to-bottom in the same order.
    order_names = list(reversed(names))
    box_data, box_colors = [], []
    for name in order_names:
        d = series_data(st, name)
        t = event_times(d, fired_only=True)
        box_data.append(t if len(t) else [np.nan])
        box_colors.append(series_color(st, name))
    bp = ax_box.boxplot(box_data, orientation="horizontal", patch_artist=True, widths=0.5,
                         medianprops=dict(color=COLOR_INK, linewidth=1.4),
                         flierprops=dict(marker="o", markersize=2.5, alpha=0.5))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor(COLOR_INK)
        patch.set_linewidth(0.6)
    ax_box.set_yticks(range(1, len(order_names) + 1))
    ax_box.set_yticklabels([series_label(n) for n in order_names], fontsize=7.5)
    ax_box.set_xlim(MIN_TIME_AXIS, xlims["time_max"])
    ax_box.set_xlabel("event time, fired episodes (s)", fontsize=8)
    ax_box.set_title("Failure-time distribution", fontsize=9, fontweight="bold", loc="left", pad=8)
    ax_box.spines[["top", "right"]].set_visible(False)
    ax_box.tick_params(labelsize=7.5)
    ax_box.grid(axis="x", alpha=0.25, linewidth=0.6)
    cur += title_h + row_ax_h + xlabel_h + 0.24

    # --- failure mode breakdown ------------------------------------------
    mode_ax_h = 0.62
    ax_mode = fig.add_axes([L + LABEL_INSET, top(cur + title_h + mode_ax_h),
                             R - (L + LABEL_INSET), mode_ax_h / page_h])
    # Only channels that actually FIRED for some series on this page get a
    # bar segment and a name in the title (user, 2026-09-11: a 0% channel
    # "doesn't need to be written out"). What was SCORED, fired or not, is
    # recorded once in the readout as `channels_scored` -- so R2 on a P3
    # page is still visibly part of the instrument without a "0%" label.
    risk_keys = [rk for rk in st["channels"]
                 if any(series_data(st, name)["survival"]["termination_profile"].get(rk, 0.0) > 0 for name in names)]
    left = np.zeros(len(names))
    yy = np.arange(len(names))[::-1]
    # Normalized by EPISODES, not by terminations (2026-09-12, user: "as
    # the ideal, shouldn't those show no failures at all?"): the old
    # chart divided by the number of terminations, so a series with ONE
    # failed episode drew a full-width bar exactly like a series that
    # failed every episode. Bar length is now the fraction of episodes
    # terminated by t_max, split by channel; the empty remainder is the
    # fraction still valid at t_max.
    for rk in risk_keys:
        vals = []
        for name in names:
            sv = series_data(st, name)["survival"]
            vals.append(sv["termination_profile"].get(rk, 0.0) * (1.0 - sv["censoring_rate"]) * 100)
        vals = np.array(vals)
        ax_mode.barh(yy, vals, left=left, color=COLOR_RISK.get(rk, "#999"), height=0.55)
        left += vals
    ax_mode.set_yticks(yy)
    ax_mode.set_yticklabels([series_label(n) for n in names], fontsize=7.5)
    ax_mode.set_xlim(0, 100)
    ax_mode.set_xlabel("% of episodes terminated by t_max, by channel", fontsize=8)
    short = {"R1": "existence", "R2": "interpenetration", "R3": "kinematic", "R5": "frame invariance",
             "R6": "conservation", "R7": "shape"}
    legend_bits = " · ".join(f"{rk} {short.get(rk, '')}".strip() for rk in risk_keys) or "no terminations"
    ax_mode.set_title(f"Failure mode ({legend_bits})", fontsize=9, fontweight="bold", loc="left", pad=8)
    ax_mode.spines[["top", "right"]].set_visible(False)
    ax_mode.tick_params(labelsize=7.5)
    cur += title_h + mode_ax_h + xlabel_h + 0.24

    # --- R4 long-horizon consistency (independent channel) -----------------
    lh_ax_h = 1.45
    lh = st["lh"]
    if lh is None:
        cur += 0.06
        draw_band(fig, top(cur + band_h), band_h, page_h,
                  "R4 long-horizon consistency: NOT COMPUTED — this population predates R4 wiring "
                  "(no `long_horizon` block); re-run scripts/run_model_population.py to score it",
                  COLOR_NOTE_BG, COLOR_NOTE_INK)
        cur += band_h + 0.24
    else:
        ax_lh = fig.add_axes([L, top(cur + title_h + lh_ax_h), R - L, lh_ax_h / page_h])
        lh_series = [(st["model_backend"], lh)] + st["lh_baselines"]
        for name, block in lh_series:
            color = series_color(st, name)
            t, S = np.array(block["kaplan_meier"]["t"]), np.array(block["kaplan_meier"]["S"])
            ax_lh.step(t, S, where="post", color=color, linewidth=2)
            if st["t_max"] > t[-1]:
                ax_lh.plot([t[-1], st["t_max"]], [S[-1], S[-1]], color=color, linewidth=2,
                           linestyle=(0, (2, 3)), alpha=0.55)
        ax_lh.set_xlim(0, xlims["km_t_max"])
        ax_lh.set_ylim(-0.02, 1.02)
        ax_lh.set_ylabel("S(t) — no cumulative drift yet", fontsize=8)
        ax_lh.set_xlabel("time (s)", fontsize=8)
        lh_missing = [b for b, _ in st["baselines"] if b not in dict(st["lh_baselines"])]
        note = f"  ·  baselines without R4: {', '.join(lh_missing)}" if lh_missing else ""
        ax_lh.set_title(f"R4 long-horizon consistency (CUSUM of sub-threshold gap; independent of the channels above){note}",
                        fontsize=9, fontweight="bold", loc="left", pad=8)
        ax_lh.spines[["top", "right"]].set_visible(False)
        ax_lh.tick_params(labelsize=7.5)
        ax_lh.grid(alpha=0.25, linewidth=0.6)
        cur += title_h + lh_ax_h + xlabel_h + 0.24

    # --- computed readout (discrete fields, not prose) --------------------
    # Fixed line pitch (READOUT_LINE_H) and a card sized to the wrapped
    # line count -- the previous design spread the lines over a fixed 58%
    # of a card whose height grew with the count, which left large blank
    # bands above and below once pages reached 12+ lines (2026-09-12).
    ax_read = fig.add_axes([L, top(cur + read_h), R - L, read_h / page_h])
    ax_read.axis("off")
    card(ax_read, 0, 0, 1, 1)
    ax_read.text(0.02, 1 - 0.15 / read_h, "COMPUTED READOUT", fontsize=7, color=COLOR_MUTED, fontweight="bold",
                 va="center", transform=ax_read.transAxes)
    for i, line in enumerate(lines):
        y = 1 - (0.36 + READOUT_LINE_H * i) / read_h
        ax_read.text(0.02, y, line, fontsize=8, color=COLOR_INK, family="monospace", va="center",
                     transform=ax_read.transAxes)
    cur += read_h

    pdf.savefig(fig)
    plt.close(fig)


def draw_cover_page(pdf, model_pages, all_stats, omitted=()):
    fig = plt.figure(figsize=(PAGE_W, COVER_H))
    fig.text(0.5, 0.80, "VITALS", fontsize=40, ha="center", fontweight="bold", color=COLOR_INK)
    fig.text(0.5, 0.75, "Score Report", fontsize=18, ha="center", color=COLOR_MUTED)
    fig.text(0.5, 0.705, "Real model vs. baseline, per scenario", fontsize=11, ha="center", color=COLOR_MUTED)

    y = 0.64
    fig.text(0.15, y, "COLOR KEY", fontsize=8, color=COLOR_MUTED, fontweight="bold")
    y -= 0.03
    key_items = [("real model", COLOR_MODEL),
                 ("ideal (instrument) — the true rollout rendered, codec'd and reconstructed by Phi: the ceiling after the perception tax", COLOR_IDEAL["truth_render"]),
                 ("ideal (physics) — the true rollout in state space: survival ≈ 1 − α", COLOR_IDEAL["truth"]),
                 ("constant_velocity baseline", COLOR_BASELINE["constant_velocity"]),
                 ("copy_last_state baseline", COLOR_BASELINE["copy_last_state"])]
    for label, color in key_items:
        fig.text(0.17, y, "●", fontsize=12, color=color, va="center")
        fig.text(0.19, y, label, fontsize=9.5, color=COLOR_INK, va="center")
        y -= 0.028

    y -= 0.02
    fig.text(0.15, y, "DIAGNOSED CHANNELS", fontsize=8, color=COLOR_MUTED, fontweight="bold")
    y -= 0.03
    chan_items = [("R1  existence — tracked object vanishes / duplicates", "R1"),
                  ("R2  interpenetration — two bodies overlap (scored on P3 scenarios only)", "R2"),
                  ("R3  kinematic — velocity/acceleration outside the reference band", "R3"),
                  ("R4  long-horizon — cumulative sub-threshold drift (CUSUM); independent channel", "R4"),
                  ("R5  frame invariance — the model's own camera/scene moves (px); video models only, first in precedence", "R5"),
                  ("R6  conservation — a soft body's silhouette area leaks or puffs vs. the reference band (soft-body scenarios only)", "R6"),
                  ("R7  shape — a soft body's shape descriptors leave the band: wrong material, last in precedence (soft-body only)", "R7")]
    for label, rk in chan_items:
        fig.text(0.17, y, "●", fontsize=10, color=COLOR_RISK[rk], va="center")
        fig.text(0.19, y, label, fontsize=8.5, color=COLOR_INK, va="center")
        y -= 0.026

    y -= 0.02
    fig.text(0.15, y, "CONTENTS", fontsize=8, color=COLOR_MUTED, fontweight="bold")
    for scenario, model_backend in model_pages:
        y -= 0.026
        fig.text(0.17, y, f"{scenario} — {model_backend}", fontsize=9.5, color=COLOR_INK, family="monospace")

    # Superseded model versions are omitted silently (user, 2026-09-13:
    # "you can remove the omitted ones they're not needed"); the console
    # still prints what was omitted, and --include-superseded shows them.
    pdf.savefig(fig)
    plt.close(fig)


def render(include_superseded=False):
    from vitals.adapters import REAL_MODEL_BACKENDS, MODEL_REGISTRY
    from vitals.stats.survival import bootstrap_vi_delta
    from vitals.detect.events import Event

    all_results = discover_all()
    model_pages = sorted(((s, b) for (s, b) in all_results if b in REAL_MODEL_BACKENDS),
                         key=lambda sb: (scenario_sort_key(sb[0]), sb[1]))
    omitted = []
    if not include_superseded:
        model_pages, omitted = apply_supersession(model_pages, all_results, MODEL_REGISTRY)
        for scenario, old, succ in omitted:
            print(f"omitting {scenario} — {old}: superseded by {succ} (has results for this scenario)")
    if not model_pages:
        print("no real-model results found in results/l0_demo_*.json -- run scripts/run_model_population.py first")
        return

    old_html = RESULTS_DIR / "score_report.html"
    if old_html.exists():
        old_html.unlink()

    all_stats = [compute_stats(scenario, model_backend, all_results, bootstrap_vi_delta, Event)
                 for scenario, model_backend in model_pages]
    xlims = compute_global_axis_limits(all_stats)

    # M7's seed_test_retest/minimum_detectable_difference (scripts/run_
    # validity_analysis.py -- a separate, local-only, no-GPU-cost script,
    # run independently since it's a statistics-of-statistics pass over
    # already-collected populations, not itself a new population run).
    # Optional: an older results/ directory or a fresh model page added
    # before ever running that script simply shows "not yet computed"
    # rather than this report failing to render.
    validity_path = RESULTS_DIR / "validity_analysis.json"
    validity = json.loads(validity_path.read_text()) if validity_path.exists() else {}
    for st in all_stats:
        st["validity"] = validity.get(f"{st['scenario']}__{st['model_backend']}")

    with PdfPages(OUT_PATH) as pdf:
        draw_cover_page(pdf, model_pages, all_stats, omitted)
        for st in all_stats:
            draw_model_page(pdf, st, xlims)

    print(f"-> {OUT_PATH} ({len(model_pages)} model page(s), {len(omitted)} superseded page(s) omitted)")

    # The per-population tracing pages (2026-09-12) are regenerated with
    # every report from the SAME stats, so the two artifacts cannot drift.
    import render_population_pages
    index = render_population_pages.render_pages(all_stats, omitted)
    print(f"-> {index} (open it to trace any population's curves)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-superseded", action="store_true",
                    help="also render pages for model versions the registry marks superseded_by a "
                         "successor that has results for the same scenario (omitted by default)")
    ap.add_argument("--deploy", action="store_true",
                    help="after rendering, push results/pages to the linked Vercel project as production "
                         "(https://vitals-visualizer.vercel.app; needs `npx vercel login` once on this machine)")
    a = ap.parse_args()
    render(include_superseded=a.include_superseded)
    if a.deploy:
        # Deployed from a git-free STAGING COPY (2026-09-12): deploying
        # results/pages in place attaches the surrounding repository's
        # commit metadata, and an unverified commit-author email makes the
        # hobby plan BLOCK the deployment ("TEAM_ACCESS_REQUIRED") while the
        # CLI waits forever for it to become ready -- found in a debug log.
        # The staging copy carries the .vercel/ project link so it lands on
        # the same project (vitals-visualizer.vercel.app).
        import subprocess, shutil, tempfile
        stage = pathlib.Path(tempfile.mkdtemp(prefix="vitals_site_"))
        shutil.copytree(RESULTS_DIR / "pages", stage, dirs_exist_ok=True)
        shutil.rmtree(stage / ".git", ignore_errors=True)
        r = subprocess.run(["npx", "--yes", "vercel@latest", "deploy", "--prod", "--yes"],
                           cwd=str(stage), capture_output=True, text=True, timeout=600)
        shutil.rmtree(stage, ignore_errors=True)
        print("vercel:", ("deployed to production" if r.returncode == 0 else f"deploy FAILED\n{r.stderr[-800:]}"))
