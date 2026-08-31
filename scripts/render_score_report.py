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

    python3 scripts/render_score_report.py
    open results/score_report.pdf
    python3 scripts/plot_survival_native.py --model cosmos   # live tracing, same data
"""
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "score_report.pdf"

BASELINE_BACKENDS = ["constant_velocity", "copy_last_state"]
MIN_N_FOR_STATS = 10
MIN_TIME_AXIS = 0.5  # shared lower bound for every failure-time axis, so t=0.5 always means the same thing on the page

COLOR_MODEL = "#2563EB"
COLOR_BASELINE = {"constant_velocity": "#4ADE80", "copy_last_state": "#16A34A"}
COLOR_INK = "#17171b"
COLOR_MUTED = "#6b6b76"
COLOR_CARD = "#f4f5f7"
COLOR_BORDER = "#dcdce2"
COLOR_BAD = "#B3261E"
COLOR_OK = "#0F7B4F"
COLOR_RISK = {"R2": "#7C3AED", "R4": "#D97706", "R5": "#2563EB"}
RISK_LABEL = {"R2": "purple = R2 existence", "R4": "orange = R4 interpenetration", "R5": "blue = R5 kinematic"}

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
MODEL_PAGE_H = 13.0
L, R = 0.09, 0.93


def fmt(v, d=2):
    if v is None:
        return "—"
    if isinstance(v, float) and v == float("inf"):
        return "∞"
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
    """The one canonical order every chart on a page must follow:
    model, then baselines in BASELINE_BACKENDS order."""
    return [st["model_backend"]] + [b for b, _ in st["baselines"]]


def series_data(st, name):
    return st["model_data"] if name == st["model_backend"] else dict(st["baselines"])[name]


def series_color(st, name):
    return COLOR_MODEL if name == st["model_backend"] else COLOR_BASELINE.get(name, "#999")


def compute_stats(scenario, model_backend, all_results, bootstrap_vi_delta, Event):
    model = all_results[(scenario, model_backend)]
    s = model["survival"]
    n = s["n"]
    small_n = n < MIN_N_FOR_STATS

    baselines = [(b, all_results[(scenario, b)]) for b in BASELINE_BACKENDS if (scenario, b) in all_results]

    deltas = []
    model_events = [Event(time=e["time"], risk=e["risk"], censored=e["censored"]) for e in s["events"]]
    for b_name, b_data in baselines:
        b_s = b_data["survival"]
        if small_n or b_s["n"] < MIN_N_FOR_STATS:
            deltas.append(dict(name=b_name, point=None, lo=None, hi=None, verdict="insufficient sample"))
            continue
        b_events = [Event(time=e["time"], risk=e["risk"], censored=e["censored"]) for e in b_s["events"]]
        point, lo, hi = bootstrap_vi_delta(model_events, b_events, q=0.5, n_boot=2000, seed=0)
        if hi < 0:
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

    beats_all = (
        s["vi50"] != float("inf")
        and all(b_data["survival"]["vi50"] != float("inf") and s["vi50"] < b_data["survival"]["vi50"]
                for _, b_data in baselines)
    ) if baselines else None

    return dict(scenario=scenario, model_backend=model_backend, n=n, small_n=small_n,
                vi50=s["vi50"], ci=s["vi50_ci95"], censoring=s["censoring_rate"],
                t_max=model["t_max"], baselines=baselines, deltas=deltas,
                frac_at_first=frac_at_first, dominant_risk=dominant_risk, prof=prof,
                beats_all=beats_all, model_data=model)


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
    (always exactly model + 2 baselines)."""
    names = series_order(st)
    xs = [L, L + 0.28, L + 0.56]
    for name, x in zip(names, xs):
        color = series_color(st, name)
        role = "real model" if name == st["model_backend"] else "baseline"
        fig.text(x, cur_y_frac, "●", fontsize=12, color=color, va="center")
        fig.text(x + 0.02, cur_y_frac, f"{name}", fontsize=8.5, color=COLOR_INK, va="center", fontweight="medium")
        fig.text(x + 0.02, cur_y_frac - 0.014, role, fontsize=6.5, color=COLOR_MUTED, va="center")


def draw_model_page(pdf, st, xlims):
    fig = plt.figure(figsize=(PAGE_W, MODEL_PAGE_H))

    def top(cur_in):
        return 1 - cur_in / MODEL_PAGE_H

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

    if st["small_n"]:
        band_h = 0.30
        band = fig.add_axes([L, top(cur + band_h), R - L, band_h / MODEL_PAGE_H])
        band.axis("off")
        band.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15",
                                       linewidth=0, facecolor="#FCE8B2", transform=band.transAxes, clip_on=False))
        band.text(0.015, 0.5, f"sample size n={st['n']} is below the n={MIN_N_FOR_STATS} minimum for "
                               f"significance testing — figures below are illustrative only",
                   transform=band.transAxes, fontsize=8, va="center", color="#5C3D00", fontweight="medium")
        cur += band_h + 0.20
    else:
        cur += 0.16

    # --- stat cards row --------------------------------------------------
    cards_h = 1.05
    cards_ax = fig.add_axes([L, top(cur + cards_h), R - L, cards_h / MODEL_PAGE_H])
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
    cards_ax.text(x + 0.02, 0.08, f"CI [{fmt(st['ci'][0])}, {fmt(st['ci'][1])}]  ·  censor {fmt(st['censoring'])}",
                  fontsize=6.6, color=COLOR_MUTED, transform=cards_ax.transAxes)
    x += cw_frac + gap_frac

    for d in st["deltas"]:
        card(cards_ax, x, 0, cw_frac, 1)
        cards_ax.text(x + 0.02, 0.72, f"Δ vs. {d['name']}", fontsize=7.2, color=COLOR_MUTED, transform=cards_ax.transAxes)
        if d["point"] is None:
            cards_ax.text(x + 0.02, 0.30, "n/a", fontsize=14, color=COLOR_MUTED, transform=cards_ax.transAxes)
            cards_ax.text(x + 0.02, 0.08, "insufficient sample", fontsize=6.6, color=COLOR_MUTED, transform=cards_ax.transAxes)
        else:
            vcolor = COLOR_BAD if "worse" in d["verdict"] else (COLOR_OK if "better" in d["verdict"] else COLOR_INK)
            cards_ax.text(x + 0.02, 0.30, f"{fmt(d['point'])}s", fontsize=16, color=vcolor, fontweight="bold", transform=cards_ax.transAxes)
            cards_ax.text(x + 0.02, 0.08, f"CI [{fmt(d['lo'])}, {fmt(d['hi'])}]s · {d['verdict']}",
                          fontsize=6.6, color=COLOR_MUTED, transform=cards_ax.transAxes)
        x += cw_frac + gap_frac
    cur += cards_h + 0.30

    # --- degradation curves (Kaplan-Meier survival) -----------------------
    title_h, xlabel_h = 0.30, 0.34
    km_ax_h = 2.05
    ax_km = fig.add_axes([L, top(cur + title_h + km_ax_h), R - L, km_ax_h / MODEL_PAGE_H])
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
    ax_bar = fig.add_axes([L + LABEL_INSET, top(cur + title_h + row_ax_h), 0.24, row_ax_h / MODEL_PAGE_H])
    ax_box = fig.add_axes([L + LABEL_INSET + 0.24 + 0.13, top(cur + title_h + row_ax_h),
                            R - (L + LABEL_INSET + 0.24 + 0.13), row_ax_h / MODEL_PAGE_H])

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
    ax_bar.set_yticklabels(names, fontsize=7.5)
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
    ax_box.set_yticklabels(order_names, fontsize=7.5)
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
                             R - (L + LABEL_INSET), mode_ax_h / MODEL_PAGE_H])
    risk_keys = sorted({k for name in names for k in series_data(st, name)["survival"]["termination_profile"].keys()})
    left = np.zeros(len(names))
    yy = np.arange(len(names))[::-1]
    for rk in risk_keys:
        vals = np.array([series_data(st, name)["survival"]["termination_profile"].get(rk, 0.0) * 100 for name in names])
        ax_mode.barh(yy, vals, left=left, color=COLOR_RISK.get(rk, "#999"), height=0.55)
        left += vals
    ax_mode.set_yticks(yy)
    ax_mode.set_yticklabels(names, fontsize=7.5)
    ax_mode.set_xlim(0, 100)
    ax_mode.set_xlabel("% of terminations", fontsize=8)
    # Built from whichever risk keys are actually present, not hardcoded to
    # R2/R5 -- a P3-scored model's own R4 (interpenetration) terminations
    # must show up here correctly labeled, not silently mislabeled or
    # dropped to the "#999" fallback color.
    legend_bits = ", ".join(RISK_LABEL.get(rk, rk) for rk in risk_keys)
    ax_mode.set_title(f"Failure mode ({legend_bits})", fontsize=9, fontweight="bold", loc="left", pad=8)
    ax_mode.spines[["top", "right"]].set_visible(False)
    ax_mode.tick_params(labelsize=7.5)
    cur += title_h + mode_ax_h + xlabel_h + 0.24

    # --- computed readout (discrete fields, not prose) --------------------
    lines = [
        f"beats_all_baselines = {st['beats_all']}",
        f"frac_failures_at_earliest_instant = {fmt(st['frac_at_first']*100, 1)}%",
        f"dominant_failure_mode = {st['dominant_risk']} ({fmt(st['prof'].get(st['dominant_risk'], 0)*100, 1)}% of terminations)",
    ]
    for d in st["deltas"]:
        v = d["verdict"] if d["point"] is None else f"{fmt(d['point'])}s, {d['verdict']}"
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

    # Card height AND inter-line spacing both scale with len(lines) --
    # originally a fixed 5-line/0.145-axes-fraction design (hardcoded
    # read_h=1.35), which silently overflowed the card's own [0,1] axes
    # bounds once M7's two new lines pushed a real page (with 2 baseline
    # deltas) to 7 lines (found directly: 0.62 - 6*0.145 = -0.25, well
    # below the card). BASE_LINES=5 reproduces the ORIGINAL fixed 0.145
    # spacing exactly when len(lines)==5 (0.58/(5-1)=0.145) -- a
    # generalization, not a behavior change, for every page that still
    # has exactly 5 lines.
    BASE_LINES = 5
    n_lines = len(lines)
    read_h = 1.35 * max(1.0, n_lines / BASE_LINES)
    line_spacing = 0.58 / max(1, n_lines - 1)
    ax_read = fig.add_axes([L, top(cur + read_h), R - L, read_h / MODEL_PAGE_H])
    ax_read.axis("off")
    card(ax_read, 0, 0, 1, 1)
    ax_read.text(0.02, 0.85, "COMPUTED READOUT", fontsize=7, color=COLOR_MUTED, fontweight="bold", transform=ax_read.transAxes)
    for i, line in enumerate(lines):
        ax_read.text(0.02, 0.62 - i * line_spacing, line, fontsize=8, color=COLOR_INK, family="monospace", transform=ax_read.transAxes)
    cur += read_h

    fig.text(L, top(MODEL_PAGE_H - 0.32),
             "Generated by scripts/render_score_report.py from results/l0_demo_*.json.",
             fontsize=6.5, color=COLOR_MUTED)

    pdf.savefig(fig)
    plt.close(fig)


def draw_cover_page(pdf, model_pages, all_stats):
    fig = plt.figure(figsize=(PAGE_W, COVER_H))
    fig.text(0.5, 0.74, "VITALS", fontsize=40, ha="center", fontweight="bold", color=COLOR_INK)
    fig.text(0.5, 0.69, "Score Report", fontsize=18, ha="center", color=COLOR_MUTED)
    fig.text(0.5, 0.645, "Real model vs. baseline, per scenario", fontsize=11, ha="center", color=COLOR_MUTED)

    y = 0.56
    fig.text(0.15, y, "COLOR KEY", fontsize=8, color=COLOR_MUTED, fontweight="bold")
    y -= 0.03
    key_items = [("real model", COLOR_MODEL), ("constant_velocity baseline", COLOR_BASELINE["constant_velocity"]),
                 ("copy_last_state baseline", COLOR_BASELINE["copy_last_state"])]
    for label, color in key_items:
        fig.text(0.17, y, "●", fontsize=12, color=color, va="center")
        fig.text(0.19, y, label, fontsize=9.5, color=COLOR_INK, va="center")
        y -= 0.028

    y -= 0.03
    fig.text(0.15, y, "CONTENTS", fontsize=8, color=COLOR_MUTED, fontweight="bold")
    for scenario, model_backend in model_pages:
        y -= 0.028
        fig.text(0.17, y, f"{scenario} — {model_backend}", fontsize=10, color=COLOR_INK, family="monospace")

    pdf.savefig(fig)
    plt.close(fig)


def render():
    from vitals.adapters import REAL_MODEL_BACKENDS
    from vitals.stats.survival import bootstrap_vi_delta
    from vitals.detect.events import Event

    all_results = discover_all()
    model_pages = sorted((s, b) for (s, b) in all_results if b in REAL_MODEL_BACKENDS)
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
        draw_cover_page(pdf, model_pages, all_stats)
        for st in all_stats:
            draw_model_page(pdf, st, xlims)

    print(f"-> {OUT_PATH} ({len(model_pages)} model page(s))")


if __name__ == "__main__":
    render()
