"""Renders results/gate1_survival.html from every results/l0_demo_*.json
found on disk -- auto-discovers scenarios, so a new scenario (e.g.
projectile, or whatever comes after ramp_descent's 3D generalization)
shows up automatically the next time this is re-run after its own
scripts/run_l0_demo.py pass, with no manual edits to this script or the
HTML needed.

This is GATE 1's own state-space survival report (Kaplan-Meier, VI_50 +
bootstrap CI, termination profile), now ALSO where real-model and baseline
populations land once run_cosmos_population.py / run_eval.py persist
their own results in the same schema.

Color/label scheme (2026-08, requested directly: "(C) for cosmos, (T) for
truth, cosmos lines blue, truth lines red"): every series is grouped into
one of three categories by its own `backend` field, each with its OWN
sequential ramp (light-to-dark, one hue) so multiple curves of the SAME
category (e.g. five different ground-truth scenarios) are still
distinguishable from each other, while every series in a category reads
as visually related at a glance:
  T (backend == "mujoco")                    -- ground-truth validation -- RED
  C (backend in vitals.adapters.REAL_MODEL_BACKENDS) -- real generative
                                                 models (Cosmos, Wan2.1,
                                                 any future one)  -- BLUE
  B (anything else)                          -- baseline predictors
                                                 (constant_velocity,
                                                 copy_last_state)  -- GREEN

`category_for` reads `REAL_MODEL_BACKENDS` from `vitals/adapters/
__init__.py` rather than hardcoding "cosmos" -- a real bug, found directly:
the original version only ever checked `backend == "cosmos"`, so Wan2.1's
own results silently fell through to "B" (baseline) the first time they
existed, categorizing a real model as if it were a trivial predictor.
Fixed by making this the ONE place both this script and `run_model_
population.py` read the same list from, not two independently-maintained
copies.

    python3 scripts/render_survival_report.py
    open results/gate1_survival.html
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "gate1_survival.html"
sys.path.insert(0, str(ROOT))
from vitals.adapters import REAL_MODEL_BACKENDS

# Sequential ramps (dataviz skill convention: one hue, light -> dark),
# NOT a shared cycled categorical palette -- shade within a ramp encodes
# "which curve of this category" (assigned by first-seen order among that
# category), the ramp's own hue encodes the category itself.
CATEGORY_RAMPS_LIGHT = {
    "T": ["#F87171", "#EF4444", "#DC2626", "#B91C1C", "#7F1D1D"],
    "C": ["#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8"],
    "B": ["#86EFAC", "#4ADE80", "#22C55E", "#16A34A", "#15803D"],
}
CATEGORY_RAMPS_DARK = {
    "T": ["#FCA5A5", "#F87171", "#EF4444", "#DC2626", "#B91C1C"],
    "C": ["#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB"],
    "B": ["#BBF7D0", "#86EFAC", "#4ADE80", "#22C55E", "#16A34A"],
}
CATEGORY_NAME = {"T": "Truth (ground-truth validation)", "C": "Real model", "B": "Baseline model"}


def category_for(backend):
    if backend == "mujoco":
        return "T"
    if backend in REAL_MODEL_BACKENDS:
        return "C"
    return "B"


def discover_results():
    files = sorted(RESULTS_DIR.glob("l0_demo_*.json"))
    out = {}
    for f in files:
        d = json.loads(f.read_text())
        key = f"{d['scenario']}_{d['backend']}"
        out[key] = d
    return out


def render(data):
    template = (ROOT / "scripts" / "_survival_report_template.html").read_text()
    keys = list(data.keys())

    cat_counts = {}
    series = []
    for k in keys:
        backend = data[k]["backend"]
        cat = category_for(backend)
        idx = cat_counts.get(cat, 0)
        cat_counts[cat] = idx + 1
        ramp_light, ramp_dark = CATEGORY_RAMPS_LIGHT[cat], CATEGORY_RAMPS_DARK[cat]
        base_label = f"{data[k]['scenario']} ({data[k]['target_property']})"
        if cat != "T":
            base_label += f" — {backend}"
        series.append({
            "key": k, "label": f"({cat}) {base_label}",
            "colorLight": ramp_light[idx % len(ramp_light)],
            "colorDark": ramp_dark[idx % len(ramp_dark)],
        })

    html = (template.replace("__DATA_JSON__", json.dumps(data))
                    .replace("__SERIES_JSON__", json.dumps(series)))
    OUT_PATH.write_text(html)
    print(f"{len(keys)} scenario(s): {', '.join(keys)}")
    for cat, name in CATEGORY_NAME.items():
        n = cat_counts.get(cat, 0)
        if n:
            print(f"  ({cat}) {name}: {n}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    data = discover_results()
    if not data:
        print(f"no results/l0_demo_*.json found -- run scripts/run_l0_demo.py first "
              f"(see its own docstring for VITALS_MANIFEST/VITALS_BACKEND usage)")
    else:
        render(data)
