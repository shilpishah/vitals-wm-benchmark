"""Renders results/gate2_instrument_tax.html from every
results/gate2_instances_*.json (+ its matching gate2_artifact_sweep_*.json,
if present) found on disk -- same auto-discovery convention as
render_survival_report.py, so a new scenario's GATE 2 pass shows up
automatically next time this is re-run, no manual edits needed.

This is GATE 2 (AGENT.md M5, "the instrument tax"): the SAME mutants,
measured TWICE -- once through ground-truth state (L0, exactly what GATE
1's survival curves use) and once through the real video -> Phi pipeline
(L1, real SAM2/DINOv2 tracking + 3D reconstruction). The gap between L0 and
L1 on IDENTICAL episodes is the instrument's own error floor -- this report
is the visual form of that gap, not a new measurement.

    python3 scripts/render_gate2_report.py
    open results/gate2_instrument_tax.html
"""
import json
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "gate2_instrument_tax.html"

PALETTE_LIGHT = ["#3B82C4", "#D9782A", "#5B8C5A", "#9B59A6", "#C0392B", "#2E9E9E"]
PALETTE_DARK = ["#6FB3E8", "#F0A059", "#7FB37E", "#C084D4", "#E07A6E", "#4FC4C4"]
MTYPE_ORDER = ["null", "vanish", "velocity_freeze", "jitter", "wrong_gravity", "duplicate", "teleport"]


def discover_instances():
    files = sorted(RESULTS_DIR.glob("gate2_instances_*.json"))
    out = {}
    for f in files:
        # gate2_instances_<scenario>_<backend>.json
        key = f.stem[len("gate2_instances_"):]
        instances = json.loads(f.read_text())
        sweep_path = RESULTS_DIR / f"gate2_artifact_sweep_{key}.json"
        sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else None
        out[key] = dict(instances=instances, artifact_sweep=sweep)
    return out


def summarize(instances):
    """Per-mtype rollup -- the same numbers AGENT.md's own published table
    reports, computed live from the raw per-episode data rather than
    hand-copied, so this can never silently drift out of sync with it."""
    by_type = defaultdict(list)
    for x in instances:
        by_type[x["mtype"]].append(x)

    rows = []
    for mtype in MTYPE_ORDER:
        xs = by_type.get(mtype)
        if not xs:
            continue
        n = len(xs)
        l0_fired = sum(1 for x in xs if not x["l0_censored"])
        l1_fired = sum(1 for x in xs if not x["l1_censored"])
        correct_l0 = sum(1 for x in xs if not x["l0_censored"] and x["l0_risk"] == x["risk_expected"])
        correct_l1 = sum(1 for x in xs if not x["l1_censored"] and x["l1_risk"] == x["risk_expected"])
        joint = [x["l1_time"] - x["l0_time"] for x in xs if not x["l0_censored"] and not x["l1_censored"]]
        bias_mean = sum(joint) / len(joint) if joint else None
        bias_std = (sum((b - bias_mean) ** 2 for b in joint) / len(joint)) ** 0.5 if joint else None
        ious = [x["mean_iou"] for x in xs if x.get("mean_iou") is not None]
        occ_accs = [x["occlusion_flag_accuracy"] for x in xs if x.get("occlusion_flag_accuracy") is not None]
        pos_errs = [x["position_error"] for x in xs if x.get("position_error") is not None]
        rows.append(dict(
            mtype=mtype, n=n, l0_fired=l0_fired, l1_fired=l1_fired,
            correct_l0=correct_l0, correct_l1=correct_l1,
            bias_mean=bias_mean, bias_std=bias_std, bias_n=len(joint),
            mean_iou=sum(ious) / len(ious) if ious else None,
            mean_occ_acc=sum(occ_accs) / len(occ_accs) if occ_accs else None,
            mean_pos_err=sum(pos_errs) / len(pos_errs) if pos_errs else None,
        ))
    return rows


def summarize_sweep(sweep):
    """Groups the artifact-injection sweep by (artifact type), each a
    series of (severity, mean_iou) points -- the "how does measurement
    quality degrade under graded, generation-style artifacts" chart."""
    if not sweep:
        return None
    by_artifact = defaultdict(list)
    for x in sweep:
        by_artifact[x["artifact"]].append((x["severity"], x["mean_iou"]))
    return {k: sorted(v) for k, v in by_artifact.items()}


def render(data):
    template = (ROOT / "scripts" / "_gate2_report_template.html").read_text()
    keys = list(data.keys())
    scenarios = []
    for i, k in enumerate(keys):
        rows = summarize(data[k]["instances"])
        sweep = summarize_sweep(data[k]["artifact_sweep"])
        scenarios.append(dict(
            key=k, label=k.replace("_mujoco", ""),
            colorLight=PALETTE_LIGHT[i % len(PALETTE_LIGHT)], colorDark=PALETTE_DARK[i % len(PALETTE_DARK)],
            summary=rows,
            instances=data[k]["instances"],
            sweep=sweep,
        ))
    html = template.replace("__SCENARIOS_JSON__", json.dumps(scenarios))
    OUT_PATH.write_text(html)
    print(f"{len(keys)} scenario(s): {', '.join(keys)}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    data = discover_instances()
    if not data:
        print("no results/gate2_instances_*.json found -- run scripts/run_gate2.py first "
              "(needs the remote GPU/Modal environment; see its own docstring)")
    else:
        render(data)
