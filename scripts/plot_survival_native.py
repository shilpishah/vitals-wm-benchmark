"""Native, non-browser survival-curve viewer -- the same data
`render_survival_report.py` renders to HTML, plotted instead with
matplotlib's own native GUI window (macosx/Qt/Tk backend, whichever this
machine has -- confirmed NOT a browser). Built per direct request
(2026-08: "i would rather a plotting tool without touching a browser at
all").

Interactive exactly like the HTML version, just through a native window
instead of a DOM: move the mouse over the plot and a crosshair + text
readout shows the EXACT (t, S) value of every series under the cursor,
second by second -- the same "trace it at every instant" the HTML
version's own hover tooltip gives, implemented here via a matplotlib
`motion_notify_event` handler instead of JS.

Same conventions as the HTML report, ported directly, not reinvented:
fixed categorical color order, solid line for observed data, dashed for
"censored past this scenario's own clip end" (each scenario's own t_max,
not necessarily the same clip length as others on the shared axis),
"REAL MODEL" suffix for any backend != "mujoco".

`--model <name>` restricts the view to one real model + its own
baselines (constant_velocity, copy_last_state) -- the same three series
`render_score_report.py` puts on that model's own PDF page, in the same
colors (imported directly from that module so the two artifacts can't
drift apart), giving this page's own degradation curves live,
second-by-second crosshair tracing (2026-08, requested directly: "the
degradation curves should ... come up in a separate viewer to allow for
live second by second tracing"). Without `--model`, behavior is
unchanged: every result found, generic cycled palette -- the GATE 1
exploratory view.

    python3 scripts/plot_survival_native.py
    python3 scripts/plot_survival_native.py --model cosmos
"""
import argparse
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Same fixed categorical palette as the HTML report (dataviz skill
# convention) -- light-mode values only; a native window has no dark-mode
# concept to match here. Used only in the unfiltered (no --model) view.
PALETTE = ["#3B82C4", "#D9782A", "#5B8C5A", "#9B59A6", "#C0392B", "#2E9E9E"]


def discover_results():
    files = sorted(RESULTS_DIR.glob("l0_demo_*.json"))
    out = {}
    for f in files:
        d = json.loads(f.read_text())
        key = f"{d['scenario']}_{d['backend']}"
        out[key] = d
    return out


def label_for(d):
    base = f"{d['scenario']} ({d['target_property']})"
    return base if d["backend"] == "mujoco" else f"{base} — {d['backend']}, REAL MODEL"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None,
                        help="restrict to one real model + its own baselines (matches that model's "
                             "score_report.pdf page, same colors, same series order); omit to see everything")
    parser.add_argument("--scenario", default="occlusion_corridor")
    args = parser.parse_args()

    data = discover_results()
    if not data:
        print("no results/l0_demo_*.json found -- run scripts/run_l0_demo.py "
              "or scripts/run_model_population.py first")
        return

    if args.model:
        from render_score_report import COLOR_MODEL, COLOR_BASELINE, BASELINE_BACKENDS
        model_key = f"{args.scenario}_{args.model}"
        if model_key not in data:
            print(f"no results for model={args.model!r} scenario={args.scenario!r} -- "
                  f"run scripts/run_model_population.py --model {args.model} --scenario {args.scenario} first")
            return
        wanted = [model_key] + [f"{args.scenario}_{b}" for b in BASELINE_BACKENDS]
        keys = [k for k in wanted if k in data]  # canonical order: model, constant_velocity, copy_last_state
        colors = {f"{args.scenario}_{args.model}": COLOR_MODEL}
        colors.update({f"{args.scenario}_{b}": COLOR_BASELINE[b] for b in BASELINE_BACKENDS})
        window_title = f"VITALS -- {args.model} vs. baseline ({args.scenario}, native)"
    else:
        keys = list(data.keys())
        colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}
        window_title = "VITALS -- Survival Curves (native)"

    labels = {k: label_for(data[k]) for k in keys}

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title(window_title)

    curves = {}   # key -> (t_solid, S_solid, t_dashed, S_dashed)
    for k in keys:
        km = data[k]["survival"]["kaplan_meier"]
        t, S = np.array(km["t"]), np.array(km["S"])
        t_max = data[k]["t_max"]
        line, = ax.step(t, S, where="post", color=colors[k], linewidth=2, label=labels[k])
        if t_max > t[-1]:
            ax.plot([t[-1], t_max], [S[-1], S[-1]], color=colors[k], linewidth=2,
                   linestyle=(0, (2, 3)), alpha=0.55)
        curves[k] = (t, S)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("S(t) -- fraction still valid")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Kaplan-Meier survival curves\n(dashed = censored past that scenario's own clip end)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # Crosshair + live text readout -- the native equivalent of the HTML
    # report's own hover tooltip. One vertical line shared across all
    # series (matching the HTML version's own crosshair), text box lists
    # every series' S(t) at the hovered time.
    vline = ax.axvline(0, color="0.4", linestyle=":", linewidth=1, visible=False)
    readout = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9, va="bottom",
                      bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            vline.set_visible(False)
            readout.set_text("")
            fig.canvas.draw_idle()
            return
        t_hover = event.xdata
        vline.set_xdata([t_hover, t_hover])
        vline.set_visible(True)
        lines = [f"t = {t_hover:.2f}s"]
        for k in keys:
            t, S = curves[k]
            idx = np.searchsorted(t, t_hover, side="right") - 1
            idx = max(0, min(idx, len(S) - 1))
            lines.append(f"  {labels[k]}: S={S[idx]:.3f}")
        readout.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    print(f"{len(keys)} scenario(s) loaded: {', '.join(keys)}")
    print("move the mouse over the plot to trace exact values at any instant -- close the window to exit.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
