"""M7 -- seed test-retest + minimum detectable difference, computed from
ALREADY-COLLECTED populations only (no new GPU/Modal cost -- reuses every
`results/l0_demo_*.json` this project already has). Answers the two
questions `bootstrap_vi_delta`'s own docstring flags as still missing:
"if I reran this exact population, would VI_50 come back the same?" and
"how big does a true difference need to be before it's even worth
bootstrapping in the first place?"

    python3 scripts/run_validity_analysis.py
    python3 scripts/run_validity_analysis.py --scenario occlusion_corridor --backend cosmos

Small populations are reported honestly, not silently skipped or
inflated to look more confident than they are: n<4 can't run
seed_test_retest at all (needs >=2 per half) and is skipped with a
stated reason; 4<=n<MIN_N_FOR_MEANINGFUL is run and reported but flagged
`small_n=True` in the output -- e.g. Wan's current n=5 populations sit
right at this floor.
"""
import argparse
import json
import pathlib
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "validity_analysis.json"

MIN_N_FOR_MEANINGFUL = 20   # below this, test-retest/MDD are computed but flagged unreliable, not refused


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=None, help="restrict to one scenario (default: every l0_demo_*.json found)")
    ap.add_argument("--backend", default=None, help="restrict to one backend")
    ap.add_argument("--n-sim", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    from vitals.types import Event
    from vitals.stats.survival import seed_test_retest, minimum_detectable_difference

    results = {}
    for path in sorted(RESULTS_DIR.glob("l0_demo_*.json")):
        d = json.loads(path.read_text())
        scenario, backend = d.get("scenario"), d.get("backend")
        if not scenario or not backend:
            continue
        if args.scenario and scenario != args.scenario:
            continue
        if args.backend and backend != args.backend:
            continue
        s = d["survival"]
        n = s["n"]
        events = [Event(time=e["time"], risk=e["risk"], censored=e["censored"]) for e in s["events"]]
        t_max = d["t_max"]

        entry = dict(scenario=scenario, backend=backend, n=n, t_max=t_max)
        if n < 4:
            entry["skipped_reason"] = f"n={n} < 4, seed_test_retest needs >=2 per half"
            print(f"{scenario}/{backend}: SKIPPED ({entry['skipped_reason']})")
            results[f"{scenario}__{backend}"] = entry
            continue

        entry["small_n"] = n < MIN_N_FOR_MEANINGFUL
        t0 = time.time()
        tr = seed_test_retest(events, n_splits=200, seed=0)
        entry["seed_test_retest"] = tr
        mdd = minimum_detectable_difference(events, t_max=t_max, n_sim=args.n_sim, n_boot=args.n_boot, seed=0)
        entry["minimum_detectable_difference"] = mdd
        dt = time.time() - t0

        flag = " [SMALL n -- treat as barely-informative]" if entry["small_n"] else ""
        # A reference population with ZERO observed spread (Cosmos's own
        # zero-variance results, or a near-never-terminates GATE 1a null
        # where nearly every split has no observed failures at all) makes
        # the resulting MDD number close to vacuous -- "any nonzero shift
        # is trivially detectable against literally no noise" is not the
        # same claim as "this is how small a real difference could be and
        # still get caught," and reporting the bare number without saying
        # so invites exactly that misread.
        vacuous = (not (tr["median_abs_delta"] == tr["median_abs_delta"])   # NaN check without importing math/np here
                   or tr["median_abs_delta"] == 0.0)
        entry["mdd_caveat"] = ("reference population has zero/undefined observed test-retest spread -- "
                               "MDD number is not a meaningful real-world detection threshold") if vacuous else None
        tr_str = (f"{tr['median_abs_delta']:.3f}s (p90={tr['p90_abs_delta']:.3f}s)"
                  if tr["median_abs_delta"] == tr["median_abs_delta"] else "undefined (no split had 2 observed failures)")
        mdd_str = f"{mdd['mdd']:.3f}s" if mdd["mdd"] is not None else "not reached within grid"
        caveat_str = "  [VACUOUS -- see mdd_caveat]" if vacuous else ""
        print(f"{scenario}/{backend} (n={n}){flag}: "
              f"test-retest median|delta|={tr_str}, "
              f"MDD(power>={mdd['target_power']:.0%})={mdd_str}{caveat_str}  [{dt:.1f}s]")
        results[f"{scenario}__{backend}"] = entry

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
