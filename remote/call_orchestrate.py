"""Submit / check / pull population runs on the deployed `vitals-orchestrate`
Modal app (remote/modal_app_orchestrate.py) -- the same "look up the
persistent deployed app" pattern remote/call.py established.

`submit` uses `.spawn()` (fire-and-forget): the run keeps executing on
Modal after this process exits, so the laptop is no longer in the failure
path. Call ids are appended to results/orchestrate_calls.jsonl locally so
they survive a terminal closing.

Usage:
    python3 remote/call_orchestrate.py smoke
    python3 remote/call_orchestrate.py submit --model wan --scenario occlusion_corridor --seeds 1,2,3 --n-video-episodes 3
    python3 remote/call_orchestrate.py status <call_id>        # or `status` alone: all recorded calls
    python3 remote/call_orchestrate.py pull                    # results/ <- vitals-results Volume
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import modal

APP_NAME = "vitals-orchestrate"
ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "orchestrate_calls.jsonl"


def _record(entry):
    LEDGER.parent.mkdir(exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _status_of(call_id):
    fc = modal.FunctionCall.from_id(call_id)
    try:
        result = fc.get(timeout=0)
        return "done", result
    except TimeoutError:
        return "running", None
    except Exception as e:   # remote raised -- surface it, don't hide it
        return "failed", repr(e)


def main():
    # `submit` bypasses argparse on purpose: everything after it belongs to
    # run_model_population.py verbatim (`--model cosmos --seeds 1,2 ...`).
    # A subparser with nargs=REMAINDER does NOT capture leading options --
    # the parent parser rejects `--model` as unrecognized first (found
    # directly on the first real submit). Splitting sys.argv by hand is
    # the unambiguous fix, and matches how the command is actually typed.
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        script_args = [a for a in sys.argv[2:] if a != "--"]
        if not script_args:
            sys.exit("submit needs run_model_population.py args, e.g. --model cosmos --scenario collision --seeds 1,2")
        f = modal.Function.from_name(APP_NAME, "run_population")
        fc = f.spawn(script_args)
        entry = dict(call_id=fc.object_id, args=script_args, submitted_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        _record(entry)
        print(f"spawned {fc.object_id}\n  args: {' '.join(script_args)}\n  recorded in {LEDGER}")
        print(f"  check: python3 remote/call_orchestrate.py status {fc.object_id}")
        return

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    sub.add_parser("submit", help="everything after `submit` is passed verbatim to run_model_population.py")
    st = sub.add_parser("status")
    st.add_argument("call_id", nargs="?")
    pl = sub.add_parser("pull", help="results/ <- vitals-results Volume. NON-destructive by default: "
                                     "existing local files are left alone (modal volume get refuses to "
                                     "overwrite without --force) -- a test/partial orchestrated run must "
                                     "never silently clobber a real local population.")
    pl.add_argument("--dest", default=str(ROOT / "results"), help="local directory (default: results/)")
    pl.add_argument("--force", action="store_true", help="overwrite local files the Volume also has")
    pl.add_argument("--only", metavar="SCENARIO_MODEL",
                    help="fetch ONE finished population's three artifacts (l0_demo_<x>.json, videos/<x>_grid.mp4, "
                         "trajectories/<x>/) and overwrite them locally -- these ARE the intended new results. "
                         "Added 2026-09-11: the bulk pull is refused whenever ANY local trajectories dir already "
                         "exists (modal volume get is all-or-nothing without --force), which after the first "
                         "population is always.")
    args = p.parse_args()

    if args.command == "smoke":
        f = modal.Function.from_name(APP_NAME, "smoke_render")
        print(json.dumps(f.remote(), indent=2))
        return

    if args.command == "submit":   # unreachable (handled above); kept so --help lists it
        f = modal.Function.from_name(APP_NAME, "run_population")
        fc = f.spawn(script_args)
        entry = dict(call_id=fc.object_id, args=script_args, submitted_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        _record(entry)
        print(f"spawned {fc.object_id}\n  args: {' '.join(script_args)}\n  recorded in {LEDGER}")
        print(f"  check: python3 remote/call_orchestrate.py status {fc.object_id}")
        return

    if args.command == "status":
        ids = [args.call_id] if args.call_id else [
            json.loads(l)["call_id"] for l in open(LEDGER)] if LEDGER.exists() else []
        if not ids:
            print("no recorded calls"); return
        for cid in ids:
            state, payload = _status_of(cid)
            print(f"{cid}: {state}")
            if state == "done" and isinstance(payload, dict):
                surv = payload.get("survival", {})
                print(f"   {payload.get('_artifact')}  n={surv.get('n')} vi50={surv.get('vi50')} "
                      f"profile={surv.get('termination_profile')} failed_seeds={payload.get('failed_seeds')}")
            elif state == "failed":
                print(f"   {payload}")
        return

    if args.command == "pull":
        # `modal volume get` is the supported bulk path. --force is opt-in
        # (found directly: the first draft defaulted to --force, which would
        # have overwritten a real n=37 local result with a 2-episode test).
        dest = pathlib.Path(args.dest)
        dest.mkdir(parents=True, exist_ok=True)
        if args.only:
            x = args.only
            (dest / "videos").mkdir(exist_ok=True)
            (dest / "trajectories").mkdir(exist_ok=True)
            rc = 0
            for remote_path, local_dir in [(f"l0_demo_{x}.json", dest), (f"videos/{x}_grid.mp4", dest / "videos"),
                                           (f"trajectories/{x}", dest / "trajectories")]:
                cmd = ["modal", "volume", "get", "--force", "vitals-results", remote_path, str(local_dir)]
                print(" ".join(cmd))
                rc |= subprocess.call(cmd)
            sys.exit(rc)
        cmd = ["modal", "volume", "get", "vitals-results", "/", str(dest)] + (["--force"] if args.force else [])
        print(" ".join(cmd))
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
