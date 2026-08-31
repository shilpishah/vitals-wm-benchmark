"""Invoke a function on the already-deployed `vitals-phi` Modal app.

Deliberately NOT `modal run remote/modal_app.py::<function>` -- that always
spins up a fresh ephemeral app per invocation (this is `modal run`'s actual
documented behavior, not a bug we hit), which is exactly what cluttered the
midcentury-labs dashboard with a growing pile of one-call, already-dead
"vitals-phi" app entries during iterative debugging. This script instead
looks up the ONE persistent deployed app (`modal deploy remote/modal_app.py`)
and calls directly into it -- every invocation is logged as a call against
the same app, not a new app.

Re-run `modal deploy remote/modal_app.py` only when the image itself changes
(new pip dependency, etc.) -- vitals/phi/*.py is mounted at container
startup (add_local_dir), so ordinary code edits take effect on the next
call here with no redeploy needed.

Usage:
    python3 remote/call.py smoke_test
    python3 remote/call.py upload_episode --local-dir /path/to/full_episode --name occlusion_corridor_seed2024
    python3 remote/call.py run_reidentification --name occlusion_corridor_seed2024 --debug
"""
import argparse
import pathlib
import sys

import modal

APP_NAME = "vitals-phi"


def upload_episode(local_dir: str, name: str):
    """Runs locally (reads local files), unlike the other subcommands --
    matches modal_app.py's own upload_episode local_entrypoint, duplicated
    here so this script is the single entry point for everything, not just
    the GPU functions."""
    volume = modal.Volume.from_name("vitals-model-cache", create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(pathlib.Path(local_dir)), f"/episodes/{name}")
    print(f"uploaded {local_dir} -> volume:/episodes/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke_test")

    p_upload = sub.add_parser("upload_episode")
    p_upload.add_argument("--local-dir", required=True)
    p_upload.add_argument("--name", required=True)

    p_reid = sub.add_parser("run_reidentification")
    p_reid.add_argument("--name", required=True)
    p_reid.add_argument("--similarity-threshold", type=float, default=0.4)
    p_reid.add_argument("--forgiveness-frames", type=int, default=1)
    p_reid.add_argument("--debug", action="store_true")
    p_reid.add_argument("--no-metric-prior", action="store_true",
                         help="reproduce the OLD pixel-only prediction path (pre defect #16 fix), for comparison")

    p_diag = sub.add_parser("diagnose_candidates")
    p_diag.add_argument("--name", required=True)
    p_diag.add_argument("--frames", default="57,59,61,65")

    p_gate2 = sub.add_parser("run_gate2_episode")
    p_gate2.add_argument("--name", required=True)
    p_gate2.add_argument("--similarity-threshold", type=float, default=0.4)
    p_gate2.add_argument("--forgiveness-frames", type=int, default=1)
    p_gate2.add_argument("--no-metric-prior", action="store_true")

    p_multi = sub.add_parser("run_multiobject_episode")
    p_multi.add_argument("--name", required=True)
    p_multi.add_argument("--similarity-threshold", type=float, default=0.4)
    p_multi.add_argument("--forgiveness-frames", type=int, default=1)
    p_multi.add_argument("--no-metric-prior", action="store_true")

    args = parser.parse_args()

    if args.command == "upload_episode":
        upload_episode(args.local_dir, args.name)
        return

    if args.command == "smoke_test":
        f = modal.Function.from_name(APP_NAME, "smoke_test")
        result = f.remote()
    elif args.command == "run_reidentification":
        f = modal.Function.from_name(APP_NAME, "run_reidentification")
        result = f.remote(name=args.name, similarity_threshold=args.similarity_threshold,
                           forgiveness_frames=args.forgiveness_frames, debug=args.debug,
                           use_metric_prior=not args.no_metric_prior)
    elif args.command == "diagnose_candidates":
        f = modal.Function.from_name(APP_NAME, "diagnose_candidates")
        result = f.remote(name=args.name, frames=args.frames)
    elif args.command == "run_gate2_episode":
        f = modal.Function.from_name(APP_NAME, "run_gate2_episode")
        result = f.remote(name=args.name, similarity_threshold=args.similarity_threshold,
                           forgiveness_frames=args.forgiveness_frames,
                           use_metric_prior=not args.no_metric_prior)
    elif args.command == "run_multiobject_episode":
        f = modal.Function.from_name(APP_NAME, "run_multiobject_episode")
        result = f.remote(name=args.name, similarity_threshold=args.similarity_threshold,
                           forgiveness_frames=args.forgiveness_frames,
                           use_metric_prior=not args.no_metric_prior)
    else:
        parser.error(f"unknown command {args.command}")
        return

    print(f"\nresult: {result}")


if __name__ == "__main__":
    main()
