"""Invoke a function on the deployed `vitals-cosmos` Modal app -- the same
"look up the persistent deployed app, don't `modal run`" pattern
`remote/call.py` already established for `vitals-phi` (see that file's own
docstring for why: `modal run` spins up a fresh ephemeral app per call).

Usage:
    python3 remote/call_cosmos.py download_checkpoints
    python3 remote/call_cosmos.py download_checkpoints --model-sizes 14B
    python3 remote/call_cosmos.py video2world --input-path prefix.mp4 --prompt "..." --save-path out.mp4
    python3 remote/call_cosmos.py video2world --model-size 14B --input-path prefix.mp4 --prompt "..." --save-path out.mp4
"""
import argparse
import pathlib

import modal

APP_NAME = "vitals-cosmos"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download_checkpoints")
    p_dl.add_argument("--model-sizes", nargs="+", default=["2B"], choices=["2B", "14B"],
                      help="which Video2World checkpoint(s) to download -- each size needs its OWN "
                           "Hugging Face gated-access acceptance first (2B and 14B are gated "
                           "separately, see modal_app_cosmos.py's own docstring)")

    p_v2w = sub.add_parser("video2world")
    p_v2w.add_argument("--input-path", required=True, help="local prefix video file (e.g. an mp4)")
    p_v2w.add_argument("--prompt", required=True)
    p_v2w.add_argument("--save-path", required=True, help="local path to write the generated continuation to")
    p_v2w.add_argument("--num-conditional-frames", type=int, default=5)
    p_v2w.add_argument("--fps", type=int, default=16)
    p_v2w.add_argument("--resolution", default="480", choices=["480", "720"])
    p_v2w.add_argument("--seed", type=int, default=0)
    p_v2w.add_argument("--model-size", default="2B", choices=["2B", "14B"])

    args = parser.parse_args()

    if args.command == "download_checkpoints":
        f = modal.Function.from_name(APP_NAME, "download_checkpoints")
        result = f.remote(model_sizes=tuple(args.model_sizes))
        print(f"\nresult: {result}")
        return

    if args.command == "video2world":
        f = modal.Function.from_name(APP_NAME, "video2world")
        prefix_bytes = pathlib.Path(args.input_path).read_bytes()
        out_bytes = f.remote(prefix_video_bytes=prefix_bytes, prompt=args.prompt,
                             num_conditional_frames=args.num_conditional_frames,
                             fps=args.fps, resolution=args.resolution, seed=args.seed,
                             model_size=args.model_size)
        pathlib.Path(args.save_path).write_bytes(out_bytes)
        print(f"\nwrote {len(out_bytes)} bytes -> {args.save_path}")
        return

    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
