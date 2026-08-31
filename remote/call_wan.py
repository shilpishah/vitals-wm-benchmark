"""Invoke a function on the deployed `vitals-wan` Modal app -- the same
"look up the persistent deployed app, don't `modal run`" pattern
`remote/call.py`/`remote/call_cosmos.py` already established (see
`remote/call.py`'s own docstring for why).

Usage:
    python3 remote/call_wan.py download_checkpoints
    python3 remote/call_wan.py image2video --input-path frame.png --prompt "..." --save-path out.mp4
"""
import argparse
import pathlib

import modal
import numpy as np

APP_NAME = "vitals-wan"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download_checkpoints")

    p_i2v = sub.add_parser("image2video")
    p_i2v.add_argument("--input-path", required=True, help="local conditioning image (e.g. a .png/.jpg)")
    p_i2v.add_argument("--prompt", required=True)
    p_i2v.add_argument("--save-path", required=True, help="local path to write the generated video to")
    p_i2v.add_argument("--seed", type=int, default=0)
    p_i2v.add_argument("--guidance-scale", type=float, default=5.0)
    p_i2v.add_argument("--num-inference-steps", type=int, default=50)

    args = parser.parse_args()

    if args.command == "download_checkpoints":
        f = modal.Function.from_name(APP_NAME, "download_checkpoints")
        result = f.remote()
        print(f"\nresult: {result}")
        return

    if args.command == "image2video":
        from PIL import Image
        import imageio

        image = np.asarray(Image.open(args.input_path).convert("RGB"))
        f = modal.Function.from_name(APP_NAME, "image2video")
        frames = f.remote(image=image, prompt=args.prompt, seed=args.seed,
                          guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps)
        imageio.mimwrite(args.save_path, frames, fps=16, quality=8)
        print(f"\nwrote {frames.shape[0]} frames -> {args.save_path}")
        return

    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
