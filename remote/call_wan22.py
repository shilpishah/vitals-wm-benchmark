"""Invoke a function on the deployed `vitals-wan22` Modal app (same pattern
as every other remote/call_*.py).

Usage:
    python3 remote/call_wan22.py download_checkpoints
    python3 remote/call_wan22.py image2video --input-path frame.png --prompt "..." --save-path out.mp4
"""
import argparse

import modal
import numpy as np

APP_NAME = "vitals-wan22"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download_checkpoints")
    p = sub.add_parser("image2video")
    p.add_argument("--input-path", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--save-path", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=81)
    args = parser.parse_args()

    if args.command == "download_checkpoints":
        print(f"\nresult: {modal.Function.from_name(APP_NAME, 'download_checkpoints').remote()}")
        return

    from PIL import Image
    import imageio
    image = np.asarray(Image.open(args.input_path).convert("RGB"))
    frames = modal.Function.from_name(APP_NAME, "image2video").remote(
        image=image, prompt=args.prompt, seed=args.seed, num_frames=args.num_frames)
    imageio.mimwrite(args.save_path, frames, fps=16, quality=8, macro_block_size=1)
    print(f"\nwrote {frames.shape[0]} frames -> {args.save_path}")


if __name__ == "__main__":
    main()
