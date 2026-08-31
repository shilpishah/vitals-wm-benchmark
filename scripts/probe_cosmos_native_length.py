"""One-off diagnostic (2026-08): Cosmos's own module docstring flags this
as genuinely UNRESOLVED -- "Cosmos's own docs do not specify how many
frames one video2world call actually returns... do not trust the padding
behavior as a real generative capability." Every real-model population
run so far has silently trusted `resample_to_target`'s pad-to-length
behavior without ever checking whether Cosmos's raw output is LONGER or
SHORTER than what's been requested (60 frames, 2s, in every run to date).

This calls the SAME `vitals-cosmos` Modal app used everywhere else, but
reads the RAW output (`_read_video`, before any resample/pad step) so the
true native (frame_count, fps) can be reported directly, once, cheaply
(ONE real GPU call) -- not assumed, not guessed from padding artifacts
after the fact.

    python3 scripts/probe_cosmos_native_length.py --scenario occlusion_corridor
"""
import argparse
import pathlib
import sys
import tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="occlusion_corridor")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import yaml
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.adapters.base import prefix_of
    from vitals.adapters.video_utils import _write_video, _read_video
    from vitals.adapters.scenario_prompts import SCENARIO_PROMPTS
    from vitals.adapters.cosmos import COSMOS_NATIVE_FPS, COSMOS_RESOLUTION

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((ROOT / f"configs/manifests/{args.scenario}.yaml").read_text())
    spec = EpisodeSpec(name=manifest["name"], scene=manifest["scene"], target_property=manifest["target_property"],
                        band=manifest["band"], lam=manifest["lam"], n_reference=manifest["n_reference"],
                        horizon_s=manifest["horizon_s"], fps=manifest["fps"], seed=manifest.get("seed", 0),
                        perturb_mode=manifest.get("perturb_mode", "full"))
    rollout = make_backend("mujoco", scene=spec.scene)
    full = rollout(spec, seed=args.seed)
    conditioning = prefix_of(full, 30)

    cfg = sg.get(args.scenario)
    renderer = MujocoRenderer(spec.scene, height=240, width=320)
    prefix_frames_obj, _ = renderer.render(conditioning, cameras=cfg["camera"])

    with tempfile.TemporaryDirectory() as tmp:
        in_path = pathlib.Path(tmp) / "prefix.mp4"
        _write_video(prefix_frames_obj.rgb, in_path, fps=30)
        prefix_bytes = in_path.read_bytes()

    print(f"calling real vitals-cosmos/video2world directly (one real GPU call, bypassing resample_to_target "
          f"entirely so the RAW native output length is visible)...")
    import modal
    f = modal.Function.from_name("vitals-cosmos", "video2world")
    out_bytes = f.remote(prefix_video_bytes=prefix_bytes, prompt=SCENARIO_PROMPTS[args.scenario],
                         num_conditional_frames=5, fps=COSMOS_NATIVE_FPS, resolution=COSMOS_RESOLUTION, seed=0)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = pathlib.Path(tmp) / "continuation.mp4"
        out_path.write_bytes(out_bytes)
        raw_frames, raw_fps = _read_video(out_path)

    print(f"\nRAW native output: {raw_frames.shape[0]} frames @ {raw_fps}fps = {raw_frames.shape[0]/raw_fps:.3f}s")
    print(f"(requested fps was {COSMOS_NATIVE_FPS}; every prior real-model run has requested only 60 target "
          f"frames (2.0s) at 30fps downstream of this call -- compare against that raw duration above to see "
          f"how much of this call's real output was being discarded, or whether it's already SHORTER than "
          f"2.0s, in which case resample_to_target has been silently PADDING every Cosmos episode so far.)")


if __name__ == "__main__":
    main()
