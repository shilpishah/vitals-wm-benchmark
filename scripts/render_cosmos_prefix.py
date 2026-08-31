"""Renders a real conditioning prefix from an actual scenario rollout and
saves it as an mp4 -- the input `video2world` (Cosmos-Predict2, AGENT.md
M6) actually needs. Not a one-off snippet: this is the same prefix a real
`VideoWorldModel.predict()` call would render internally
(`vitals/adapters/video_model.py`), pulled out standalone so it can be
inspected/reused directly against `remote/call_cosmos.py video2world`
without running the full adapter.

    python3 scripts/render_cosmos_prefix.py --scenario occlusion_corridor --seed 1 --save-path /tmp/prefix.mp4
"""
import argparse
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="occlusion_corridor")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prefix-frames", type=int, default=30, help="1.0s at 30fps -- the frozen protocol default")
    parser.add_argument("--save-path", required=True)
    args = parser.parse_args()

    import yaml
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.adapters.base import prefix_of
    from vitals.adapters.cosmos import _write_video

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((ROOT / f"configs/manifests/{args.scenario}.yaml").read_text())
    spec = EpisodeSpec(name=manifest["name"], scene=manifest["scene"], target_property=manifest["target_property"],
                        band=manifest["band"], lam=manifest["lam"], n_reference=manifest["n_reference"],
                        horizon_s=manifest["horizon_s"], fps=manifest["fps"], seed=manifest.get("seed", 0),
                        perturb_mode=manifest.get("perturb_mode", "full"))
    rollout = make_backend("mujoco", scene=spec.scene)
    full = rollout(spec, seed=args.seed)
    conditioning = prefix_of(full, args.prefix_frames)

    cfg = sg.get(args.scenario)
    renderer = MujocoRenderer(spec.scene, height=240, width=320)
    frames_obj, _ = renderer.render(conditioning, cameras=cfg["camera"])

    _write_video(frames_obj.rgb, args.save_path, fps=spec.fps)
    print(f"wrote {frames_obj.rgb.shape[0]} frames ({frames_obj.rgb.shape[0]/spec.fps:.2f}s at {spec.fps}fps) "
          f"-> {args.save_path}")


if __name__ == "__main__":
    main()
