"""Runs a REAL Cosmos-Predict2 output through the REAL Phi reconstruction
pipeline (SAM2 + DINOv2 + 3D reconstruction, on the `vitals-phi` Modal
app) -- the actual, quantitative answer to "is that near-static Cosmos
output a real finding or a call-parameter issue," rather than eyeballing
pixel differences (AGENT.md M6, 2026-08).

Renders the SAME prefix used for the Cosmos call, stitches on Cosmos's
own real continuation (resampled from its native 16fps/432x768 to this
project's 30fps/320x240 convention -- the same resample this adapter's
own generate_fn would apply), reconstructs the whole sequence through
real SAM2 tracking, then compares the reconstructed continuation against
the TRUE original rollout's own continuation (known, since this is a
real MuJoCo episode, not a black box) -- giving an actual position-error
number instead of a vibe.

    python3 scripts/run_cosmos_output_through_phi.py \\
        --scenario occlusion_corridor --seed 1 \\
        --continuation-path /tmp/cosmos_continuation.mp4 --n-continuation-frames 60
"""
import argparse
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="occlusion_corridor")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prefix-frames", type=int, default=30)
    parser.add_argument("--continuation-path", required=True, help="real generated continuation mp4 (any fps/resolution)")
    parser.add_argument("--n-continuation-frames", type=int, default=60,
                        help="target frame count at this project's 30fps convention -- default 60 (2.0s), "
                             "matching the true rollout's own remaining length for a direct comparison")
    args = parser.parse_args()

    import yaml
    from vitals.types import EpisodeSpec, Trajectory
    from vitals.physics import make_backend
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.adapters.base import prefix_of
    from vitals.adapters.cosmos import _read_video, _resample_to_target

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
    prefix_frames_obj, prefix_gt = renderer.render(conditioning, cameras=cfg["camera"])
    frame0_mask = (prefix_gt.segmentation[0] == 0)

    raw_continuation, raw_fps = _read_video(args.continuation_path)
    print(f"real Cosmos output: {raw_continuation.shape[0]} frames at {raw_fps}fps, "
          f"{raw_continuation.shape[1]}x{raw_continuation.shape[2]}")
    continuation = _resample_to_target(raw_continuation, raw_fps, spec.fps, args.n_continuation_frames,
                                        (prefix_frames_obj.rgb.shape[1], prefix_frames_obj.rgb.shape[2]))

    all_frames = np.concatenate([prefix_frames_obj.rgb, continuation], axis=0)
    print(f"stitched sequence: {all_frames.shape[0]} frames "
          f"({args.prefix_frames} real prefix + {args.n_continuation_frames} Cosmos continuation)")

    import modal
    f = modal.Function.from_name("vitals-phi", "run_phi_reconstruct_from_frames")
    print("calling real SAM2/DINOv2 Phi pipeline on Modal...")
    full_traj = f.remote(all_frames=all_frames, prefix_len=args.prefix_frames, scenario_name=args.scenario,
                         frame0_mask=frame0_mask, cam_pos=prefix_gt.cam_pos, cam_mat=prefix_gt.cam_mat,
                         fovy_deg=prefix_gt.fovy_deg, fps=spec.fps)

    n = full_traj.T - args.prefix_frames
    cont_pos = full_traj.pos[args.prefix_frames:]
    cont_present = full_traj.present[args.prefix_frames:]
    visible = cont_present[:, 0] & ~np.isnan(cont_pos[:, 0, 0])
    print(f"\nreconstructed continuation: {n} frames, {visible.sum()}/{n} visible/tracked")

    true_cont_pos = full.pos[args.prefix_frames:args.prefix_frames + n, 0]
    if visible.sum() > 0:
        err = np.linalg.norm(cont_pos[visible, 0] - true_cont_pos[visible], axis=-1)
        print(f"position error vs. TRUE original rollout continuation: mean={err.mean():.3f}m max={err.max():.3f}m")

    # How far the reconstructed (Cosmos-generated) ball actually moved,
    # vs. how far the TRUE ball moved over the same window -- the direct,
    # quantitative version of "did it just stay still."
    if visible.sum() >= 2:
        idx = np.where(visible)[0]
        cosmos_displacement = np.linalg.norm(cont_pos[idx[-1], 0] - cont_pos[idx[0], 0])
        true_displacement = np.linalg.norm(true_cont_pos[idx[-1]] - true_cont_pos[idx[0]])
        print(f"\nCosmos-reconstructed ball displacement over tracked window: {cosmos_displacement:.3f}m")
        print(f"TRUE ball displacement over the same window:                 {true_displacement:.3f}m")


if __name__ == "__main__":
    main()
