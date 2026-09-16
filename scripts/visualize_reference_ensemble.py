"""Render a scenario's reference ensemble so you can actually see what
lambda-scale perturbation does, and what a detected defect looks like
against it.

Nothing about the scoring pipeline depends on this -- it's a dev/inspection
tool, not part of the measurement path. Manifest-driven, same as
scripts/run_l0_demo.py: each manifest is a fully isolated scenario
(AGENT.md 3.6), so this never combines two scenes into one render. Writes
four things to --out:

  breadcrumbs.png   static scene + every reference trajectory traced as
                     small markers (color = time), plus one mutant traced
                     in red for contrast
  ensemble.mp4       all M references rolling simultaneously (as marker
                     spheres, not the model's own ball body) + the mutant
                     in red -- watch the whole band diverge live
  rollout_single.mp4  one nominal rollout, real physics, real geom
  rollouts/           one MP4 per reference, each its own scene
  ensemble.json       t, x, z for every reference + the mutant, for plotting

    python scripts/visualize_reference_ensemble.py --out /path/to/dir
    python scripts/visualize_reference_ensemble.py --out /path/to/dir \\
        --manifest configs/manifests/occlusion_corridor.yaml
"""
import sys, pathlib, argparse, json, subprocess, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import mujoco
from vitals.types import EpisodeSpec
from vitals.physics import make_backend
from vitals.mutants import library as mut

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Per-scenario camera framing -- the two scenes occupy very different regions
# of world space (ramp_descent is compact around the origin; occlusion_corridor
# runs from x=0 out past the wall at x=5), so one fixed camera can't frame both.
CAMERAS = {
    "ramp_descent":       dict(lookat=[0.5, 0.0, 0.6], distance=8.5, azimuth=-90, elevation=-12),
    "occlusion_corridor": dict(lookat=[3.0, 0.0, 0.3], distance=9.5, azimuth=-90, elevation=-10),
}

# Demonstrative mutant per scenario, for breadcrumbs.png / ensemble.mp4's red
# trace -- wrong_gravity only means something where there's vertical motion
# to corrupt (ramp_descent); occlusion_corridor gets `duplicate` instead,
# still R1 but "erroneous extra object" rather than "vanished".
DEMO_MUTANT = {
    "ramp_descent":       lambda base, horizon_s: mut.wrong_gravity(base, factor=0.6),
    "occlusion_corridor": lambda base, horizon_s: mut.duplicate(base, t_star=horizon_s * 0.25),
}


def _make_camera(cam_kwargs):
    cam = mujoco.MjvCamera()
    cam.lookat = cam_kwargs["lookat"]
    cam.distance = cam_kwargs["distance"]
    cam.azimuth = cam_kwargs["azimuth"]
    cam.elevation = cam_kwargs["elevation"]
    return cam


def render_breadcrumbs(scene_path, cam_kwargs, refs, mutant_traj, out_path, every=4):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = _make_camera(cam_kwargs)

    renderer.update_scene(data, camera=cam)
    scene = renderer.scene

    def add_marker(pos, rgba, size=0.035):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                             np.array([size, 0, 0]), np.array(pos),
                             np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        scene.ngeom += 1

    T = refs[0].T
    for r in refs:
        for i in range(0, T, every):
            frac = i / T
            add_marker(r.pos[i, 0], [0.15 + 0.5 * frac, 0.35, 0.85 - 0.5 * frac, 0.55], size=0.03)

    for i in range(0, mutant_traj.T, every):
        pos = mutant_traj.pos[i, 0]
        if np.any(np.isnan(pos)):
            continue
        add_marker(pos, [0.95, 0.1, 0.1, 0.85], size=0.045)

    pixels = renderer.render()
    _save_png(pixels, out_path)
    renderer.close()


def _save_png(pixels, path):
    from PIL import Image
    Image.fromarray(pixels).save(path)


def render_rollout_video(scene_path, cam_kwargs, traj, out_path, fps=30):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=854)
    cam = _make_camera(cam_kwargs)

    tmp_dir = out_path.parent / "_frames_tmp"
    tmp_dir.mkdir(exist_ok=True)
    for i in range(traj.T):
        data.qpos[:3] = traj.pos[i, 0]
        data.qpos[3:7] = traj.quat[i, 0]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        _save_png(renderer.render(), tmp_dir / f"f{i:04d}.png")
    renderer.close()

    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / "f%04d.png"),
        "-pix_fmt", "yuv420p", "-loglevel", "error", str(out_path)
    ], check=True)
    shutil.rmtree(tmp_dir)


def render_ensemble_video(scene_path, cam_kwargs, refs, mutant_traj, out_path, fps=30):
    """All M references + the mutant, animated together in one video. None
    of them is the model's own 'ball' body -- that gets parked off-scene so
    every reference (and the mutant) is drawn identically, as a marker
    sphere, and there's no visually-privileged "the" rollout among them."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0, 0, -50]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=854)
    cam = _make_camera(cam_kwargs)

    def add_marker(scene, pos, rgba, size):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                             np.array([size, 0, 0]), np.array(pos),
                             np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        scene.ngeom += 1

    T = refs[0].T
    tmp_dir = out_path.parent / "_frames_tmp_ens"
    tmp_dir.mkdir(exist_ok=True)

    for i in range(T):
        renderer.update_scene(data, camera=cam)
        scene = renderer.scene
        for r in refs:
            add_marker(scene, r.pos[i, 0], [0.16, 0.42, 0.85, 0.9], size=0.15)
        mpos = mutant_traj.pos[i, 0]
        if not np.any(np.isnan(mpos)):
            add_marker(scene, mpos, [0.9, 0.13, 0.1, 0.95], size=0.17)
        _save_png(renderer.render(), tmp_dir / f"f{i:04d}.png")

    renderer.close()
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / "f%04d.png"),
        "-pix_fmt", "yuv420p", "-loglevel", "error", str(out_path)
    ], check=True)
    shutil.rmtree(tmp_dir)


def render_individual_rollouts(scene_path, cam_kwargs, refs, seeds, out_dir, fps=30):
    """One MP4 per reference, each its own file/scene -- not overlaid with
    the others. Model+renderer are created once and reused across all M
    rollouts purely for speed; each rollout's frames are fully independent."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=854)
    cam = _make_camera(cam_kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_frames_tmp"

    for idx, traj in enumerate(refs):
        tmp_dir.mkdir(exist_ok=True)
        for i in range(traj.T):
            data.qpos[:3] = traj.pos[i, 0]
            data.qpos[3:7] = traj.quat[i, 0]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            _save_png(renderer.render(), tmp_dir / f"f{i:04d}.png")

        out_path = out_dir / f"ref_{idx:02d}_seed{seeds[idx]}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / "f%04d.png"),
            "-pix_fmt", "yuv420p", "-loglevel", "error", str(out_path)
        ], check=True)
        shutil.rmtree(tmp_dir)
        print(f"  [{idx + 1}/{len(refs)}] {out_path.name}")

    renderer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="configs/manifests/ramp_descent.yaml")
    ap.add_argument("--M", type=int, default=None, help="overrides manifest n_reference")
    ap.add_argument("--lam", type=float, default=None, help="overrides manifest lam")
    args = ap.parse_args()

    manifest = yaml.safe_load((ROOT / args.manifest).read_text())
    scene_path = str(ROOT / manifest["scene"])
    name = manifest["name"]
    cam_kwargs = CAMERAS.get(name, CAMERAS["ramp_descent"])

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    M = args.M or manifest["n_reference"]
    lam = args.lam if args.lam is not None else manifest["lam"]

    spec = EpisodeSpec(name=name, scene=manifest["scene"],
                        target_property=manifest["target_property"], band=manifest["band"],
                        lam=lam, n_reference=M, horizon_s=manifest["horizon_s"], fps=30,
                        perturb_mode=manifest.get("perturb_mode", "full"))
    roll = make_backend("mujoco", scene=scene_path)
    seeds = [1000 + i for i in range(M)]
    refs = [roll(spec, s) for s in seeds]

    m = DEMO_MUTANT.get(name, DEMO_MUTANT["ramp_descent"])(refs[0], spec.horizon_s)

    print(f"scenario={name}  M={M}  lambda={lam}")

    print("rendering breadcrumbs...")
    render_breadcrumbs(scene_path, cam_kwargs, refs, m.traj, out / "breadcrumbs.png")

    print(f"rendering ensemble video ({M} rollouts + 1 mutant)...")
    render_ensemble_video(scene_path, cam_kwargs, refs, m.traj, out / "ensemble.mp4")

    print("rendering single rollout video...")
    render_rollout_video(scene_path, cam_kwargs, refs[0], out / "rollout_single.mp4")

    print(f"rendering {M} individual rollout videos into rollouts/...")
    render_individual_rollouts(scene_path, cam_kwargs, refs, seeds, out / "rollouts")

    print("dumping trajectory data...")
    data = {
        "t": refs[0].t.tolist(),
        "refs": [{"x": r.pos[:, 0, 0].tolist(), "z": r.pos[:, 0, 2].tolist()} for r in refs],
        "mutant": {"name": m.name, "t_star": m.t_star,
                   "x": m.traj.pos[:, 0, 0].tolist(), "z": m.traj.pos[:, 0, 2].tolist()},
    }
    (out / "ensemble.json").write_text(json.dumps(data))

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
