"""Modal app `vitals-orchestrate` -- runs `scripts/run_model_population.py`
ITSELF on Modal (2026-09), so a multi-hour population no longer depends
on a laptop process staying alive.

Why (found directly, not hypothetically): two n=50 wan populations were
killed mid-run by the local machine's own low-memory guard (the
orchestrating process used ~55MB but the machine had ~60MB free), each
time after ~26 minutes of real, already-billed GPU work whose results
then had nowhere to land. Wan is the worst case -- ~28 min per call, so
the local process must survive for hours with nothing to show until the
end -- but every population run had the same exposure. Cosmos runs only
"worked" because their ~90s calls landed results continuously.

What runs where, unchanged: the two GPU calls (generation on
`vitals-cosmos`/`vitals-wan`, reconstruction on `vitals-phi`) are still
separate deployed apps, called Modal-to-Modal via `Function.from_name`
-- the SAME closures `run_model_population.py` already builds
(`make_generate_fn`/`make_modal_phi_fn`), which just `import modal` at
call time; they work identically from inside a container. This app only
moves the ORCHESTRATOR (physics rollouts, MuJoCo prefix rendering,
threshold/R4 calibration, episode threading, trajectory caching, grid
video tiling, JSON writing) off the laptop. CPU-only -- no GPU is burned
for hours of orchestration.

Zero changes to run_model_population.py: the repo is mounted at
/root/vitals_repo, so the script's own `ROOT = parents[1]` resolves
there; `/root/vitals_repo/results` is a symlink to the `vitals-results`
Volume, so every path the script already writes (results/l0_demo_*.json,
results/trajectories/<scenario>_<model>/seed*.npz, results/videos/
*_grid.mp4) lands in the Volume with the script none the wiser. Pull
them down with `python3 remote/call_orchestrate.py pull`.

Headless rendering: MuJoCo via OSMesa (CPU software GL; MUJOCO_GL=osmesa)
-- never done on Modal in this project before this file, so
`smoke_render` below exists specifically to validate it with a real
render BEFORE any population is trusted to it (same "first real call is
a debugging pass" discipline every model integration here already went
through). Depth accuracy under OSMesa may differ from the laptop's own
GL, but nothing downstream reads prefix depth (reconstruction is
ray/plane on masks), so RGB + segmentation correctness is the check.

Detached execution: `remote/call_orchestrate.py submit` uses `.spawn()`,
not `.remote()` -- the FunctionCall keeps running server-side after the
client disconnects. That, not the relocation itself, is what removes the
laptop from the failure path.

Usage:
    modal deploy remote/modal_app_orchestrate.py
    python3 remote/call_orchestrate.py smoke                      # validate headless render FIRST
    python3 remote/call_orchestrate.py submit --model wan --scenario occlusion_corridor --seeds 1,2,3 --n-video-episodes 3
    python3 remote/call_orchestrate.py status <call_id>
    python3 remote/call_orchestrate.py pull                       # results/ <- vitals-results Volume
"""
import pathlib
import modal

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/vitals_repo"

app = modal.App("vitals-orchestrate")

# Every artifact run_model_population.py writes under results/ -- separate
# from every model's own weight-cache volume (different lifetime, different
# owner: outputs, not inputs).
results_volume = modal.Volume.from_name("vitals-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # OSMesa: CPU software OpenGL for headless MuJoCo rendering. libglew is
    # what mujoco's own render path links against; libgl1-mesa-dev for the
    # GL headers/loader OSMesa needs alongside it.
    .apt_install("libosmesa6-dev", "libgl1-mesa-dev", "libglew-dev", "ffmpeg")
    .pip_install(
        "numpy", "pyyaml", "pillow", "mujoco>=3.0", "imageio[ffmpeg]>=2.30",
        # runwayml only matters if an API model is orchestrated here; cheap,
        # and keeps the dispatch table fully importable.
        "runwayml>=5.20",
    )
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
    # The whole repo (scripts/, configs/, scenes/, vitals/), mounted at
    # container start (copy=False, same idiom as modal_app.py's own
    # add_local_dir) so local edits show up without an image rebuild.
    # results/ is excluded on purpose -- it's the Volume, symlinked in at
    # runtime; mounting a local copy would shadow it.
    .add_local_dir(str(REPO_DIR), REMOTE_REPO,
                   ignore=[".venv", "results", ".git", "__pycache__", "*.pyc", "*.npz", "*.mp4",
                           "*.pdf", "*.html", "MUJOCO_LOG.TXT"])
)


def _link_results():
    """/root/vitals_repo/results -> /results (the Volume). Idempotent."""
    import os
    link = pathlib.Path(REMOTE_REPO) / "results"
    if link.is_symlink() or link.exists():
        return
    os.symlink("/results", str(link))


@app.function(image=image, volumes={"/results": results_volume}, cpu=2, memory=4096, timeout=600)
def smoke_render():
    """Validates headless MuJoCo rendering on Modal with a REAL prefix
    render of occlusion_corridor: RGB shape/dtype and the segmentation
    convention (-1 background, 0 = the ball) exactly as the laptop's own
    renderer produces them. Run this before trusting any population to
    this app -- if OSMesa is misconfigured this is where it fails, cheaply."""
    import sys
    sys.path.insert(0, REMOTE_REPO)
    import numpy as np
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.adapters.base import prefix_of

    scene = f"{REMOTE_REPO}/scenes/occlusion_corridor.xml"
    spec = EpisodeSpec(name="occlusion_corridor", scene=scene, target_property="P2", band="I",
                       lam=6.0, n_reference=1, horizon_s=2.0, fps=30, perturb_mode="velocity_x_only")
    full = make_backend("mujoco", scene=scene)(spec, seed=1)
    conditioning = prefix_of(full, 10)
    frames_obj, gt = MujocoRenderer(scene, height=240, width=320).render(
        conditioning, cameras=sg.get("occlusion_corridor")["camera"])
    rgb = frames_obj.rgb
    seg_vals = sorted(int(v) for v in np.unique(gt.segmentation[0]))
    return dict(rgb_shape=list(rgb.shape), rgb_dtype=str(rgb.dtype),
                rgb_nonblank=bool(rgb.std() > 1.0), segmentation_values=seg_vals,
                ball_pixels_frame0=int((gt.segmentation[0] == 0).sum()))


@app.function(image=image, volumes={"/results": results_volume},
              # vitals-runway (2026-09): lets API models (runway_*) run through
              # this orchestrator -- the adapter reads RUNWAYML_API_SECRET from
              # the environment, same env var the SDK itself uses. Created via
              # `modal secret create vitals-runway RUNWAYML_API_SECRET=...`.
              secrets=[modal.Secret.from_name("vitals-runway")],
              cpu=4, memory=8192, timeout=86400)   # 24h, Modal's own ceiling -- a 50-episode wan run is ~3h+
def run_population(argv: list[str]) -> dict:
    """Runs run_model_population.main() with `argv` as its CLI args, inside
    this container, writing every artifact to the results Volume. Returns
    the parsed summary JSON so `status` can show it without a pull."""
    import json
    import os
    import sys
    _link_results()
    os.chdir(REMOTE_REPO)
    sys.path.insert(0, REMOTE_REPO)
    sys.path.insert(0, f"{REMOTE_REPO}/scripts")
    import run_model_population as rmp

    sys.argv = ["run_model_population.py", *argv]
    rmp.main()
    results_volume.commit()

    # The script names its output deterministically from --model/--scenario.
    args = rmp.argparse.ArgumentParser(add_help=False)
    args.add_argument("--model"); args.add_argument("--scenario", default="occlusion_corridor")
    known, _ = args.parse_known_args(argv)
    out = pathlib.Path("/results") / f"l0_demo_{known.scenario}_{known.model}.json"
    summary = json.load(open(out)) if out.exists() else {"note": "no results file written (every episode failed?)"}
    summary["_artifact"] = str(out.name)
    return summary
