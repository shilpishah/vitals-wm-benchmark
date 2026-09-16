"""One real, paid Runway API call through `vitals/adapters/runway.py` --
the API-model equivalent of `remote/call_hunyuan.py download_checkpoints`:
validate the whole path (secret -> SDK auth -> task submit -> poll ->
download -> decode -> resample to exactly n_frames) with a SINGLE call
before any population is trusted to it. Deliberately its own app, not a
function on `vitals-orchestrate`: attaching the secret there requires a
redeploy, and this was first run while a multi-hour population was live
in an orchestrator container (redeploying risks killing it).

Renders the conditioning prefix LOCALLY (MuJoCo on the dev machine, the
same renderer/camera `run_model_population.py` uses) and ships the
frames to a CPU-only container that holds the secret -- the key never
needs to exist in a local shell at all.

    modal run remote/modal_app_runway_smoke.py                 # default: gen4.5, collision, 60 frames
    modal run remote/modal_app_runway_smoke.py --model gen4_turbo
"""
import pathlib
import modal

VITALS_DIR = pathlib.Path(__file__).resolve().parent.parent / "vitals"

app = modal.App("vitals-runway-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy", "pillow", "pyyaml", "imageio[ffmpeg]>=2.30", "runwayml>=5.20")
    .add_local_dir(str(VITALS_DIR), "/root/vitals_pkg/vitals")
)


@app.function(image=image, secrets=[modal.Secret.from_name("vitals-runway")], timeout=900)
def one_call(prefix_frames, scenario: str, model: str, n_frames: int) -> dict:
    import sys, time
    sys.path.insert(0, "/root/vitals_pkg")
    import numpy as np
    from vitals.adapters.runway import make_runway_generate_fn

    logs = []
    # A registry key (`runway_veo3.1`) resolves to that entry's own model
    # id / ratio / allowed durations -- the gateway models reject gen4.5's
    # 1104:832 ratio and 2-10s integer durations (2026-09-14). A bare id
    # (`gen4.5`) keeps the old defaults.
    from vitals.adapters import MODEL_REGISTRY
    meta = MODEL_REGISTRY.get(model, {})
    kw = {}
    if meta.get("provider") == "runway":
        model = meta["model_id"]
        kw = dict(ratio=meta.get("ratio", "1104:832"), allowed_durations=meta.get("durations"),
                  supports_seed=meta.get("supports_seed", True))
    gen = make_runway_generate_fn(scenario, model=model, log=logs.append, **kw)
    t0 = time.time()
    out = gen(np.asarray(prefix_frames), n_frames)
    elapsed = time.time() - t0
    out = np.asarray(out)
    motion = float(np.abs(out[1:].astype(np.int16) - out[:-1].astype(np.int16)).mean()) if out.shape[0] > 1 else 0.0
    return dict(model=model, scenario=scenario, shape=list(out.shape), dtype=str(out.dtype),
                elapsed_s=round(elapsed, 1), nonblank=bool(out.std() > 1.0),
                mean_abs_frame_diff=round(motion, 3), log="\n".join(logs))


@app.function(image=image, secrets=[modal.Secret.from_name("vitals-runway")], timeout=120)
def credit_balance() -> dict:
    """FREE call (no generation): the organization's current credit
    balance, so a population is never submitted into an account that
    cannot pay for it -- the earlier runway calls stopped on insufficient
    credits mid-validation (2026-09). Fields are whatever the SDK returns
    (`credit_balance` on runwayml 5.x); passed through, not interpreted."""
    from runwayml import RunwayML
    org = RunwayML().organization.retrieve()
    d = org.model_dump() if hasattr(org, "model_dump") else dict(org)
    # Identity fields (id/name/...) are returned too, so a zero balance can
    # be told apart from "the stored key belongs to a different org than
    # the one that was funded" (2026-09-12). Never the key itself.
    keep = {k: v for k, v in d.items() if any(s in k.lower() for s in ("credit", "tier", "usage", "id", "name", "org"))}
    keep["_all_fields"] = sorted(d.keys())
    return keep


@app.local_entrypoint()
def credits():
    """modal run remote/modal_app_runway_smoke.py::credits"""
    for k, v in credit_balance.remote().items():
        print(f"{k}: {v}")


@app.local_entrypoint()
def main(model: str = "gen4.5", scenario: str = "collision", n_frames: int = 60):
    import sys
    sys.path.insert(0, str(VITALS_DIR.parent))
    import yaml
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.adapters.base import prefix_of

    m = yaml.safe_load(open(VITALS_DIR.parent / f"configs/manifests/{scenario}.yaml"))
    spec = EpisodeSpec(name=m["name"], scene=str(VITALS_DIR.parent / m["scene"]), target_property=m["target_property"],
                       band=m["band"], lam=m["lam"], n_reference=1, horizon_s=m["horizon_s"], fps=m["fps"],
                       perturb_mode=m.get("perturb_mode", "full"))
    full = make_backend("mujoco", scene=spec.scene)(spec, seed=1)
    prefix = prefix_of(full, 30)
    frames_obj, _ = MujocoRenderer(spec.scene, height=240, width=320).render(prefix, cameras=sg.get(scenario)["camera"])
    print(f"rendered prefix {frames_obj.rgb.shape} locally; calling runway:{model} for {n_frames} frames ...")
    result = one_call.remote(frames_obj.rgb, scenario, model, n_frames)
    for k, v in result.items():
        print(f"{k}: {v}")
