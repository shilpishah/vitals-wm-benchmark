"""Modal app for VITALS' Phi pipeline (SAM2 + DINOv2 GPU inference).

Replaces the earlier CMU GHC lab-machine SSH workflow (AGENT.md's remote-
GPU history), which hit real, un-fixable infrastructure limits on that
specific shared machine: a 7.77GB RTX 2080 (Turing -- confirmed
incompatible with `expandable_segments:True`, a standard PyTorch
fragmentation mitigation), a hard 16GB *virtual*-memory ulimit imposed by
shared-machine policy (not raisable, not GPU-related), and ephemeral
/tmp + /var/tmp wiped on every reboot (sometimes mid-session), which
repeatedly cost a full environment rebuild.

Modal fixes all three at once: a dedicated container per invocation (no
shared-tenancy ulimits), a GPU tier chosen per job (`GPU_TYPE` below, not
whatever one machine happens to have), and a persistent Volume for model
weights + package code that survives indefinitely instead of getting wiped.

Local MuJoCo physics/rendering (cheap, CPU-only) stays local, matching the
same split already used with the SSH workflow -- only the GPU-heavy Phi
inference (SAM2 tracking, DINOv2 embedding) runs remotely. See
scripts/ or the phi/ package itself for what actually runs where.

This file defines the app/image/functions only -- it deliberately has no
`@app.local_entrypoint()` and should not be invoked via `modal run`. Every
`modal run remote/modal_app.py::<fn>` spins up a brand new EPHEMERAL app
that tears down when the call finishes -- fine for the first exploratory
smoke test, but it means every debugging iteration leaves behind a new,
already-dead "vitals-phi" entry in the midcentury-labs dashboard (this is
`modal run`'s actual documented behavior, not a bug -- confirmed directly:
the workspace accumulated 8 of these during defect #16's investigation).

Deploy once (redeploy only when the image itself changes -- new pip
dependency, etc.; vitals/phi/*.py is mounted at container startup, so
ordinary code edits need no redeploy):
    modal deploy remote/modal_app.py

Then invoke through remote/call.py, which looks up and calls into the ONE
persistent deployed app instead of creating a new ephemeral one per call:
    python3 remote/call.py smoke_test
    python3 remote/call.py upload_episode --local-dir /path/to/full_episode --name occlusion_corridor_seed2024
    python3 remote/call.py run_reidentification --name occlusion_corridor_seed2024 --debug
"""
import pathlib
import modal

GPU_TYPE = "A10G"   # 24GB -- see AGENT.md for why this replaces the RTX 2080
VITALS_DIR = pathlib.Path(__file__).resolve().parent.parent / "vitals"

app = modal.App("vitals-phi")

# Persistent across invocations -- this is what the old workflow never had.
# /cache/hf, /cache/torch: model weight downloads (SAM2 checkpoint, DINOv2).
# /cache/episodes/<name>/: uploaded frames + GT masks for a validation episode.
model_cache = modal.Volume.from_name("vitals-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")   # debian_slim has no git by default; needed for the pip+git install below
    .pip_install(
        "numpy",
        "pillow",
        "huggingface_hub",
    )
    # Official SAM2 (facebookresearch/sam2), not the suspicious third-party
    # PyPI "sam2" package flagged earlier this project -- verify the source
    # again if this line is ever changed.
    .pip_install("git+https://github.com/facebookresearch/sam2.git")
    # SAM2's own dependency spec pulls in an unpinned torch/torchvision pair
    # (measured directly on the old machine: this alone broke CUDA
    # compatibility there). Re-pin explicitly, last, as the final image
    # layer, same fix as the old remote_setup.sh -- except here it's baked
    # into the image ONCE at build time, not re-run (and re-risked) every
    # session.
    .pip_install("torch==2.6.0", "torchvision==0.21.0")
    .env({"HF_HOME": "/cache/hf", "TORCH_HOME": "/cache/torch"})
    # Mounted at container startup (copy=False, the default), not baked into
    # the image -- local edits to vitals/phi/*.py show up on the next
    # `modal run` with no image rebuild, matching the fast-iteration
    # workflow this project has actually been using.
    .add_local_dir(str(VITALS_DIR), "/root/vitals_pkg/vitals")
)


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": model_cache}, timeout=600)
def smoke_test():
    """Confirms the image + GPU + model downloads all actually work,
    before anything from vitals/phi/ gets mounted and run against it --
    same "validate the instrument before trusting it" discipline as the
    rest of this project."""
    import torch
    print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    mask_gen = SAM2AutomaticMaskGenerator.from_pretrained(
        "facebook/sam2.1-hiera-tiny", device="cuda", points_per_side=8, points_per_batch=8)
    print("SAM2 mask generator loaded OK")

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino = dino.to("cuda").eval()
    print("DINOv2 loaded OK")

    model_cache.commit()  # persist the downloaded weights for next time
    return "SETUP_DONE"


# Reference-ensemble deceleration for occlusion_corridor -- was its own
# local copy here (1.246, from detect/physical_constants.py::
# fit_corridor_deceleration, CV=0.12%), now imported from vitals.phi.
# scene_geometry (sg.OCCLUSION_CORRIDOR_DECELERATION below) instead of
# redefined -- the SAME consolidation this module already did once for
# OCCLUSION_CORRIDOR_WALL_BOUNDS (this file's own comment on that constant
# tells that exact story), applied a second time after this constant's own
# un-migrated local copy caused a real, confirmed regression in GATE 2's
# velocity_freeze accuracy once `run_gate2_episode` was unified onto the
# shared `track_and_reconstruct` path and started reading `scene_geometry`
# directly instead of this local value (AGENT.md M2.5).

# The wall's own empirical visual occlusion span (world-x) -- NOT its own
# box half-extent, and NOT redefined here anymore (2026-08 consolidation,
# AGENT.md M6): this used to be its own local copy, `(4.39, 5.87)`, and it
# had already drifted from the naive `(4.25, 5.75)` value still quoted as
# an example in `motion_prior.py`'s own docstring -- see `vitals/phi/
# scene_geometry.py`'s own `OCCLUSION_CORRIDOR_WALL_BOUNDS` for the full
# derivation (5mm-resolution visibility sweep) and why the two values
# differ. Imported inside each function body below, matching this file's
# own established per-function `sys.path.insert` + import pattern (module
# top-level here runs LOCALLY at deploy time, not remotely, so vitals
# imports are deliberately kept out of it throughout this file).


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": model_cache}, timeout=1200)
def run_reidentification(name: str, similarity_threshold: float = 0.4, forgiveness_frames: int = 1,
                          debug: bool = False, use_metric_prior: bool = True):
    """End-to-end organic validation of track_with_reidentification -- the
    same test as AGENT.md's build-sequence step 27, now on infrastructure
    that doesn't OOM. Real SAM2 video tracking from frame 0 on the uploaded
    episode, hitting the physics-prior search codepath naturally when SAM2
    loses the object; IoU measured against ground truth throughout,
    including post-reacquisition.

    use_metric_prior=True (default) exercises defects #16 and #17's fixes
    -- reads camera calibration from the episode's own meta.json (added
    when defect #16's fix needed it) and passes it plus
    sg.OCCLUSION_CORRIDOR_DECELERATION and OCCLUSION_CORRIDOR_WALL_BOUNDS
    through to track_with_reidentification's metric-anchored path. Set
    False to reproduce the OLD pixel-only behavior for comparison."""
    import json
    import functools
    import os
    import numpy as np
    from PIL import Image
    import sys
    sys.path.insert(0, "/root/vitals_pkg")
    from vitals.phi import segmentation as seg
    from vitals.phi import reidentify as reid
    from vitals.phi import scene_geometry as sg

    if debug:
        os.environ["VITALS_REID_DEBUG"] = "1"

    episode_dir = pathlib.Path(f"/cache/episodes/{name}")
    meta = json.load(open(episode_dir / "meta.json"))
    gt_masks = np.load(episode_dir / "gt_masks.npy")
    T = meta["T"]
    frame_paths = [str(episode_dir / "frames" / f"{i:05d}.jpg") for i in range(T)]
    print(f"episode: T={T}  drop={meta['drop']}  rise={meta['rise']}")

    camera = None
    deceleration = None
    occluder_bounds = None
    if use_metric_prior:
        if "cam_pos" not in meta:
            raise ValueError(f"episode '{name}' has no camera calibration in meta.json -- "
                              f"re-generate it (gen_full_episode.py) or pass use_metric_prior=False")
        camera = dict(cam_pos=np.array(meta["cam_pos"]), cam_mat=np.array(meta["cam_mat"]),
                      fovy_deg=meta["fovy_deg"], width=meta["width"], height=meta["height"],
                      plane_z=meta["plane_z"])
        deceleration = sg.OCCLUSION_CORRIDOR_DECELERATION
        occluder_bounds = sg.OCCLUSION_CORRIDOR_WALL_BOUNDS

    predictor = seg.load_predictor(device="cuda")
    dino = reid.load_dino(device="cuda")
    mask_gen_factory = functools.partial(reid.load_mask_generator, device="cuda",
                                          points_per_side=32, points_per_batch=32)

    result = reid.track_with_reidentification(
        predictor, dino, mask_gen_factory, frame_paths, gt_masks[0],
        similarity_threshold=similarity_threshold, forgiveness_frames=forgiveness_frames,
        device="cuda", dt=1.0 / 30, camera=camera, deceleration=deceleration,
        occluder_bounds=occluder_bounds)

    def iou(a, b):
        inter = (a & b).sum()
        union = (a | b).sum()
        return float(inter) / float(union) if union > 0 else (1.0 if a.sum() == b.sum() == 0 else 0.0)

    ious = np.array([iou(result.masks.get(i, np.zeros_like(gt_masks[i])), gt_masks[i]) for i in range(T)])
    drop_frame = meta["drop"][0] if meta["drop"] else T
    rise_frame = meta["rise"][0] if meta["rise"] else T

    summary = dict(
        n_reid_events=len(result.reid_events),
        reid_events=[(int(fi), float(s) if s is not None else None, bool(m), me)
                     for fi, s, m, me in result.reid_events],
        mean_iou_all=float(ious.mean()),
        mean_iou_pre_occlusion=float(ious[:drop_frame].mean()) if drop_frame > 0 else None,
        mean_iou_post_reacquisition=float(ious[rise_frame + 5:].mean()) if rise_frame + 5 < T else None,
    )
    print(json.dumps(summary, indent=2))
    model_cache.commit()
    return summary


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": model_cache}, timeout=1200)
def run_gate2_episode(name: str, similarity_threshold: float = 0.4, forgiveness_frames: int = 1,
                       use_metric_prior: bool = True, debug_recon: bool = False):
    """GATE 2 (AGENT.md M5, the instrument tax): identical to
    `run_reidentification`'s real SAM2/DINO tracking, but additionally
    reconstructs a real `vitals.types.Trajectory` via `phi/reconstruct.py`
    -- the actual L1 (pixel-measured) counterpart to the mutant's own L0
    (ground-truth-state) Trajectory. Returns that reconstructed trajectory
    as plain arrays (not privileged state -- assembled purely from tracked
    masks/reid_events, same as any other Phi output) so the caller can run
    the SAME sigma_k/threshold/event-extraction code GATE 1 already used in
    state space, on this pixel-measured input instead -- the event-time
    delta between the two IS the instrument tax.

    Ground-truth metric position is deliberately NOT read or returned here
    -- this function only ever touches gt_masks.npy (for IoU and the
    occlusion-flag accuracy check below, both pixel-space comparisons
    against what Phi *itself* could have gated on), never the mutant's
    true 3D trajectory. Position-error / ID-switch / event-time-bias
    metrics that need the true trajectory are computed by the caller
    locally, which already has it (it built the mutant).

    Track+reconstruct itself now goes through `vitals.adapters.video_
    model.track_and_reconstruct` -- the SAME scene-mode-aware function
    `default_phi_reconstruct` (real-model scoring) uses (2026-08, real
    drift fix: this function used to have its OWN separate copy of this
    logic, hardcoded to occlusion_corridor's own `plane_z`/`OCCLUSION_
    CORRIDOR_DECELERATION`/`OCCLUSION_CORRIDOR_WALL_BOUNDS`, silently
    incompatible with a `plane_pieces` scenario -- see that function's
    own docstring). `meta.json` now carries `scenario_name` (written by
    `scripts/run_gate2.py`'s own `render_episode()`) so this function can
    look up the right config itself, the same way every other Phi caller
    already does, rather than being handed occlusion_corridor-shaped
    kwargs directly."""
    import json
    import os
    import numpy as np
    import sys
    sys.path.insert(0, "/root/vitals_pkg")
    from vitals.adapters.video_model import track_and_reconstruct

    if debug_recon:
        # 2026-08, same convention as run_reidentification's own `debug`
        # -> VITALS_REID_DEBUG -- gates fit_ballistic_trajectory's own
        # normalized-cost printout (vitals/phi/reconstruct.py), added
        # while diagnosing GATE 2's own 0/42 ballistic-reconstruction
        # success rate on projectile.
        os.environ["VITALS_RECON_DEBUG"] = "1"

    episode_dir = pathlib.Path(f"/cache/episodes/{name}")
    meta = json.load(open(episode_dir / "meta.json"))
    gt_masks = np.load(episode_dir / "gt_masks.npy")
    T = meta["T"]
    frame_paths = [str(episode_dir / "frames" / f"{i:05d}.jpg") for i in range(T)]

    # secondary_gt_masks.npy (2026-08, the occlusion_corridor_moving
    # wiring fix): only present for a scenario that registered
    # `occluder_geometry` (scripts/run_gate2.py's own render_episode) --
    # absent (None) for every other scenario, a strict no-op matching
    # track_and_reconstruct's own default.
    secondary_path = episode_dir / "secondary_gt_masks.npy"
    frame0_mask_secondary = np.load(secondary_path)[0] if secondary_path.exists() else None

    traj, result = track_and_reconstruct(
        frame_paths, gt_masks[0], meta["scenario_name"],
        cam_pos=np.array(meta["cam_pos"]), cam_mat=np.array(meta["cam_mat"]), fovy_deg=meta["fovy_deg"],
        width=meta["width"], height=meta["height"], fps=30,
        similarity_threshold=similarity_threshold, forgiveness_frames=forgiveness_frames,
        device="cuda", use_metric_prior=use_metric_prior,
        frame0_mask_secondary=frame0_mask_secondary)

    def iou(a, b):
        inter = (a & b).sum()
        union = (a | b).sum()
        return float(inter) / float(union) if union > 0 else (1.0 if a.sum() == b.sum() == 0 else 0.0)

    ious = np.array([iou(result.masks.get(i, np.zeros_like(gt_masks[i])), gt_masks[i]) for i in range(T)])

    # occlusion-flag accuracy: did Phi's own raw per-frame visibility (mask
    # non-empty) agree with ground truth per-frame visibility? Pixel-space
    # only, same status as the IoU check above -- NOT the same thing as
    # `present` in the reconstructed Trajectory below, which additionally
    # incorporates re-identification's gap-closed verdict (module docstring,
    # phi/reconstruct.py).
    phi_visible = np.array([bool(result.masks.get(i) is not None and result.masks[i].sum() > 0)
                             for i in range(T)])
    gt_visible = np.array([bool(gt_masks[i].sum() > 0) for i in range(T)])
    occlusion_flag_accuracy = float((phi_visible == gt_visible).mean())

    summary = dict(
        n_reid_events=len(result.reid_events),
        reid_events=[(int(fi), float(s) if s is not None else None, bool(m), me)
                     for fi, s, m, me in result.reid_events],
        mean_iou_all=float(ious.mean()),
        occlusion_flag_accuracy=occlusion_flag_accuracy,
        traj_t=traj.t.tolist(),
        traj_pos=np.where(np.isnan(traj.pos), None, traj.pos).tolist(),
        traj_present=traj.present.tolist(),
    )
    model_cache.commit()
    return summary


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": model_cache}, timeout=1200)
def run_multiobject_episode(name: str, similarity_threshold: float = 0.4, forgiveness_frames: int = 1,
                             use_metric_prior: bool = True):
    """AGENT.md M5.6 (scoped phase-2 multi-object support): real end-to-end
    validation of track_with_reidentification's frame0_mask_secondary path
    -- tracks a SECOND object (SAM2 obj_id=2, no search-and-reprompt of its
    own) alongside the primary object's full machinery, then reconstructs
    BOTH into one merged K=2 Trajectory via reconstruct_trajectory (called
    twice) + the new merge_trajectories.

    Expects the uploaded episode to also have decoy_gt_masks.npy (the
    phase-1 distractor scene's own format -- occlusion_corridor_distractor
    episodes already save this) alongside the usual gt_masks.npy. Returns
    per-object IoU (primary against gt_masks.npy, secondary against
    decoy_gt_masks.npy) plus the merged trajectory's arrays, same shape
    convention as run_gate2_episode's traj_pos/traj_present."""
    import json
    import functools
    import numpy as np
    import sys
    sys.path.insert(0, "/root/vitals_pkg")
    from vitals.phi import segmentation as seg
    from vitals.phi import reidentify as reid
    from vitals.phi import reconstruct as recon
    from vitals.phi import scene_geometry as sg

    episode_dir = pathlib.Path(f"/cache/episodes/{name}")
    meta = json.load(open(episode_dir / "meta.json"))
    gt_masks = np.load(episode_dir / "gt_masks.npy")
    decoy_gt_masks = np.load(episode_dir / "decoy_gt_masks.npy")
    T = meta["T"]
    frame_paths = [str(episode_dir / "frames" / f"{i:05d}.jpg") for i in range(T)]

    camera_kwargs = dict(cam_pos=np.array(meta["cam_pos"]), cam_mat=np.array(meta["cam_mat"]),
                          fovy_deg=meta["fovy_deg"], width=meta["width"], height=meta["height"],
                          plane_z=meta["plane_z"])
    camera = camera_kwargs if use_metric_prior else None
    deceleration = sg.OCCLUSION_CORRIDOR_DECELERATION if use_metric_prior else None
    occluder_bounds = sg.OCCLUSION_CORRIDOR_WALL_BOUNDS if use_metric_prior else None

    predictor = seg.load_predictor(device="cuda")
    dino = reid.load_dino(device="cuda")
    mask_gen_factory = functools.partial(reid.load_mask_generator, device="cuda",
                                          points_per_side=32, points_per_batch=32)

    result = reid.track_with_reidentification(
        predictor, dino, mask_gen_factory, frame_paths, gt_masks[0],
        similarity_threshold=similarity_threshold, forgiveness_frames=forgiveness_frames,
        device="cuda", dt=1.0 / 30, camera=camera, deceleration=deceleration,
        occluder_bounds=occluder_bounds, frame0_mask_secondary=decoy_gt_masks[0])

    def iou(a, b):
        inter = (a & b).sum()
        union = (a | b).sum()
        return float(inter) / float(union) if union > 0 else (1.0 if a.sum() == b.sum() == 0 else 0.0)

    ious_primary = np.array([iou(result.masks.get(i, np.zeros_like(gt_masks[i])), gt_masks[i]) for i in range(T)])
    ious_secondary = np.array([iou(result.masks_secondary.get(i, np.zeros_like(decoy_gt_masks[i])),
                                    decoy_gt_masks[i]) for i in range(T)])

    traj_primary = recon.reconstruct_trajectory(
        result.masks, fps=30, planar_motion=True, reid_events=result.reid_events, T=T,
        name="primary", forgiveness_frames=forgiveness_frames,
        known_occluded_frames=result.known_occluded_frames, **camera_kwargs)
    traj_secondary = recon.reconstruct_trajectory(
        result.masks_secondary, fps=30, planar_motion=True, reid_events=[], T=T,
        name="secondary", forgiveness_frames=forgiveness_frames, **camera_kwargs)
    merged = recon.merge_trajectories([traj_primary, traj_secondary])

    summary = dict(
        n_reid_events=len(result.reid_events),
        mean_iou_primary=float(ious_primary.mean()),
        mean_iou_secondary=float(ious_secondary.mean()),
        merged_names=merged.names,
        traj_t=merged.t.tolist(),
        traj_pos=np.where(np.isnan(merged.pos), None, merged.pos).tolist(),
        traj_present=merged.present.tolist(),
    )
    model_cache.commit()
    return summary


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": model_cache}, timeout=300)
def diagnose_candidates(name: str, frames: str = "57,59,61,65"):
    """One-off diagnostic, kept under the same deployed app (not a
    separate one -- a prior version of this mistakenly created a second
    "vitals-phi-diag" app via `modal.App(...)` instead of adding a
    function to this one; caught and fixed). At specific frames of an
    uploaded episode, checks whether SAM2's automatic mask generator
    proposes ANY candidate near the true ball position at all -- used to
    tell apart "prediction is accurate but no matching candidate exists"
    from other failure modes (AGENT.md defect #18's investigation)."""
    import numpy as np
    from PIL import Image
    import torch
    import sys
    sys.path.insert(0, "/root/vitals_pkg")
    from vitals.phi import reidentify as reid
    from vitals.phi import motion_prior as mp

    episode_dir = pathlib.Path(f"/cache/episodes/{name}")
    gt_masks = np.load(episode_dir / "gt_masks.npy")
    mask_gen = reid.load_mask_generator(device="cuda", points_per_side=32, points_per_batch=32)

    out = []
    for i in [int(x) for x in frames.split(",")]:
        rgb = np.array(Image.open(episode_dir / "frames" / f"{i:05d}.jpg").convert("RGB"))
        gt = gt_masks[i]
        gt_cen = mp.centroid(gt)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            candidates = mask_gen.generate(rgb)
        best = None
        for c in candidates:
            cen = mp.centroid(c["segmentation"])
            if cen is None:
                continue
            d = float(np.hypot(cen[0] - gt_cen[0], cen[1] - gt_cen[1])) if gt_cen else None
            inter = (c["segmentation"] & gt).sum()
            union = (c["segmentation"] | gt).sum()
            iou = float(inter) / float(union) if union > 0 else 0.0
            if best is None or (d is not None and d < best[0]):
                best = (d, cen, int(c["area"]), iou)
        row = dict(frame=i, gt_visible=bool(gt.sum() > 0), gt_area=int(gt.sum()), gt_centroid=gt_cen,
                   n_candidates=len(candidates),
                   closest_dist=best[0] if best else None, closest_centroid=best[1] if best else None,
                   closest_area=best[2] if best else None, closest_iou=best[3] if best else None)
        print(row)
        out.append(row)
    return out


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": model_cache}, timeout=1200)
def run_phi_reconstruct_from_frames(all_frames, prefix_len, scenario_name, frame0_mask,
                                    cam_pos, cam_mat, fovy_deg, fps=30,
                                    similarity_threshold=0.4, forgiveness_frames=1):
    """Runs the REAL production Phi pipeline (`vitals/adapters/video_model.
    py`'s own `default_phi_reconstruct` -- SAM2 tracking + DINOv2
    re-identification, then 3D reconstruction) on an ALREADY-GENERATED
    frame sequence -- e.g. a real model's continuation stitched onto its
    own conditioning prefix (AGENT.md M6, 2026-08: "run it through the Phi
    reconstruction pipeline", the first real Cosmos output). This is the
    Modal-remote counterpart `default_phi_reconstruct` itself assumes GPU
    access for -- that function is real, production code, but was never
    itself wired as a callable Modal function until this; lets a local
    caller (no GPU) invoke it exactly like every other real GPU call in
    this project already works (SAM2/DINOv2 via this same app).

    all_frames, frame0_mask, cam_pos, cam_mat: plain numpy arrays, passed
    directly over Modal's own RPC -- cloudpickle handles numpy fine at this
    size (a few hundred frames, tens of MB at most), no manual
    encoding needed.

    Returns the reconstructed `Trajectory` object directly -- `vitals.
    types.Trajectory` is importable on both sides via the same mounted
    `vitals/` package everything else in this file already uses."""
    import sys
    sys.path.insert(0, "/root/vitals_pkg")
    from vitals.adapters.video_model import default_phi_reconstruct

    return default_phi_reconstruct(all_frames, prefix_len, scenario_name, frame0_mask,
                                    cam_pos, cam_mat, fovy_deg, fps=fps,
                                    similarity_threshold=similarity_threshold,
                                    forgiveness_frames=forgiveness_frames, device="cuda")
