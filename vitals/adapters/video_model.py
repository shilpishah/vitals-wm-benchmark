"""WorldModel adapter for a REAL video-generation model (AGENT.md M6).

Two clearly-marked injection points, each independently swappable and
independently testable -- this is the "one adapter per model, config/data
per scenario" shape discussed and built out in `scene_geometry.py`, applied
to the actual model-calling half:

  generate_fn(prefix_frames, n_frames) -> continuation_frames
      THE REAL MODEL'S OWN inference call plugs in here. Takes the rendered
      conditioning prefix as RGB frames ((T,H,W,3) uint8) plus how many new
      frames to generate, returns EXACTLY that many new RGB frames -- no
      other input, matching the input-normalization protocol's own rule 1
      (pixels only, no privileged state/camera/masks handed to the model).
      Has NO default -- there is no generic stand-in for "a real video
      model"; a caller must supply one.

  phi_fn(all_frames, prefix_len, frame0_mask, cam_pos, cam_mat, fovy_deg)
      -> Trajectory
      THE REAL PHI PIPELINE (segmentation, re-identification, 3D
      reconstruction) plugs in here, defaulted to `default_phi_reconstruct`
      below -- the actual production path (real SAM2/DINOv2 tracking via
      `reidentify.track_with_reidentification`, then `reconstruct.
      reconstruct_trajectory` using `scene_geometry`'s canonical per-
      scenario config). Needs GPU, exactly like every other real Phi call
      in this project. Receives the FULL frame sequence (prefix +
      generated continuation stitched together) and the prefix's own true
      frame-0 mask as the ONLY segmentation prompt (matching `render/
      __init__.py`'s own `GroundTruth` docstring: "Phi is only ever
      allowed the frame-0 mask... every other field is post-hoc validation
      data, not something Phi gets to see while running") -- returns the
      reconstructed Trajectory for the WHOLE sequence; `predict()` below
      slices it down to just the continuation before returning, matching
      the `WorldModel` protocol's own contract (a continuation starting
      fresh at t=0, not the prefix+continuation stitched together --
      `CopyLastState`/`ConstantVelocity` already follow this same
      contract).

Known, scoped limitation of `default_phi_reconstruct`, stated plainly, not
silently papered over: `motion_prior.fit_metric_track`'s physics-prior
re-identification tier only understands ONE flat plane (a single scalar
`plane_z`) -- it has no notion of a piecewise plane (ramp_descent) or no
plane at all (projectile, ballistic). For those two scenarios,
`default_phi_reconstruct` passes `camera=None` to `track_with_
reidentification`, which correctly falls back to the pixel-only
`PixelTrackFit` re-identification tier -- a real, already-existing, tested
code path, just less accurate than the metric-anchored one. Extending
physics-prior re-identification itself to piecewise/ballistic motion is a
separate, not-yet-scoped follow-up (this file composes what already
exists correctly; it does not build new Phi capability).
"""
from __future__ import annotations
import functools
import pathlib
import tempfile
import numpy as np
from ..types import Trajectory


def track_and_reconstruct(frame_paths, frame0_mask, scenario_name, cam_pos, cam_mat, fovy_deg,
                           width, height, fps=30, similarity_threshold=0.4, forgiveness_frames=1,
                           device="cuda", use_metric_prior=True, frame0_mask_secondary=None):
    """Scene-mode-aware track + reconstruct, shared by `default_phi_
    reconstruct` (below) and the remote GATE 2 pipeline (`remote/
    modal_app.py::run_gate2_episode`) -- extracted (2026-08) after finding
    a real drift bug: `run_gate2_episode` had grown its OWN separate copy
    of this exact logic, hardcoded to occlusion_corridor specifically
    (always `plane_z`, always `OCCLUSION_CORRIDOR_DECELERATION`/
    `OCCLUSION_CORRIDOR_WALL_BOUNDS`), which silently broke the moment a
    `plane_pieces` scenario (`ramp_descent_high_friction`) needed GATE 2 --
    `default_phi_reconstruct` itself already branched on scene mode
    correctly; GATE 2's own copy never did, because it was written before
    that branching existed and never updated to match. One implementation
    now, not two that can drift apart again.

    `use_metric_prior=False` is GATE 2's own ablation switch (does re-id
    do better/worse WITHOUT the metric-anchored prior?) -- gates the
    camera/deceleration prior specifically; `occluder_bounds` is NOT part
    of that ablation (matches `default_phi_reconstruct`'s own original,
    always-on `cfg.get("occluder_bounds")` -- a small, deliberate semantic
    fix while unifying: the pre-unification `run_gate2_episode` gated
    occluder_bounds under the SAME flag, which was never actually the
    question that flag was asking).

    frame0_mask_secondary: OPTIONAL (2026-08, a real gap found and fixed
    while running GATE 2 on `occlusion_corridor_moving` for the first
    time -- this function never threaded a second tracked object through
    at all, so that scenario's own dynamic-occluder-awareness, already
    built and calibrated at the state-space level, M5.7, was silently
    UNREACHABLE from either GATE 2 or real-model scoring, neither of
    which crashed -- `occluder_bounds=None` is a valid, if degraded,
    input -- so this sat undetected until a scenario that actually needed
    it was tried through this path). When given (and `cfg.get(
    "occluder_geometry")` is registered, e.g. `OCCLUSION_CORRIDOR_MOVING_
    OCCLUDER_GEOMETRY`), tracks a SECOND object (SAM2 obj_id=2, no
    search-and-reprompt of its own -- `track_with_reidentification`'s own
    documented scope) and uses ITS live tracked position, not a fixed
    tuple, to gate the primary object's own physics-prior search --
    exactly `reidentify.py`'s own `occluder_geometry` parameter, just
    finally wired to a real caller. Requires `camera` to also be resolved
    (use_metric_prior=True and mode=="plane_z") -- `track_with_
    reidentification` itself raises if occluder_geometry is given without
    it, so this is gated the SAME way here to fail at the right layer
    with the right message, not pass a doomed combination through.
    Still None (a strict no-op) for every scenario without a registered
    `occluder_geometry` -- e.g. `occlusion_corridor_distractor`'s own
    decoy object gets no special handling here, matching its own scoped
    design (M5.6: an inert decoy, not something the primary ball's own
    reconstruction needs to track).

    Returns (Trajectory, TrackResult) -- `default_phi_reconstruct` only
    needs the Trajectory; GATE 2 also needs the raw `TrackResult` (mask
    IoU, occlusion-flag accuracy -- pixel-space diagnostics `reconstruct_
    trajectory`'s own output doesn't carry)."""
    from ..phi import segmentation as seg
    from ..phi import reidentify as reid
    from ..phi import reconstruct as recon
    from ..phi import scene_geometry as sg

    cfg = sg.get(scenario_name)
    camera = None
    deceleration = None
    if use_metric_prior and cfg["mode"] == "plane_z":
        # Only the single-flat-plane case is wired for the physics-prior
        # re-id tier today -- see module docstring's own scoped-limitation
        # note for plane_pieces/ballistic.
        camera = dict(cam_pos=cam_pos, cam_mat=cam_mat, fovy_deg=fovy_deg,
                      width=width, height=height, plane_z=cfg["reconstruct_kwargs"]["plane_z"])
        deceleration = cfg.get("deceleration")   # None where not yet calibrated -- honest, not guessed

    # Only meaningful (and only safe to pass -- track_with_reidentification
    # itself requires `camera`) when the metric prior is actually active;
    # see this function's own docstring on frame0_mask_secondary above.
    occluder_geometry = cfg.get("occluder_geometry") if camera is not None else None

    predictor = seg.load_predictor(device=device)
    dino = reid.load_dino(device=device)
    mask_gen_factory = functools.partial(reid.load_mask_generator, device=device,
                                          points_per_side=32, points_per_batch=32)

    result = reid.track_with_reidentification(
        predictor, dino, mask_gen_factory, frame_paths, frame0_mask,
        similarity_threshold=similarity_threshold, forgiveness_frames=forgiveness_frames,
        device=device, dt=1.0 / fps, camera=camera, deceleration=deceleration,
        occluder_bounds=cfg.get("occluder_bounds"),
        frame0_mask_secondary=frame0_mask_secondary, occluder_geometry=occluder_geometry)

    traj = recon.reconstruct_trajectory(
        result.masks, fps=fps, cam_pos=cam_pos, cam_mat=cam_mat, fovy_deg=fovy_deg,
        width=width, height=height, name="ball", planar_motion=True,
        reid_events=result.reid_events, T=len(frame_paths),
        known_occluded_frames=result.known_occluded_frames,
        **cfg["reconstruct_kwargs"])
    return traj, result


def default_phi_reconstruct(all_frames, prefix_len, scenario_name, frame0_mask,
                             cam_pos, cam_mat, fovy_deg, fps=30,
                             similarity_threshold=0.4, forgiveness_frames=1, device="cuda"):
    """The real, production Phi pipeline -- needs GPU (SAM2 + DINOv2), same
    as every other real Phi call in this project. `all_frames`: (T,H,W,3)
    uint8, prefix and generated continuation already concatenated.
    `frame0_mask`: the prefix's own known ground-truth segmentation at
    frame 0 -- the ONLY privileged information Phi is ever allowed
    (`render/__init__.py`'s own `GroundTruth` docstring), used purely as
    SAM2's initial prompt, exactly like every other real Phi call in this
    project (`remote/modal_app.py`'s own `gt_masks[0]` usage).
    `cam_pos`/`cam_mat`/`fovy_deg`: the FIXED camera calibration captured
    once from rendering the prefix -- legitimate known instrument
    calibration (AGENT.md T3's note), not privileged access to the
    object's own state, and correct to reuse for the continuation too
    since the camera never moves (input-normalization protocol item 10).

    Scope boundary, stated plainly (2026-08): this function never passes
    `frame0_mask_secondary` through to `track_and_reconstruct`, unlike
    GATE 2's own `run_gate2_episode` (fixed this same session) -- so
    real-model scoring on `occlusion_corridor_moving` still runs WITHOUT
    dynamic-occluder-awareness today, the exact gap GATE 2 just found and
    fixed for ITSELF, not yet propagated here. A real video model has no
    ground-truth frame-0 mask for the occluder to hand this function
    anyway (real-model conditioning is pixels-only, input-normalization
    protocol item 1) -- closing this gap for real models needs the
    occluder's own position tracked from video too (e.g. reusing
    `default_phi_reconstruct`'s own primary-object SAM2 pass, prompted a
    second time), a genuinely different, not-yet-scoped problem from
    GATE 2's (which legitimately has a real frame-0 ground-truth mask for
    both objects, being a state-space validation pass). Left open, not
    silently implied fixed by this session's GATE 2 work."""
    from PIL import Image

    T, H, W = all_frames.shape[:3]

    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = pathlib.Path(tmp)
        frame_paths = []
        for i in range(T):
            p = frame_dir / f"{i:05d}.jpg"
            Image.fromarray(all_frames[i]).save(p)
            frame_paths.append(str(p))

        traj, _result = track_and_reconstruct(
            frame_paths, frame0_mask, scenario_name, cam_pos, cam_mat, fovy_deg, W, H,
            fps=fps, similarity_threshold=similarity_threshold, forgiveness_frames=forgiveness_frames,
            device=device)

    return traj


def _slice_continuation(full_traj: Trajectory, prefix_len: int, dt: float) -> Trajectory:
    """The continuation-only portion of a reconstructed prefix+continuation
    Trajectory, with time re-zeroed to start at the continuation's own
    first frame -- matches `CopyLastState`/`ConstantVelocity`'s own
    `predict()` contract (a fresh `t` starting at 0, not the stitched
    whole)."""
    n = full_traj.T - prefix_len
    t = np.arange(n) * dt
    return Trajectory(t=t, pos=full_traj.pos[prefix_len:].copy(), quat=full_traj.quat[prefix_len:].copy(),
                      present=full_traj.present[prefix_len:].copy(), names=list(full_traj.names),
                      meta=dict(full_traj.meta))


class VideoWorldModel:
    """WorldModel adapter for a real video-generation model. See module
    docstring for the two injection points (`generate_fn`, `phi_fn`).

    scenario_name: looked up in `scene_geometry.SCENES` for camera framing
    and reconstruction mode -- unlike `CopyLastState`/`ConstantVelocity`
    (fully scenario-agnostic, they only ever touch a Trajectory's own
    arrays), a real video model's adapter is NOT scenario-agnostic: it has
    to render the prefix with the right camera and reconstruct with the
    right plane/ballistic geometry, both scenario-specific. Bound once at
    construction, matching the input-normalization protocol's own item 10
    (the same fixed camera throughout one scenario's own scoring).

    scene: path to that scenario's own MJCF (e.g. `scenes/occlusion_
    corridor.xml`) -- needed to construct the renderer; the caller already
    has this (from the scenario's own `EpisodeSpec`/manifest), so this
    class does not reach into manifest-loading itself (separation of
    concerns: an adapter renders and reconstructs, it does not parse
    config files)."""

    def __init__(self, scenario_name, scene, generate_fn, name="video_model",
                 phi_fn=None, fps=30, height=240, width=320,
                 similarity_threshold=0.4, forgiveness_frames=1, device="cuda"):
        self.scenario_name = scenario_name
        self.scene = scene
        self.generate_fn = generate_fn
        self.name = name
        self.fps, self.height, self.width = fps, height, width
        if phi_fn is not None:
            self.phi_fn = phi_fn
        else:
            self.phi_fn = lambda all_frames, prefix_len, frame0_mask, cam_pos, cam_mat, fovy_deg: (
                default_phi_reconstruct(all_frames, prefix_len, scenario_name, frame0_mask,
                                         cam_pos, cam_mat, fovy_deg, fps=fps,
                                         similarity_threshold=similarity_threshold,
                                         forgiveness_frames=forgiveness_frames, device=device))

    def predict(self, conditioning: Trajectory, horizon_s: float, n_samples: int,
                capture_frames: bool = False) -> list[Trajectory]:
        """capture_frames: when True, stashes the last sample's raw pixels
        and reconstructed positions on `self.last_capture` (2026-08, added
        for `run_model_population.py`'s grid-video feature -- "all of the
        videos should be being generated... in one single video... to save
        space" -- so a caller building a video for a handful of episodes
        doesn't have to re-run the model a second time just to get the
        frames scoring already threw away). Default False: zero behavior/
        memory change for every existing caller that doesn't pass it."""
        from ..render.mujoco_renderer import MujocoRenderer
        from ..phi import scene_geometry as sg

        cfg = sg.get(self.scenario_name)
        renderer = MujocoRenderer(self.scene, height=self.height, width=self.width)
        prefix_frames_obj, prefix_gt = renderer.render(conditioning, cameras=cfg["camera"])
        prefix_frames = prefix_frames_obj.rgb
        prefix_len = conditioning.T
        n_continuation = max(1, int(round(horizon_s / conditioning.dt)))
        frame0_mask = (prefix_gt.segmentation[0] == 0)   # obj=0 convention, same as every other real Phi call

        samples = []
        for _ in range(n_samples):
            continuation_frames = np.asarray(self.generate_fn(prefix_frames, n_continuation))
            if continuation_frames.shape[0] != n_continuation:
                raise ValueError(f"generate_fn must return exactly {n_continuation} frames "
                                  f"(horizon_s={horizon_s} at {1/conditioning.dt:.1f}fps), "
                                  f"got {continuation_frames.shape[0]}")
            all_frames = np.concatenate([prefix_frames, continuation_frames], axis=0)
            full_traj = self.phi_fn(all_frames, prefix_len, frame0_mask,
                                     prefix_gt.cam_pos, prefix_gt.cam_mat, prefix_gt.fovy_deg)
            if capture_frames:
                self.last_capture = dict(all_frames=all_frames, prefix_len=prefix_len,
                                          recon_pos=full_traj.pos[:, 0].copy(),
                                          cam_pos=prefix_gt.cam_pos, cam_mat=prefix_gt.cam_mat,
                                          fovy_deg=prefix_gt.fovy_deg)
            samples.append(_slice_continuation(full_traj, prefix_len, conditioning.dt))
        return samples
