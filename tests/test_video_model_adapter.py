"""adapters/video_model.py -- the real-model WorldModel adapter (AGENT.md
M6). Requires MuJoCo (rendering); skipped (not failed) if absent, matching
test_gates.py's convention. Does NOT require SAM2/DINOv2/GPU -- `phi_fn` is
injected with a ground-truth-segmentation-mask stand-in throughout, same
isolation convention as tests/test_reconstruct.py: this validates the
adapter's OWN glue (rendering the prefix, calling generate_fn with the
right frame count, stitching+reconstructing+slicing back to a continuation)
independently of SAM2's separately-measured tracking error.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

SCENE = str(pathlib.Path(__file__).resolve().parents[1] / "scenes" / "occlusion_corridor.xml")


def _setup():
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend
    spec = EpisodeSpec(name="occlusion_corridor", scene=SCENE, target_property="P2", band="I",
                        lam=6.0, n_reference=1, horizon_s=3.0, fps=30, perturb_mode="velocity_x_only")
    rollout = make_backend("mujoco", scene=SCENE)
    return rollout, spec


def _perfect_generate_fn_factory(true_continuation_traj, renderer, camera):
    """A stand-in "model" that already knows the true continuation and just
    re-renders it -- NOT a claim about how a real model works, purely a
    round-trip check that the adapter's own plumbing (render -> generate_fn
    -> stitch -> reconstruct -> slice) doesn't lose or corrupt information
    anywhere along the way. A real model's generate_fn would call actual
    inference instead; this one exists only to give predict() something
    deterministic and checkable to call."""
    def generate_fn(prefix_frames, n_frames):
        assert n_frames == true_continuation_traj.T, "adapter asked for the wrong frame count"
        frames_obj, _ = renderer.render(true_continuation_traj, cameras=camera)
        return frames_obj.rgb
    return generate_fn


def _ground_truth_phi_fn_factory(full_true_traj, renderer, camera):
    """Ground-truth-segmentation-mask stand-in for the real SAM2/DINOv2 Phi
    pipeline (same convention as test_reconstruct.py's own _phi_traj) --
    re-renders the FULL true prefix+continuation trajectory to get its own
    ground-truth masks, ignores the frames the adapter actually generated
    (this test's own stub already made those pixel-identical to the truth
    anyway) and reconstructs from the true masks directly, bypassing
    segmentation/re-identification entirely. Isolates this test from
    SAM2's own, separately-measured tracking error."""
    from vitals.phi.reconstruct import reconstruct_trajectory
    from vitals.phi import scene_geometry as sg

    def phi_fn(all_frames, prefix_len, frame0_mask, cam_pos, cam_mat, fovy_deg):
        frames_obj, gt = renderer.render(full_true_traj, cameras=camera)
        masks = {i: (gt.segmentation[i] == 0) for i in range(full_true_traj.T)}
        cfg = sg.get("occlusion_corridor")
        return reconstruct_trajectory(masks, fps=30, cam_pos=cam_pos, cam_mat=cam_mat,
                                       fovy_deg=fovy_deg, width=frames_obj.rgb.shape[2],
                                       height=frames_obj.rgb.shape[1], name="ball",
                                       planar_motion=True, reid_events=[], T=full_true_traj.T,
                                       **cfg["reconstruct_kwargs"])
    return phi_fn


def test_predict_recovers_the_true_continuation_end_to_end():
    """The actual GATE-3-style sanity check for this adapter: with a
    "perfect" generate_fn (regurgitates the true continuation) and a
    ground-truth phi_fn (bypasses SAM2), predict() must recover a
    continuation trajectory close to the true one -- proving the
    surrounding glue (render/stitch/reconstruct/slice, all NEW code this
    session) doesn't lose or corrupt information, independent of whether
    any real model or real segmentation is involved yet."""
    from vitals.adapters.video_model import VideoWorldModel
    from vitals.adapters.base import prefix_of
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg

    rollout, spec = _setup()
    full = rollout(spec, seed=1)
    prefix_len = 20
    conditioning = prefix_of(full, prefix_len)

    cfg = sg.get("occlusion_corridor")
    renderer = MujocoRenderer(SCENE, height=240, width=320)

    n_continuation = full.T - prefix_len
    from vitals.types import Trajectory
    true_continuation = Trajectory(
        t=np.arange(n_continuation) * full.dt,
        pos=full.pos[prefix_len:].copy(), quat=full.quat[prefix_len:].copy(),
        present=full.present[prefix_len:].copy(), names=list(full.names), meta=dict(full.meta))

    generate_fn = _perfect_generate_fn_factory(true_continuation, renderer, cfg["camera"])
    phi_fn = _ground_truth_phi_fn_factory(full, renderer, cfg["camera"])

    model = VideoWorldModel(scenario_name="occlusion_corridor", scene=SCENE, generate_fn=generate_fn,
                            phi_fn=phi_fn, fps=30, height=240, width=320)
    horizon_s = n_continuation * full.dt
    samples = model.predict(conditioning, horizon_s=horizon_s, n_samples=1)

    assert len(samples) == 1
    cont = samples[0]
    assert cont.T == n_continuation, f"expected {n_continuation} frames back, got {cont.T}"
    assert cont.t[0] == 0.0, "predict() must return a continuation with time re-zeroed to 0, not stitched"

    visible = cont.present[:, 0] & ~np.isnan(cont.pos[:, 0, 0])
    assert visible.sum() > 10, "expected most of this pre-occlusion window to reconstruct"
    err = np.linalg.norm(cont.pos[visible, 0] - true_continuation.pos[visible, 0], axis=-1)
    assert err.mean() < 0.15, f"end-to-end adapter round trip error too high: mean={err.mean():.3f}m"


def test_predict_capture_frames_stores_last_capture():
    """capture_frames=True (2026-08, run_model_population.py's grid-video
    feature) must stash pixels + reconstructed positions on self.last_
    capture WITHOUT changing predict()'s own return value -- the capture
    is purely additive, read afterward by the caller, never substituted
    for the real return."""
    from vitals.adapters.video_model import VideoWorldModel
    from vitals.adapters.base import prefix_of
    from vitals.render.mujoco_renderer import MujocoRenderer
    from vitals.phi import scene_geometry as sg
    from vitals.types import Trajectory

    rollout, spec = _setup()
    full = rollout(spec, seed=1)
    prefix_len = 20
    conditioning = prefix_of(full, prefix_len)

    cfg = sg.get("occlusion_corridor")
    renderer = MujocoRenderer(SCENE, height=240, width=320)
    n_continuation = full.T - prefix_len
    true_continuation = Trajectory(
        t=np.arange(n_continuation) * full.dt,
        pos=full.pos[prefix_len:].copy(), quat=full.quat[prefix_len:].copy(),
        present=full.present[prefix_len:].copy(), names=list(full.names), meta=dict(full.meta))

    generate_fn = _perfect_generate_fn_factory(true_continuation, renderer, cfg["camera"])
    phi_fn = _ground_truth_phi_fn_factory(full, renderer, cfg["camera"])
    model = VideoWorldModel(scenario_name="occlusion_corridor", scene=SCENE, generate_fn=generate_fn,
                            phi_fn=phi_fn, fps=30, height=240, width=320)

    assert not hasattr(model, "last_capture"), "must not exist before any capture=True call"
    horizon_s = n_continuation * full.dt
    samples = model.predict(conditioning, horizon_s=horizon_s, n_samples=1, capture_frames=True)

    cap = model.last_capture
    n_episode_frames = prefix_len + n_continuation
    assert cap["all_frames"].shape == (n_episode_frames, 240, 320, 3)
    # (T,K,3), not (T,3) -- 2026-09, billiards/multi-collision scoping:
    # recon_pos now always carries every reconstructed object (K=1 here,
    # a strict single-object scenario), not just a hardcoded object 0.
    assert cap["recon_pos"].shape == (n_episode_frames, 1, 3)
    assert cap["prefix_len"] == prefix_len
    assert cap["cam_pos"] is not None and cap["cam_mat"] is not None
    # predict()'s own return value is unaffected by capture_frames
    assert samples[0].T == n_continuation


def test_predict_rejects_wrong_frame_count_from_generate_fn():
    """generate_fn returning the wrong number of frames is a contract
    violation the adapter must catch, not silently mis-stitch."""
    from vitals.adapters.video_model import VideoWorldModel
    from vitals.adapters.base import prefix_of

    rollout, spec = _setup()
    full = rollout(spec, seed=1)
    conditioning = prefix_of(full, 20)

    def bad_generate_fn(prefix_frames, n_frames):
        return np.zeros((n_frames - 1, 240, 320, 3), dtype=np.uint8)   # one short, on purpose

    model = VideoWorldModel(scenario_name="occlusion_corridor", scene=SCENE, generate_fn=bad_generate_fn,
                            phi_fn=lambda *a, **kw: None)
    try:
        model.predict(conditioning, horizon_s=1.0, n_samples=1)
        assert False, "should reject a generate_fn that returns the wrong frame count"
    except ValueError:
        pass


def test_predict_threads_frame0_mask_secondary_only_for_p3():
    """2026-09: R2/sigma_interpenetration wired into real-model scoring --
    predict() must extract and pass a second object's own ground-truth
    frame-0 mask to phi_fn ONLY when target_property == "P3", and must
    NOT pass it (not even as None) for a P2 scenario -- verifies both the
    opt-in AND the no-op side of VideoWorldModel's own new `target_
    property` docstring, using a K=2 scene (occlusion_corridor_distractor.
    xml, reused as-is by occlusion_corridor_interpenetration's own
    manifest) so there IS a real second object to find."""
    from vitals.adapters.video_model import VideoWorldModel
    from vitals.adapters.base import prefix_of
    from vitals.types import EpisodeSpec
    from vitals.physics import make_backend

    scene = str(pathlib.Path(__file__).resolve().parents[1] / "scenes" / "occlusion_corridor_distractor.xml")
    spec = EpisodeSpec(name="occlusion_corridor_interpenetration", scene=scene, target_property="P3",
                        band="I", lam=6.0, n_reference=1, horizon_s=2.0, fps=30,
                        perturb_mode="velocity_x_only")
    rollout = make_backend("mujoco", scene=scene)
    full = rollout(spec, seed=1)
    conditioning = prefix_of(full, 10)
    assert conditioning.K >= 2, "test fixture needs a real second object to be meaningful"

    calls = []

    def recording_phi_fn(all_frames, prefix_len, frame0_mask, cam_pos, cam_mat, fovy_deg,
                          frame0_mask_secondary=None):
        calls.append(frame0_mask_secondary)
        from vitals.types import Trajectory
        T = all_frames.shape[0]
        return Trajectory(t=np.arange(T) / 30.0, pos=np.zeros((T, 1, 3)), quat=np.zeros((T, 1, 4)),
                          present=np.ones((T, 1), dtype=bool), names=["ball"])

    model_p3 = VideoWorldModel(scenario_name="occlusion_corridor_interpenetration", scene=scene,
                               generate_fn=lambda frames, n: frames[:n], phi_fn=recording_phi_fn,
                               fps=30, height=240, width=320, target_property="P3")
    model_p3.predict(conditioning, horizon_s=0.3, n_samples=1)
    assert len(calls) == 1 and calls[0] is not None, "P3 must pass a real frame0_mask_secondary"
    assert calls[0].any(), "the second object's own mask must be non-empty"

    calls.clear()
    model_p2 = VideoWorldModel(scenario_name="occlusion_corridor_interpenetration", scene=scene,
                               generate_fn=lambda frames, n: frames[:n], phi_fn=recording_phi_fn,
                               fps=30, height=240, width=320, target_property="P2")
    model_p2.predict(conditioning, horizon_s=0.3, n_samples=1)
    assert len(calls) == 1 and calls[0] is None, "non-P3 must stay a strict no-op (no secondary mask at all)"


def test_video_world_model_construction_needs_no_gpu():
    """Constructing a VideoWorldModel with the DEFAULT phi_fn (the real
    production Phi pipeline, needs SAM2/DINOv2/GPU when actually CALLED)
    must not eagerly import torch/SAM2 at construction time -- only inside
    the lambda closure, evaluated on first real predict() call. This is
    what lets this exact class be constructed and unit-tested (with an
    override) on a machine with no GPU at all, like this one."""
    from vitals.adapters.video_model import VideoWorldModel
    model = VideoWorldModel(scenario_name="occlusion_corridor", scene=SCENE,
                            generate_fn=lambda frames, n: frames[:n])
    assert model.phi_fn is not None   # constructed without raising -- the actual assertion


if __name__ == "__main__":
    test_video_world_model_construction_needs_no_gpu()
    print("PASS  test_video_world_model_construction_needs_no_gpu")
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIP  mujoco not installed")
    else:
        test_predict_recovers_the_true_continuation_end_to_end()
        print("PASS  test_predict_recovers_the_true_continuation_end_to_end")
        test_predict_capture_frames_stores_last_capture()
        print("PASS  test_predict_capture_frames_stores_last_capture")
        test_predict_rejects_wrong_frame_count_from_generate_fn()
        print("PASS  test_predict_rejects_wrong_frame_count_from_generate_fn")
