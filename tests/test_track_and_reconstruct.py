"""vitals/adapters/video_model.py::track_and_reconstruct -- the
occluder_geometry/frame0_mask_secondary wiring specifically (2026-08, a
real gap found running GATE 2 on occlusion_corridor_moving for the first
time: this function never threaded a second tracked object through at
all, so that scenario's own dynamic-occluder-awareness -- already built
and calibrated at the state-space level, M5.7 -- was silently
UNREACHABLE from real video, GATE 2 or real-model scoring, neither of
which crashed (occluder_bounds=None is a valid, degraded input)).

No GPU/SAM2/DINO needed: `reid.track_with_reidentification` and
`recon.reconstruct_trajectory` are monkeypatched to fakes that just
RECORD what they were called with, isolating this test to the wiring
logic (does the right cfg produce the right kwargs) rather than real
tracking/reconstruction accuracy (separately tested elsewhere, GPU-only,
tests/test_reconstruct.py)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from vitals.phi import scene_geometry as sg


class _FakeReidResult:
    def __init__(self):
        self.masks = {}
        self.masks_secondary = {}
        self.reid_events = []
        self.known_occluded_frames = {}


def _patch_gpu_pieces(monkeypatch, calls):
    """Patches every GPU-touching piece track_and_reconstruct calls, plus
    the two functions whose call KWARGS this test actually inspects
    (track_with_reidentification, reconstruct_trajectory) -- appends a
    dict of the kwargs each was called with to `calls` so the test can
    assert on them afterward."""
    from vitals.phi import segmentation as seg
    from vitals.phi import reidentify as reid
    from vitals.phi import reconstruct as recon

    monkeypatch.setattr(seg, "load_predictor", lambda device: "fake-predictor")
    monkeypatch.setattr(reid, "load_dino", lambda device: "fake-dino")
    monkeypatch.setattr(reid, "load_mask_generator", lambda **kw: "fake-mask-gen")

    def fake_track(*args, **kwargs):
        calls.append(("track_with_reidentification", kwargs))
        return _FakeReidResult()

    def fake_reconstruct(*args, **kwargs):
        calls.append(("reconstruct_trajectory", kwargs))
        from vitals.types import Trajectory
        return Trajectory(t=np.zeros(1), pos=np.zeros((1, 1, 3)),
                          quat=np.tile([1.0, 0, 0, 0], (1, 1, 1)),
                          present=np.zeros((1, 1), dtype=bool), names=["ball"])

    monkeypatch.setattr(reid, "track_with_reidentification", fake_track)
    monkeypatch.setattr(recon, "reconstruct_trajectory", fake_reconstruct)


def _call_track_and_reconstruct(scenario_name, frame0_mask_secondary=None, use_metric_prior=True):
    from vitals.adapters.video_model import track_and_reconstruct
    return track_and_reconstruct(
        frame_paths=["/dev/null"], frame0_mask=np.zeros((240, 320), dtype=bool),
        scenario_name=scenario_name, cam_pos=np.zeros(3), cam_mat=np.eye(3), fovy_deg=45.0,
        width=320, height=240, use_metric_prior=use_metric_prior,
        frame0_mask_secondary=frame0_mask_secondary)


def test_occluder_geometry_passed_through_for_occlusion_corridor_moving():
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    calls = []
    try:
        _patch_gpu_pieces(monkeypatch, calls)
        secondary_mask = np.ones((240, 320), dtype=bool)
        _call_track_and_reconstruct("occlusion_corridor_moving", frame0_mask_secondary=secondary_mask)

        track_kwargs = dict(calls[0][1])
        assert track_kwargs["occluder_geometry"] == sg.OCCLUSION_CORRIDOR_MOVING_OCCLUDER_GEOMETRY
        assert track_kwargs["frame0_mask_secondary"] is secondary_mask
        assert track_kwargs["camera"] is not None, "occluder_geometry needs a resolved camera"
    finally:
        monkeypatch.undo()


def test_occluder_geometry_forced_none_without_metric_prior():
    """The safety gate this fix specifically added: track_with_
    reidentification itself raises if occluder_geometry is given without
    camera, so occluder_geometry must be suppressed (not just left to
    crash downstream) whenever use_metric_prior=False makes camera=None."""
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    calls = []
    try:
        _patch_gpu_pieces(monkeypatch, calls)
        secondary_mask = np.ones((240, 320), dtype=bool)
        _call_track_and_reconstruct("occlusion_corridor_moving", frame0_mask_secondary=secondary_mask,
                                     use_metric_prior=False)

        track_kwargs = dict(calls[0][1])
        assert track_kwargs["camera"] is None
        assert track_kwargs["occluder_geometry"] is None
    finally:
        monkeypatch.undo()


def test_occluder_geometry_stays_none_for_a_scenario_that_never_registered_it():
    """occlusion_corridor_distractor's own decoy is an inert object (M5.6
    scoping) -- passing a frame0_mask_secondary for it must NOT suddenly
    activate occluder tracking that was never registered for this
    scenario in scene_geometry.py."""
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    calls = []
    try:
        _patch_gpu_pieces(monkeypatch, calls)
        decoy_mask = np.ones((240, 320), dtype=bool)
        _call_track_and_reconstruct("occlusion_corridor_distractor", frame0_mask_secondary=decoy_mask)

        track_kwargs = dict(calls[0][1])
        assert track_kwargs["occluder_geometry"] is None
        # frame0_mask_secondary itself is still passed through unconditionally --
        # track_with_reidentification's own track_secondary flag is what
        # decides whether to actually USE it, not this function's job to gate.
        assert track_kwargs["frame0_mask_secondary"] is decoy_mask
    finally:
        monkeypatch.undo()


def test_occlusion_corridor_unaffected_no_secondary_mask_given():
    """Every existing caller (occlusion_corridor itself, ramp_descent_
    high_friction, ...) never passes frame0_mask_secondary at all --
    confirms the new parameter is a strict no-op by default."""
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    calls = []
    try:
        _patch_gpu_pieces(monkeypatch, calls)
        _call_track_and_reconstruct("occlusion_corridor")

        track_kwargs = dict(calls[0][1])
        assert track_kwargs["occluder_geometry"] is None
        assert track_kwargs["frame0_mask_secondary"] is None
    finally:
        monkeypatch.undo()


if __name__ == "__main__":
    import inspect
    fns = [obj for name, obj in sorted(globals().items())
           if name.startswith("test_") and inspect.isfunction(obj)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
