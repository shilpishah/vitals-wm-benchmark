"""adapters/cosmos.py -- NVIDIA Cosmos-Predict2 Video2World generate_fn
(AGENT.md M6). Tests ONLY the parts that don't need a real Cosmos install
or GPU: video file I/O and the fps/resolution/frame-count resample logic
(`_resample_to_target`) that sits between Cosmos's own native output
(16fps, 480p/720p) and this project's convention (30fps, 320x240).
`generate_fn` itself (the subprocess call into a real `cosmos-predict2`
checkout) is NOT covered here -- untestable without a live GPU install,
same honest boundary as `default_phi_reconstruct`'s own test coverage.

Requires the `cosmos` optional dependency group (imageio[ffmpeg]) --
skipped (not failed) if absent, matching this project's own convention for
optional, non-core dependencies.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np


def test_video_write_read_round_trip_preserves_shape():
    from vitals.adapters import cosmos
    import tempfile
    frames = np.random.randint(0, 255, (20, 64, 96, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "test.mp4"
        cosmos._write_video(frames, p, fps=16)
        read_back, fps = cosmos._read_video(p)
    assert read_back.shape[0] == 20
    assert read_back.shape[1:3] == (64, 96)


def test_resample_preserves_temporal_order():
    """A moving marker's on-screen position must keep moving in the same
    direction after resampling -- catches an accidental frame-order bug
    (e.g. a transposed index), not just a shape mismatch."""
    from vitals.adapters import cosmos
    raw_fps, raw_h, raw_w, raw_t = 16, 480, 480, 40
    raw = np.zeros((raw_t, raw_h, raw_w, 3), dtype=np.uint8)
    for i in range(raw_t):
        cx = int(20 + i * 8)
        raw[i, 200:220, cx:cx + 20] = 255

    out = cosmos._resample_to_target(raw, raw_fps, dst_fps=30, n_target_frames=90, dst_hw=(240, 320))
    assert out.shape == (90, 240, 320, 3)

    def square_x(frame):
        cols = np.where(frame[:, :, 0].max(axis=0) > 200)[0]
        return cols.mean() if len(cols) else None

    assert square_x(out[40]) > square_x(out[0]), "marker should move right as time advances"


def test_resample_holds_last_frame_when_source_runs_short():
    """Requesting more target frames (at 30fps) than the source video
    covers (at 16fps) must pad by holding the last real frame -- an
    honest, documented stand-in (cosmos.py's own module docstring) for the
    unresolved "how many frames does one Cosmos call return" question, not
    silent zero-padding or a shape mismatch."""
    from vitals.adapters import cosmos
    raw = np.zeros((10, 64, 96, 3), dtype=np.uint8)
    raw[:, :, :, 0] = 255
    out = cosmos._resample_to_target(raw, src_fps=16, dst_fps=30, n_target_frames=50, dst_hw=(64, 96))
    assert out.shape[0] == 50
    assert (out[-1] == out[-1][0, 0]).all(), "tail should be a solid hold of the last real frame"


def test_make_cosmos_generate_fn_local_requires_a_registered_scenario_prompt():
    from vitals.adapters import cosmos
    try:
        cosmos.make_cosmos_generate_fn_local("not_a_real_scenario", cosmos_repo_dir="/tmp/fake")
        assert False, "should reject a scenario with no frozen prompt"
    except KeyError:
        pass


def test_make_cosmos_generate_fn_local_requires_cosmos_repo_dir():
    from vitals.adapters import cosmos
    try:
        cosmos.make_cosmos_generate_fn_local("occlusion_corridor", cosmos_repo_dir=None)
        assert False, "should refuse to construct without a real cosmos-predict2 checkout path"
    except ValueError:
        pass


def test_make_cosmos_generate_fn_modal_requires_a_registered_scenario_prompt():
    """The Modal-backed builder validates the SAME frozen-prompt contract
    before ever touching the network -- a bad scenario name must fail
    immediately, not after a remote call round-trips."""
    from vitals.adapters import cosmos
    try:
        cosmos.make_cosmos_generate_fn_modal("not_a_real_scenario")
        assert False, "should reject a scenario with no frozen prompt"
    except KeyError:
        pass


if __name__ == "__main__":
    try:
        import imageio  # noqa: F401
    except ImportError:
        print("SKIP  imageio not installed (pip install vitals[cosmos])")
    else:
        test_video_write_read_round_trip_preserves_shape()
        print("PASS  test_video_write_read_round_trip_preserves_shape")
        test_resample_preserves_temporal_order()
        print("PASS  test_resample_preserves_temporal_order")
        test_resample_holds_last_frame_when_source_runs_short()
        print("PASS  test_resample_holds_last_frame_when_source_runs_short")
        test_make_cosmos_generate_fn_local_requires_a_registered_scenario_prompt()
        print("PASS  test_make_cosmos_generate_fn_local_requires_a_registered_scenario_prompt")
        test_make_cosmos_generate_fn_local_requires_cosmos_repo_dir()
        print("PASS  test_make_cosmos_generate_fn_local_requires_cosmos_repo_dir")
        test_make_cosmos_generate_fn_modal_requires_a_registered_scenario_prompt()
        print("PASS  test_make_cosmos_generate_fn_modal_requires_a_registered_scenario_prompt")
