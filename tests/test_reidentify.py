"""vitals/phi/reidentify.py::_maybe_recache_embedding -- pure/local-IO
logic extracted from track_with_reidentification's own inline
re-embedding step (2026-08, after a real, reproduced-but-not-fully-
root-caused `IndexError: list index out of range` at `frame_paths[
frame_idx - 1]` on a jitter-mutant/ramp_descent_high_friction episode,
AGENT.md M2.6). track_with_reidentification itself needs a real SAM2
predictor + DINO model (GPU-only, not available on this dev machine) --
this function doesn't, which is the whole point of extracting it."""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from vitals.phi import reidentify as reid


def _write_fake_frames(tmpdir, n):
    paths = []
    for i in range(n):
        p = pathlib.Path(tmpdir) / f"{i:05d}.jpg"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    return paths


def test_frame_zero_already_empty_does_not_crash():
    """prev_idx = -1 -- the exact case AGENT.md's own writeup hypothesized
    as the trigger. Must return None, not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = _write_fake_frames(tmp, 3)
        assert reid._maybe_recache_embedding(0, {}, frame_paths, dino_model=None, device="cpu") is None


def test_prev_idx_beyond_frame_paths_does_not_crash():
    """The actual reported crash: frame_paths[prev_idx] with no upper-bound
    guard. Simulates a frame_idx=4 (prev_idx=3) that would put prev_idx
    at/past the end of a 3-frame frame_paths (e.g. a length mismatch
    between frame_paths and whatever SAM2 actually propagated over) --
    must return None, not raise IndexError.

    Confirmed directly (not assumed) that the OLD unsafe logic really
    does crash on this exact input before trusting this test: `masks.
    get(prev_idx)` non-empty at prev_idx=3, then `frame_paths[3]` on a
    length-3 list raises `IndexError: list index out of range`, byte-
    identical to the originally reported crash."""
    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = _write_fake_frames(tmp, 3)   # valid indices 0,1,2
        masks = {3: np.ones((4, 4), dtype=bool)}   # "visible" at prev_idx=3, but frame_paths has no index 3
        assert reid._maybe_recache_embedding(4, masks, frame_paths, dino_model=None, device="cpu") is None


def test_no_prior_mask_recorded_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = _write_fake_frames(tmp, 3)
        assert reid._maybe_recache_embedding(1, {}, frame_paths, dino_model=None, device="cpu") is None


def test_prior_mask_empty_returns_none_without_opening_the_image():
    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = _write_fake_frames(tmp, 3)
        masks = {0: np.zeros((4, 4), dtype=bool)}   # tracked, but sum()==0
        assert reid._maybe_recache_embedding(1, masks, frame_paths, dino_model=None, device="cpu") is None


def test_valid_prior_mask_calls_embed_region_and_returns_its_result(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = _write_fake_frames(tmp, 3)
        masks = {0: np.ones((8, 8), dtype=bool)}
        sentinel = np.array([1.0, 2.0, 3.0])
        calls = []

        def fake_embed_region(dino_model, frame_rgb, mask, device):
            calls.append((dino_model, frame_rgb.shape, mask.shape, device))
            return sentinel

        monkeypatch.setattr(reid, "embed_region", fake_embed_region)
        result = reid._maybe_recache_embedding(1, masks, frame_paths, dino_model="fake-dino", device="cpu")
        assert result is sentinel
        assert len(calls) == 1
        assert calls[0][0] == "fake-dino" and calls[0][3] == "cpu"


if __name__ == "__main__":
    import inspect

    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    fns = [obj for name, obj in sorted(globals().items())
           if name.startswith("test_") and inspect.isfunction(obj)]
    passed = 0
    for fn in fns:
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                fn(_FakeMonkeypatch())
            else:
                fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
