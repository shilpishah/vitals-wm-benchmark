"""vitals/adapters/runway.py -- API-access model adapter. No key, no
network, no runwayml import needed: a fake client is injected via the
adapter's own `client_factory` injection point, and the "downloaded" mp4
is a real local file written with imageio so the decode/resample path is
exercised for real, not mocked away."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import base64
import numpy as np
import pytest

from vitals.adapters import runway as rw
from vitals.adapters.video_utils import _write_video


class _Task:
    def __init__(self, status="SUCCEEDED", output=None, credits=7):
        self.id = "task_test"
        self.status = status
        self.output = output or []
        self.cost = type("C", (), {"credits": credits})()

    def wait_for_task_output(self, timeout=None):
        if self.status == "FAILED":
            raise RuntimeError("task failed")
        return self


class _FakeClient:
    """Records the create() kwargs and returns a canned task."""
    def __init__(self, task, calls):
        self._task, self._calls = task, calls
        self.image_to_video = self

    def create(self, **kwargs):
        self._calls.append(kwargs)
        return self._task


def _fake_video_url(tmp_path, n_frames, fps, hw=(120, 160)):
    """Write a real small mp4 and hand back a file:// URL -- urlretrieve
    handles file:// natively, so the adapter's download path runs
    unchanged."""
    frames = (np.random.default_rng(0).integers(0, 255, size=(n_frames, *hw, 3))).astype(np.uint8)
    p = tmp_path / "gen.mp4"
    _write_video(frames, p, fps=fps)
    return p.as_uri()


def test_frame_to_data_uri_is_lossless_png():
    frame = np.random.default_rng(1).integers(0, 255, size=(8, 10, 3)).astype(np.uint8)
    uri = rw.frame_to_data_uri(frame)
    assert uri.startswith("data:image/png;base64,")
    from PIL import Image
    import io
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))
    assert np.array_equal(decoded, frame), "PNG must round-trip the exact rendered pixels"


def test_duration_covers_n_frames_and_clamps_to_runway_range():
    assert rw.duration_for(60, 30) == 2      # 2.0s -> 2
    assert rw.duration_for(61, 30) == 3      # 2.03s -> ceil -> 3, never less than needed
    assert rw.duration_for(120, 30) == 4
    assert rw.duration_for(10, 30) == 2      # below Runway's floor -> clamp up
    assert rw.duration_for(3000, 30) == 10   # above Runway's ceiling -> clamp down


def test_generate_fn_sends_matched_conditioning_and_returns_exact_frames(tmp_path):
    calls = []
    url = _fake_video_url(tmp_path, n_frames=72, fps=24)      # 3s @24fps, like a real return
    client = _FakeClient(_Task(output=[url]), calls)
    logs = []
    gen = rw.make_runway_generate_fn("collision", model="gen4.5", seed=0,
                                     client_factory=lambda: client, log=logs.append)

    prefix = np.zeros((30, 240, 320, 3), dtype=np.uint8)
    prefix[-1, 5, 7] = (200, 30, 30)                          # a marker only on the LAST frame
    out = gen(prefix, n_frames=60)

    assert out.shape == (60, 240, 320, 3) and out.dtype == np.uint8, "exactly n_frames at the render size"
    assert len(calls) == 1
    kw = calls[0]
    assert kw["model"] == "gen4.5" and kw["ratio"] == rw.DEFAULT_RATIO and kw["seed"] == 0
    assert kw["duration"] == 2, "smallest integer seconds covering 60 frames @30fps"
    from vitals.adapters.scenario_prompts import SCENARIO_PROMPTS
    assert kw["prompt_text"] == SCENARIO_PROMPTS["collision"], "the frozen prompt, byte-identical to wan/hunyuan"
    assert kw["prompt_image"] == rw.frame_to_data_uri(rw._upscale_for_prompt(prefix[-1])), "conditioned on the LAST prefix frame only (upscaled to the gateway minimum side)"
    assert any("cost_credits=7" in l for l in logs), "per-call cost must be logged, not hidden"


def test_generate_fn_raises_on_failed_task_never_returns_fabricated_frames():
    calls = []
    client = _FakeClient(_Task(status="FAILED"), calls)
    gen = rw.make_runway_generate_fn("collision", client_factory=lambda: client)
    with pytest.raises(RuntimeError):
        gen(np.zeros((5, 240, 320, 3), dtype=np.uint8), n_frames=60)


def test_generate_fn_rejects_unregistered_scenario_prompt():
    with pytest.raises(KeyError):
        rw.make_runway_generate_fn("no_such_scenario", client_factory=lambda: None)


def test_registry_exposes_api_models_as_real_backends():
    """API models must flow into REAL_MODEL_BACKENDS so render_score_report/
    render_survival_report categorize them as real models automatically
    (both derive from the registry, never a hardcoded list)."""
    from vitals.adapters import MODEL_REGISTRY, REAL_MODEL_BACKENDS
    for name in ("runway_gen4.5", "runway_veo3.1", "runway_seedance2.5", "runway_gemini_omni", "runway_h3_max"):   # flagships only (2026-09-14)
        assert MODEL_REGISTRY[name]["access"] == "api"
        assert MODEL_REGISTRY[name]["provider"] == "runway"
        assert name in REAL_MODEL_BACKENDS
