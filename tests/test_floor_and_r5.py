"""2026-09-12: the VI50-floor summaries (RMVT, S(t) at horizons) and the
wired R5 background-change channel. Checked against closed forms and
against the behaviors they exist to separate, not against themselves.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from vitals.types import Event
from vitals.stats.survival import restricted_mean_validity, survival_at, bootstrap_rmvt, validity_interval
from vitals.phi.frame_consistency import background_change_sigma, object_footprint_mask, codec_roundtrip


def _ev(times, censored=None):
    censored = censored or [False] * len(times)
    return [Event(time=t, risk=None if c else "R3", censored=c) for t, c in zip(times, censored)]


def test_rmvt_matches_hand_computed_step_area():
    # 4 episodes: fail at 1, 2, censored at 3 (t_max=3), fail at 2.5
    # KM: S=1 on [0,1), 0.75 on [1,2), 0.5 on [2,2.5), 0.25 on [2.5,3]
    ev = _ev([1.0, 2.0, 3.0, 2.5], censored=[False, False, True, False])
    area = 1 * 1.0 + 0.75 * 1.0 + 0.5 * 0.5 + 0.25 * 0.5
    assert abs(restricted_mean_validity(ev, 3.0) - area) < 1e-9
    assert abs(restricted_mean_validity(ev, 2.0) - (1.0 + 0.75)) < 1e-9     # truncation at t_max


def test_rmvt_ranks_populations_that_share_a_vi50_at_the_floor():
    """The floor problem itself: both populations have VI50 = 1.03 (a
    point), but one keeps 40% of episodes alive to the end. VI50 cannot
    tell them apart; RMVT and S(t) can."""
    a = _ev([1.03] * 10)
    b = _ev([1.03] * 6 + [3.0] * 4, censored=[False] * 6 + [True] * 4)
    assert validity_interval(a) == validity_interval(b) == 1.03
    assert restricted_mean_validity(b, 3.0) > restricted_mean_validity(a, 3.0) + 0.7
    assert survival_at(a, [2.0]) == [0.0] and abs(survival_at(b, [2.0])[0] - 0.4) < 1e-9
    lo, hi = bootstrap_rmvt(b, 3.0, n_boot=200)
    assert lo <= restricted_mean_validity(b, 3.0) <= hi


def test_survival_at_reads_the_step_function_including_before_first_event():
    ev = _ev([1.0, 2.0])
    assert survival_at(ev, [0.5, 1.0, 1.5, 2.0, 9.0]) == [1.0, 0.5, 0.5, 0.0, 0.0]


def _scene(T=20, H=48, W=64, seed=0):
    """A static textured background with a moving bright disc on it."""
    rng = np.random.default_rng(seed)
    # Structured background (a gradient, stripes, a few squares) plus mild
    # noise -- the estimator keys on edge structure; pure i.i.d. noise
    # decorrelates under any zoom and is not a scene this project has.
    yy, xx = np.mgrid[0:H, 0:W]
    base = 70 + 30 * np.sin(xx / 5.0) + 20 * (yy // 8 % 2) + 0.3 * xx
    for (cy, cx) in ((H // 4, W // 4), (3 * H // 4, 3 * W // 4), (H // 4, 3 * W // 4)):
        base[cy - 4:cy + 4, cx - 4:cx + 4] += 60
    bg = np.clip(base + rng.normal(0, 2, size=(H, W)), 0, 255)
    bg = np.repeat(bg[..., None], 3, axis=-1).astype(np.uint8)
    frames = np.repeat(bg[None], T, axis=0).copy()
    fp = np.zeros((T, H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    for t in range(T):
        cx, cy = 10 + 2 * t, H // 2
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= 16
        frames[t][disc] = 230
        fp[t] = (xx - cx) ** 2 + (yy - cy) ** 2 <= 36   # footprint dilated past the disc
    return frames, fp


def test_background_change_is_near_zero_for_a_moving_object_on_a_static_scene():
    frames, fp = _scene()
    s = background_change_sigma(frames, fp, ref_idx=5)
    assert np.all(np.isfinite(s)) and s.max() < 1e-6, "the ball's own travel must never count as background change"


def test_background_change_fires_on_a_pan_a_zoom_and_a_recolor():
    frames, fp = _scene()
    s0 = background_change_sigma(frames, fp, 5)
    panned = np.roll(frames, 3, axis=2)                                  # 3-px camera pan from frame 10 on
    mixed = frames.copy(); mixed[10:] = panned[10:]
    s_pan = background_change_sigma(mixed, fp, 5)
    assert s_pan[:10].max() < 1e-6 and s_pan[10:].min() > 5.0
    recolored = frames.copy(); recolored[10:] = np.clip(frames[10:].astype(int) + 40, 0, 255)
    assert background_change_sigma(recolored, fp, 5)[10:].min() > 30.0
    from PIL import Image
    zoomed = frames.copy()
    for t in range(10, frames.shape[0]):
        big = np.asarray(Image.fromarray(frames[t]).resize((frames.shape[2] * 5 // 4, frames.shape[1] * 5 // 4)))
        h, w = frames.shape[1:3]; y0, x0 = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
        zoomed[t] = big[y0:y0 + h, x0:x0 + w]
    assert background_change_sigma(zoomed, fp, 5)[10:].min() > 5.0
    assert s0.max() < 1e-6


def test_global_motion_sigma_is_the_channel_pan_zoom_fire_recolor_does_not():
    """R5 as wired: geometric, anchored at the first generated frame
    (ref_idx=10 here), brightness-invariant. Static -> ~0 px; a 3-px pan
    -> ~3 px; a 25% zoom -> several px (quadrants move apart); a +40
    recolor -> ~0 px (that is the photometric diagnostic's job)."""
    from vitals.phi.frame_consistency import global_motion_sigma
    frames, fp = _scene(T=24, H=64, W=96)
    s = global_motion_sigma(frames, fp, 10)
    assert np.all(s[:10] == 0.0) and np.nanmax(s[10:]) < 0.3, s
    panned = frames.copy(); panned[16:] = np.roll(frames[16:], 3, axis=2)
    sp = global_motion_sigma(panned, fp, 10)
    assert np.nanmax(sp[10:16]) < 0.3 and 2.5 < np.nanmin(sp[16:]) < 3.5, sp
    from PIL import Image
    zoomed = frames.copy()
    for t in range(16, frames.shape[0]):
        big = np.asarray(Image.fromarray(frames[t]).resize((frames.shape[2] * 5 // 4, frames.shape[1] * 5 // 4)))
        h, w = frames.shape[1:3]; y0, x0 = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
        zoomed[t] = big[y0:y0 + h, x0:x0 + w]
    sz = global_motion_sigma(zoomed, fp, 10)
    assert np.nanmin(sz[16:]) > 2.0, sz
    recolored = frames.copy(); recolored[16:] = np.clip(frames[16:].astype(int) + 40, 0, 255)
    assert np.nanmax(global_motion_sigma(recolored, fp, 10)[10:]) < 0.3


def test_codec_roundtrip_adds_only_small_noise_and_keeps_length():
    frames, fp = _scene(T=12)
    back = codec_roundtrip(frames, fps=30)
    assert back.shape == frames.shape and back.dtype == np.uint8
    noise = background_change_sigma(np.concatenate([frames[:1], back[1:]]), fp, 0)
    assert 0 < np.nanmean(noise[1:]) < 6.0, f"codec noise should be a few intensity levels, got {np.nanmean(noise[1:]):.2f}"


def test_object_footprint_mask_covers_the_projected_object_and_nothing_when_absent():
    from vitals.phi.reconstruct import _project_raw
    H, W = 240, 320
    cam_pos = np.array([0.0, -8.0, 4.0]); fwd = -cam_pos / np.linalg.norm(cam_pos)
    right = np.array([1.0, 0.0, 0.0]); up = np.cross(right, fwd)
    cam_mat = np.stack([right, up, -fwd], axis=1)      # columns: right, up, -forward (MuJoCo convention)
    pos = np.array([[[0.0, 0.0, 0.15]], [[0.0, 0.0, 0.15]]]); present = np.array([[True], [False]])
    fp = object_footprint_mask(pos, present, cam_pos, cam_mat, 45.0, W, H, 0.15)
    depth, px, py = _project_raw(pos[0, 0], cam_pos, cam_mat, 45.0, W, H)
    assert depth > 0 and fp[0, int(round(py)), int(round(px))], "the projected center must be inside the footprint"
    assert fp[0].sum() > 20 and fp[1].sum() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
