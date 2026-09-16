"""R5 (frame structure / invariant substrate) -- the "P1-lite" scoping
`AGENT.md` §11 records (2026-09): a classical-CV, no-new-model substitute
for the full static-keypoint/camera-pose tracker full P1 would need
(genuinely out of scope, real research effort). The claim this measures
is much weaker but still real: nothing not explicitly modeled as a
dynamic object should change, because every scenario in this project
shares a fixed camera and static, non-object scene geometry BY
CONSTRUCTION (input-normalization protocol item 10) -- that is true of
every manifest here, including ones that don't exist yet, which is what
makes this generalizable rather than scenario-specific.

Two DIFFERENT checks, both anchored at the real, rendered PREFIX's own
frame-0 ground truth (same privilege tier `frame0_mask` already uses
elsewhere in this project -- never the model's own possibly-already-wrong
first generated frame):

  Background patches -- sampled wherever frame 0's own segmentation shows
  no modeled object (`-1`, `render/__init__.py`'s own convention), a
  FIXED pixel location expected to stay put and look the same for the
  whole clip. Generalizes to a scene with a dynamic occluder for free:
  the occluder IS a modeled object, so its region is already excluded
  from "background" the same way the tracked ball already is, no special
  case needed.

  Object patches -- anchored each frame at the TRACKED object's own
  reconstructed position (already computed by `reconstruct.py`,
  projected back to pixel space via `reconstruct.project_to_pixel`),
  checking ONLY appearance (size/color/material) against its own frame-0
  appearance, never position -- position drift is already R3's job, and
  double-counting it here would manufacture a second finding out of
  evidence R3 already owns (same "one necessary property per detector"
  discipline `sigma_interpenetration`'s own docstring states for R1 vs
  R2).

Deliberately NOT using SAM2/DINO/any learned tracker -- template-match /
normalized cross-correlation only, small fixed search radius, pure numpy.
This is a WEAK, cheap substitute for full P1, not the real thing; see
this module's own known-sharp-edges notes below and `AGENT.md`'s own
scoping writeup for what it does not do.
"""
from __future__ import annotations
import numpy as np


def sample_background_patches(segmentation0, n_patches=6, patch_half=6, min_separation=20, rng=None):
    """Up to `n_patches` well-separated background-pixel locations from
    frame 0's own ground-truth segmentation (-1 = background) -- no
    scenario-specific region name anywhere, works identically for a
    scenario that doesn't exist yet. `min_separation`: Manhattan distance
    between chosen patch centers, so patches don't all crowd into one
    small area and call it "coverage." Returns a list of (cy, cx); can be
    shorter than `n_patches` (even empty) if the scene has little/no
    background -- an honest degraded-N situation, not padded out."""
    rng = rng if rng is not None else np.random.default_rng(0)
    H, W = segmentation0.shape
    bg_mask = segmentation0 == -1
    valid = np.zeros_like(bg_mask)
    valid[patch_half:H - patch_half, patch_half:W - patch_half] = True
    candidates = np.argwhere(bg_mask & valid)
    if len(candidates) == 0:
        return []
    order = rng.permutation(len(candidates))
    chosen = []
    for idx in order:
        cy, cx = candidates[idx]
        if all(abs(int(cy) - py) + abs(int(cx) - px) >= min_separation for py, px in chosen):
            chosen.append((int(cy), int(cx)))
        if len(chosen) >= n_patches:
            break
    return chosen


def _extract(gray, cy, cx, half):
    return gray[cy - half:cy + half + 1, cx - half:cx + half + 1]


def _ncc(a, b):
    """Normalized cross-correlation, [-1, 1] for two patches of identical
    shape. A flat/textureless patch (zero variance) can't meaningfully
    correlate with anything -- returns 0.0 (uninformative, neither a
    match nor a mismatch), not a divide-by-zero and not a spurious 1.0
    from two all-equal patches matching by construction."""
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.sum(a * b) / denom)


def _object_pixel_radius(depth, object_radius, height, fovy_deg):
    """Rough apparent pixel radius of a sphere of known metric radius at
    known depth, same pinhole-projection focal length `reconstruct.
    _project_raw` already uses -- used only to decide whether a
    background patch is currently covered by a modeled object, not as a
    precision measurement in its own right (see `object_appearance_drift`
    for the actual size/appearance check, which compares patches
    directly rather than trusting a derived radius number)."""
    f = height / (2 * np.tan(np.radians(fovy_deg) / 2))
    return f * object_radius / max(depth, 1e-6)


def track_background_patch(frames, cy0, cx0, patch_half=6, search_radius=4):
    """Tracks ONE background patch, anchored at its OWN fixed frame-0
    location, across `frames` (T,H,W,3) uint8 -- a small local search
    (`search_radius` px each direction) for the best-matching offset each
    frame, not a full-frame search: this is checking near-zero expected
    drift, not chasing a moving target. Returns (drift_px, ncc) per
    frame, both T-long; frame 0 is drift=0, ncc=1 by definition (compared
    to itself). A frame where the search window would run off the image
    edge is skipped (drift/ncc held at their PREVIOUS frame's value --
    same "carry the last known value forward, don't fabricate zero"
    convention `sigma_interpenetration` already uses past `Tc`)."""
    T = frames.shape[0]
    gray0 = frames[0].mean(axis=-1)
    template = _extract(gray0, cy0, cx0, patch_half)
    H, W = gray0.shape
    drift = np.zeros(T)
    ncc = np.ones(T)
    for t in range(1, T):
        gray = frames[t].mean(axis=-1)
        best_ncc, best_dy, best_dx, found = -2.0, 0, 0, False
        for dy in range(-search_radius, search_radius + 1):
            cy = cy0 + dy
            if cy - patch_half < 0 or cy + patch_half >= H:
                continue
            for dx in range(-search_radius, search_radius + 1):
                cx = cx0 + dx
                if cx - patch_half < 0 or cx + patch_half >= W:
                    continue
                score = _ncc(template, _extract(gray, cy, cx, patch_half))
                if score > best_ncc:
                    best_ncc, best_dy, best_dx, found = score, dy, dx, True
        if found:
            drift[t], ncc[t] = float(np.hypot(best_dy, best_dx)), best_ncc
        else:
            drift[t], ncc[t] = drift[t - 1], ncc[t - 1]
    return drift, ncc


def object_appearance_drift(frames, positions3d, present, cam_pos, cam_mat, fovy_deg, patch_half=6):
    """Tracks the TRACKED object's own APPEARANCE (never position -- R3
    already owns that) across `frames`, anchored each frame at `reconstruct.
    project_to_pixel(positions3d[t], ...)` -- the anchor is TRUSTED, not
    locally searched, since this checks "does it still look like it did,"
    not "is it in the right place." NaN wherever `present[t]` is False,
    the projection lands behind the camera, or the patch would run off
    the frame edge -- same "NaN, not a spurious finding" convention
    `sigma_interpenetration` already established for a missing object."""
    from .reconstruct import project_to_pixel

    T = frames.shape[0]
    H, W = frames.shape[1:3]
    sigma = np.full(T, np.nan)
    if not present[0]:
        return sigma
    px0 = project_to_pixel(positions3d[0], cam_pos, cam_mat, fovy_deg, W, H)
    if px0 is None:
        return sigma
    cx0, cy0 = int(round(px0[0])), int(round(px0[1]))
    if not (patch_half <= cy0 < H - patch_half and patch_half <= cx0 < W - patch_half):
        return sigma
    gray0 = frames[0].mean(axis=-1)
    template = _extract(gray0, cy0, cx0, patch_half)
    sigma[0] = 0.0
    for t in range(1, T):
        if not present[t]:
            continue
        px = project_to_pixel(positions3d[t], cam_pos, cam_mat, fovy_deg, W, H)
        if px is None:
            continue
        cx, cy = int(round(px[0])), int(round(px[1]))
        if not (patch_half <= cy < H - patch_half and patch_half <= cx < W - patch_half):
            continue
        gray = frames[t].mean(axis=-1)
        sigma[t] = 1.0 - _ncc(template, _extract(gray, cy, cx, patch_half))
    return sigma


def frame_consistency_sigma(frames, segmentation0, obj_positions=None, obj_present=None,
                             obj_radius=None, cam_pos=None, cam_mat=None, fovy_deg=None,
                             n_background_patches=6, patch_half=6, search_radius=4, rng=None):
    """The full R5 signal for one episode's FULL (prefix+continuation)
    frame sequence -- max over every background patch's own drift/
    appearance deviation and (when object tracking info is given) the
    tracked object's own appearance-only deviation. max, not mean: one
    bad patch is real evidence regardless of how many good ones exist
    alongside it (`sigma_interpenetration`'s own one-sided logic, reused,
    not reinvented).

    obj_positions/obj_present/obj_radius: OPTIONAL, all three together --
    the tracked object's own reconstructed (T,3) position array, (T,)
    presence mask, and known metric radius (`scene_geometry.py`'s own
    `object_radius`, the same legitimate per-scenario config tier as
    camera calibration). When given, background patches the object's own
    projected footprint currently covers are NaN'd for those frames
    ONLY -- a rolling ball crossing a sampled background patch is
    expected scene content, not a P1 violation, and scoring it as one
    would manufacture a spurious finding out of ordinary object motion.

    Returns NaN at any frame where NO patch has a defined value there
    (never silently coerced to 0 -- an undefined comparison stays
    undefined, same convention as every other detector here)."""
    T = frames.shape[0]
    H, W = frames.shape[1:3]
    bg_patches = sample_background_patches(segmentation0, n_patches=n_background_patches,
                                            patch_half=patch_half, rng=rng)

    have_object = obj_positions is not None and obj_present is not None and obj_radius is not None
    occluder_px = None
    if have_object and cam_pos is not None:
        from .reconstruct import _project_raw
        occluder_px = np.full((T, 3), np.nan)   # (cx, cy, pixel_radius) per frame
        for t in range(T):
            if not obj_present[t]:
                continue
            depth, px, py = _project_raw(obj_positions[t], cam_pos, cam_mat, fovy_deg, W, H)
            if depth <= 0:
                continue
            occluder_px[t] = (px, py, _object_pixel_radius(depth, obj_radius, H, fovy_deg))

    sigmas = []
    for cy0, cx0 in bg_patches:
        drift, ncc = track_background_patch(frames, cy0, cx0, patch_half=patch_half, search_radius=search_radius)
        s = np.maximum(drift / max(search_radius, 1), 1.0 - ncc)
        if occluder_px is not None:
            covered = np.hypot(occluder_px[:, 0] - cx0, occluder_px[:, 1] - cy0) < (occluder_px[:, 2] + patch_half)
            s = np.where(covered, np.nan, s)
        sigmas.append(s)

    if have_object:
        sigmas.append(object_appearance_drift(frames, obj_positions, obj_present, cam_pos, cam_mat, fovy_deg,
                                              patch_half=patch_half))

    if not sigmas:
        return np.full(T, np.nan)
    stacked = np.stack(sigmas)
    with np.errstate(invalid="ignore", all="ignore"):
        all_nan = np.all(np.isnan(stacked), axis=0)
        out = np.nanmax(stacked, axis=0)
    out[all_nan] = np.nan
    return out


# ---------------------------------------------------------------------------
# R5 as WIRED (2026-09-12): background change, not patch drift
# ---------------------------------------------------------------------------
# The patch-drift statistic above is ill-posed on this project's own scenes
# and was measured to be so before wiring anything (AGENT.md, Cosmos 3
# results entry): the floors are featureless, a flat patch has NCC 0 by
# `_ncc`'s own convention so `1 - ncc` saturates at 1.0, and even textured
# patches on the spotlight gradient drifted to the search-window edge on
# REAL prefix frames. The P1-lite claim itself is simpler than patch
# tracking -- "nothing not explicitly modeled as a dynamic object should
# change" -- and on a fixed-camera, static-geometry scene it can be tested
# directly: the non-object pixels of every frame should look like the
# non-object pixels of the last REAL (rendered prefix) frame, up to codec
# noise. A pan, zoom, re-framing, floor recoloring or lighting drift moves
# most of those pixels; a rolling ball moves almost none (its own footprint,
# projected from the reconstructed trajectory and dilated, is excluded).
#
# The null distribution is measured, not assumed: rendered reference
# rollouts are passed through the same mp4 write/read the model outputs go
# through, the statistic is computed on them identically, and the threshold
# is the same whole-path LOO calibration every other channel uses. State-
# space baselines have no pixels and are simply not scored on R5.
#
# Two statistics live here. `global_motion_sigma` is the CHANNEL (R5):
# estimated image motion of the static geometry, in pixels, relative to
# the first generated frame. `background_change_sigma` (photometric, in
# intensity levels) is a DIAGNOSTIC only -- see make_frame_invariance_
# channel's docstring for why the first live calibration forced that split.

def object_footprint_mask(pos3d, present, cam_pos, cam_mat, fovy_deg, width, height, obj_radius,
                          dilate_px=4):
    """(T,H,W) bool: True where a modeled object's projected disc (radius
    from its known metric radius and depth, plus `dilate_px`) covers the
    pixel. pos3d (T,K,3), present (T,K). NaN/absent -> no footprint that
    frame (the pixel then counts as background -- if the object was in
    fact there, R1 owns that evidence, not R5)."""
    from .reconstruct import _project_raw
    T, K = present.shape
    yy, xx = np.mgrid[0:height, 0:width]
    out = np.zeros((T, height, width), bool)
    for t in range(T):
        for k in range(K):
            if not present[t, k] or not np.all(np.isfinite(pos3d[t, k])):
                continue
            depth, px, py = _project_raw(pos3d[t, k], cam_pos, cam_mat, fovy_deg, width, height)
            if depth <= 0:
                continue
            r = _object_pixel_radius(depth, obj_radius, height, fovy_deg) + dilate_px
            out[t] |= (xx - px) ** 2 + (yy - py) ** 2 <= r * r
    return out


def background_change_sigma(frames, footprint, ref_idx):
    """R5 per-frame statistic: mean absolute grayscale change of all
    background (non-footprint) pixels between frame t and the anchor
    frame `ref_idx` (the last real prefix frame), in intensity levels
    (0-255). Pixels inside EITHER frame's footprint are excluded so the
    ball's own travel never counts. NaN if a frame has no background
    pixels left (cannot happen on these scenes; kept for the convention).
    frames (T,H,W,3) uint8; footprint (T,H,W) bool."""
    g = frames.astype(np.float32).mean(axis=-1)
    anchor = g[ref_idx]
    anchor_fp = footprint[ref_idx]
    out = np.full(frames.shape[0], np.nan)
    for t in range(frames.shape[0]):
        bg = ~(footprint[t] | anchor_fp)
        if bg.any():
            out[t] = float(np.abs(g[t][bg] - anchor[bg]).mean())
    return out


def _grad_mag(g):
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def _phase_correlation_shift(a, b, mask=None):
    """Sub-pixel (dx, dy) that maps patch `a` onto patch `b` by plain
    (NOT whitened) cross-correlation of the two patches' GRADIENT
    MAGNITUDE images, Hann-windowed, with parabolic peak refinement, and
    the normalized correlation at the peak in [0,1] as confidence.

    Gradient magnitude makes the estimate invariant to brightness offset
    and gain (generated video re-renders every frame with its own shading
    and flicker; only GEOMETRY is evidence of a moved camera). Plain
    rather than phase correlation, found on the first real clips
    (2026-09-12): phase correlation whitens the spectrum, so on a nearly
    featureless quadrant a small moving blob -- the ball's SHADOW, which
    the footprint mask does not cover -- dominates and the estimate
    tracks the shadow (18 px median "motion" on a fixed-camera Cosmos 3
    clip whose photometric drift was 0.4 levels). With energy-weighted
    correlation the long static edges (wall, spotlight rim) decide the
    peak and a small blob cannot. Both inputs float32 (h, w), same shape.
    `mask` (h, w) bool: pixels whose gradient energy is ZEROED in both
    images -- the modeled objects' footprints. Zeroing, not filling with
    the anchor's pixels: filling makes that region identical in both
    frames and manufactures a confident zero-shift peak (found on the
    synthetic zoom test)."""
    a = _grad_mag(a); b = _grad_mag(b)
    if mask is not None:
        a = np.where(mask, 0.0, a); b = np.where(mask, 0.0, b)
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    aw = (a - a.mean()) * win
    bw = (b - b.mean()) * win
    fa = np.fft.fft2(aw); fb = np.fft.fft2(bw)
    r = np.real(np.fft.ifft2(fa * np.conj(fb)))
    norm = float(np.sqrt(np.sum(aw * aw) * np.sum(bw * bw)))
    if norm < 1e-9:
        return 0.0, 0.0, 0.0
    r = r / norm
    py, px = np.unravel_index(np.argmax(r), r.shape)
    peak = float(r[py, px])

    def refine(vals):   # parabolic interpolation around the peak, 1-D
        c0, c1, c2 = vals
        denom = c0 - 2 * c1 + c2
        return 0.0 if abs(denom) < 1e-12 else 0.5 * (c0 - c2) / denom
    dy = py + refine((r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px]))
    dx = px + refine((r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w]))
    if dy > h / 2: dy -= h
    if dx > w / 2: dx -= w
    return float(dx), float(dy), max(0.0, min(1.0, peak))


R5_FOOTPRINT_DILATE_PX = 14   # covers the object's own cast shadow too, not just its disc (measured on real clips)


def global_motion_sigma(frames, footprint, ref_idx, min_texture_std=2.0, min_peak=0.25):
    """R5 per-frame statistic AS WIRED: the largest estimated image
    motion, in pixels, among the four quadrants of frame t relative to
    the anchor frame `ref_idx` -- with every modeled object's footprint
    (in either frame) filled with the anchor's own pixels first, so the
    ball's travel is invisible to the estimate. A camera pan moves all
    four quadrants alike; a zoom or dolly moves them apart; a tilt or
    re-framing shears them; a static camera moves none. Quadrants with
    no texture (std below `min_texture_std`, e.g. the black surround) or
    no confident correlation peak are NaN and ignored; a frame with no
    usable quadrant is NaN (undefined, never 0). Frames before ref_idx
    are 0 by definition when the anchor is the first generated frame:
    they are real renders."""
    g = frames.astype(np.float32).mean(axis=-1)
    T, H, W = g.shape
    anchor = g[ref_idx].copy()
    hh, hw = H // 2, W // 2
    quads = [(slice(0, hh), slice(0, hw)), (slice(0, hh), slice(hw, W)), (slice(hh, H), slice(0, hw)), (slice(hh, H), slice(hw, W))]
    out = np.full(T, np.nan)
    for t in range(T):
        if t < ref_idx:
            out[t] = 0.0
            continue
        cur = g[t]
        fill = footprint[t] | footprint[ref_idx]
        best = np.nan
        for ys, xs in quads:
            qa, qb, qm = anchor[ys, xs], cur[ys, xs], fill[ys, xs]
            if qa[~qm].std() < min_texture_std if (~qm).any() else True:
                continue
            dx, dy, peak = _phase_correlation_shift(qa, qb, mask=qm)
            if peak < min_peak:
                continue
            m = float(np.hypot(dx, dy))
            best = m if np.isnan(best) else max(best, m)
        out[t] = best
    return out


R5_MIN_PX = 1.0   # threshold floor: sub-pixel estimated camera motion on generated video is estimator noise, not a violation


def codec_roundtrip(frames, fps, quality=8):
    """Write/read the frames through the SAME imageio/ffmpeg mp4 path
    `_write_video` uses for model outputs, so reference frames carry the
    same compression noise a model's decoded frames do. Returns (T,H,W,3)
    uint8 of the same length (trimmed/padded to be safe)."""
    import tempfile, os
    import imageio
    fd, path = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    try:
        imageio.mimwrite(path, frames, fps=fps, quality=quality)
        reader = imageio.get_reader(path)
        back = np.stack([f for f in reader]); reader.close()
    finally:
        os.unlink(path)
    T = frames.shape[0]
    if back.shape[0] >= T:
        return back[:T]
    return np.concatenate([back, np.repeat(back[-1:], T - back.shape[0], axis=0)])


def make_frame_invariance_channel(refs, scene_path, camera, obj_radius, n_frames, prefix_len, fps,
                                  height=240, width=320, alpha=0.01, log=print):
    """Builds R5 for one scenario. Renders the reference ensemble (first
    `n_frames` of each), codec-roundtrips it, computes `global_motion_
    sigma` per reference anchored at frame `prefix_len` (the first frame
    a model generates -- see below), calibrates theta_R5 with the SAME
    `estimate_threshold` every other channel uses (LOO over M
    references), and floors it at R5_MIN_PX. Returns (theta_R5
    (n_frames,), sigma_fn, diag_fn):
      sigma_fn(frames, recon_pos, recon_present, cam_pos, cam_mat,
               fovy_deg) -> (n_frames,) R5 statistic for one candidate
      diag_fn(...) same args -> dict of photometric diagnostics (see
               `background_change_sigma`): NOT a channel.

    Why the anchor is the FIRST GENERATED frame, not the last real one
    (found on the first live calibration, 2026-09-12, before this was
    ever scored on a model): rendered references through the codec differ
    from each other by ~0.01 intensity levels, so any statistic anchored
    at the real prefix calibrates to ~0.02 -- and a generative model's
    first frame is never pixel-identical to a MuJoCo render (its own
    shading, blur, color), so such a channel would fire on EVERY model at
    t=1.0s and, ranked first in precedence, swallow every other
    diagnosis. Anchoring inside the model's own output asks the P1
    question that is actually answerable from pixels: does the model's
    own scene stay put while its objects move. The real->generated
    boundary is reported by diag_fn as a photometric jump, not judged.

    Why geometric (phase correlation), not photometric: generated video
    re-renders every frame with its own flicker; only motion of the
    static geometry is evidence of a moved camera, and phase correlation
    is invariant to brightness/gain. Photometric drift (a floor turning
    white) is a real failure the diagnostics record, but its null
    distribution cannot be measured from renders, so it is not a
    calibrated channel yet."""
    from ..render.mujoco_renderer import MujocoRenderer
    from ..adapters.base import prefix_of
    from ..detect.thresholds import estimate_threshold

    renderer = MujocoRenderer(scene_path, height=height, width=width)
    sig_by_id = {}
    for i, ref in enumerate(refs):
        clip = prefix_of(ref, n_frames)
        fr, gt = renderer.render(clip, cameras=camera)
        frames = codec_roundtrip(fr.rgb, fps)
        fp = object_footprint_mask(clip.pos, clip.present, gt.cam_pos, gt.cam_mat, gt.fovy_deg,
                                   width, height, obj_radius, dilate_px=R5_FOOTPRINT_DILATE_PX)
        sig_by_id[id(ref)] = global_motion_sigma(frames, fp, prefix_len)
        if i == 0:
            s = sig_by_id[id(ref)]
            log(f"R5 calibration: reference 0 estimated motion median={np.nanmedian(s[prefix_len:]):.3f} px, "
                f"max={np.nanmax(s[prefix_len:]):.3f} px (render+codec null)")
    theta = estimate_threshold(refs, lambda traj, others: sig_by_id[id(traj)], alpha=alpha)
    theta = np.where(np.isfinite(theta), np.maximum(theta, R5_MIN_PX), theta)

    def sigma_fn(frames, recon_pos, recon_present, cam_pos, cam_mat, fovy_deg):
        fp = object_footprint_mask(recon_pos, recon_present, cam_pos, cam_mat, fovy_deg, width, height, obj_radius,
                                   dilate_px=R5_FOOTPRINT_DILATE_PX)
        return global_motion_sigma(np.asarray(frames), fp, prefix_len)

    def diag_fn(frames, recon_pos, recon_present, cam_pos, cam_mat, fovy_deg):
        fr = np.asarray(frames)
        fp = object_footprint_mask(recon_pos, recon_present, cam_pos, cam_mat, fovy_deg, width, height, obj_radius,
                                   dilate_px=R5_FOOTPRINT_DILATE_PX)
        vs_real = background_change_sigma(fr, fp, prefix_len - 1)     # anchored at the last REAL frame
        vs_gen = background_change_sigma(fr, fp, prefix_len)          # anchored at the first generated frame
        return dict(photometric_boundary_jump=float(vs_real[prefix_len]) if prefix_len < len(vs_real) else float("nan"),
                    photometric_drift_median=float(np.nanmedian(vs_gen[prefix_len:])),
                    photometric_drift_max=float(np.nanmax(vs_gen[prefix_len:])))

    return theta, sigma_fn, diag_fn
