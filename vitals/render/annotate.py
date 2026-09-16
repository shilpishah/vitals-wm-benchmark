"""Shared frame-annotation helpers (2026-08) -- extracted from `scripts/
render_annotated_comparison_video.py` when `scripts/run_model_population.py`
needed the SAME marker/border drawing to build a multi-episode grid video,
rather than duplicating `draw_border`/`draw_marker`/the per-frame loop a
second time (the same "one implementation, not two that can drift apart"
choice already made for `adapters/video_model.py::track_and_reconstruct`).
"""
from __future__ import annotations
import numpy as np


def draw_border(frame, color, thickness=8):
    """Draws a solid rectangular border flush with the frame's own edge --
    pure numpy slicing. GREEN = real, ground-truth conditioning prefix; RED
    = the model's own generated continuation."""
    out = frame.copy()
    out[:thickness, :, :] = color
    out[-thickness:, :, :] = color
    out[:, :thickness, :] = color
    out[:, -thickness:, :] = color
    return out


def draw_marker(frame, px, py, color, radius=6, thickness=2):
    """Draws a hollow circle outline at (px, py) directly into an RGB
    uint8 frame -- PIL's ImageDraw for a clean anti-aliased circle rather
    than a hand-rolled Bresenham loop."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    if px is not None and py is not None and np.isfinite(px) and np.isfinite(py):
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], outline=color, width=thickness)
    return np.asarray(img)


def annotate_episode_frames(frames, true_pos, recon_pos, cam_pos, cam_mat, fovy_deg,
                             prefix_frames, border_thickness=8):
    """frames: (T,H,W,3) uint8 -- prefix+continuation already concatenated.
    true_pos/recon_pos: (T,3) world positions for the single-object case
    (unchanged from before this docstring note), OR (T,K,3) for K>1
    (2026-09, billiards/multi-collision scoping -- previously hardcoded
    to object 0 only, both here and at every existing call site, so a
    K=2 scenario's own second object was never visualized despite R2
    already scoring it). recon_pos may contain NaN rows -- unreconstructed/
    occluded frames, skipped, per-object. Returns a NEW (T,H,W,3) array:
    green circle=true physics position, red circle=Phi-reconstructed
    position, ONE PAIR PER OBJECT when K>1, border green=real prefix /
    red=model-generated continuation. Pure post-hoc drawing -- never
    mutates `frames`."""
    from ..phi.reconstruct import project_to_pixel

    T, H, W = frames.shape[:3]
    true_pos = np.asarray(true_pos)
    recon_pos = np.asarray(recon_pos)
    if true_pos.ndim == 2:      # (T,3) -- single-object, backward compatible
        true_pos = true_pos[:, None, :]
        recon_pos = recon_pos[:, None, :]
    K = true_pos.shape[1]

    annotated = np.empty_like(frames)
    for i in range(T):
        frame = frames[i]
        for k in range(K):
            true_px = project_to_pixel(true_pos[i, k], cam_pos, cam_mat, fovy_deg, W, H)
            recon_px = (project_to_pixel(recon_pos[i, k], cam_pos, cam_mat, fovy_deg, W, H)
                        if not np.isnan(recon_pos[i, k]).any() else None)
            frame = draw_marker(frame, *(true_px or (None, None)), color=(0, 220, 0))
            frame = draw_marker(frame, *(recon_px or (None, None)), color=(230, 40, 40))
        if border_thickness > 0:
            border_color = (0, 220, 0) if i < prefix_frames else (230, 40, 40)
            frame = draw_border(frame, border_color, thickness=border_thickness)
        annotated[i] = frame
    return annotated


def tile_grid_video(episode_frames, sep=4, sep_color=(90, 90, 90)):
    """episode_frames: list of (T,H,W,3) uint8 arrays, all the SAME T/H/W
    (one project-fixed episode length/resolution, per-episode annotated
    frames from `annotate_episode_frames`) -- tiles them into one (T,
    rows*H + gaps, cols*W + gaps, 3) mosaic video, a single file showing
    many rollouts at once instead of one file per episode (2026-08,
    requested directly: "all of the videos should be being generated
    possibly in one single video that shows all of the rollouts to save
    space"). rows/cols chosen close to square via floor(sqrt(K))/ceil(K/
    rows); any leftover tiles in the grid are filled with `sep_color`
    (never left as random memory, never silently duplicated from an
    existing episode)."""
    K = len(episode_frames)
    if K == 0:
        raise ValueError("tile_grid_video needs at least one episode")
    T, H, W = episode_frames[0].shape[:3]
    for e in episode_frames:
        if e.shape[:3] != (T, H, W):
            raise ValueError(f"all episodes must share the same (T,H,W) -- got {e.shape[:3]} vs {(T, H, W)}")

    rows = max(1, int(np.floor(np.sqrt(K))))
    cols = int(np.ceil(K / rows))

    canvas_h = rows * H + (rows - 1) * sep
    canvas_w = cols * W + (cols - 1) * sep
    canvas = np.empty((T, canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[...] = sep_color

    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        y0, x0 = r * (H + sep), c * (W + sep)
        if idx < K:
            canvas[:, y0:y0 + H, x0:x0 + W, :] = episode_frames[idx]
        # else: leftover tile stays sep_color -- an incomplete grid (e.g.
        # K=5 in a 2x3 layout) is filled, not left undefined or wrapped.
    return canvas
