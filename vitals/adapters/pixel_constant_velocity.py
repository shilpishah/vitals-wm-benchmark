"""A second, genuinely different `generate_fn` for `VideoWorldModel`
(AGENT.md M6) -- built specifically to validate that the adapter/pipeline
is actually model-agnostic, not just in theory. Deliberately as UNLIKE
`cosmos.py`'s own implementation style as possible while satisfying the
identical contract:

               cosmos.py                    this file
  process       subprocess -> external CLI    pure numpy, in-process
  I/O           writes/reads video FILES      operates on arrays directly
  config        needs a per-scenario prompt    needs NOTHING scenario-specific
  dependency    imageio/ffmpeg, a real         none beyond numpy (already
                cosmos-predict2 install         a hard dependency of this
                                                 whole project)

If both plug into the exact same `VideoWorldModel` and produce a Trajectory
through the exact same downstream pipeline (segmentation/reconstruction/
detection/scoring) with zero code changes anywhere else, that IS the
architecture's model-agnosticism, demonstrated rather than asserted.

This is also a REAL, reusable artifact, not a throwaway test double -- the
pixel-space analog of `adapters/base.py`'s own `ConstantVelocity` baseline
(constant STATE velocity) and exactly as principled: extrapolate the
tracked object's own recent PIXEL motion linearly forward, using nothing
but the rendered RGB frames it's given (no privileged state, no scenario
config, matching the input-normalization protocol's own rule 1). Same
diagnostic role GATE 3 already established for the state-space version:
this should track real motion reasonably while it's genuinely close to
constant pixel velocity, and fall apart the moment true dynamics deviate
(a bounce, an occlusion, a ramp transition) -- a short validity interval
there is the CORRECT, expected outcome, not a bug in this baseline.
"""
import numpy as np


def _find_blob(frame, rgb_target, tol):
    """Nearest-color pixel mask + its centroid, or None if nothing within
    tolerance is visible in this frame -- pure color thresholding on the
    RGB pixels this function is handed, nothing else."""
    diff = np.abs(frame.astype(np.int16) - np.array(rgb_target, dtype=np.int16)).sum(axis=-1)
    mask = diff < tol
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean()), mask


def _background_color(frame, exclude_mask):
    """Mode color of the frame OUTSIDE the object's own mask -- used to
    paint over the object's old position so a translated copy doesn't
    leave a ghost behind. Self-contained: no scene-specific floor-color
    constant hardcoded here, estimated fresh from whatever frame it's
    given, same "only use the pixels you're handed" discipline as
    `_find_blob`."""
    bg_pixels = frame[~exclude_mask]
    if bg_pixels.size == 0:
        return np.array([128, 128, 128], dtype=np.uint8)
    # mode per channel via a coarse histogram -- robust to a few stray
    # pixels (shadows, anti-aliasing) that a plain mean would blur toward
    vals, counts = np.unique(bg_pixels.reshape(-1, 3), axis=0, return_counts=True)
    return vals[np.argmax(counts)]


def make_pixel_constant_velocity_generate_fn(object_rgb=(204, 26, 26), tol=60, patch_radius=12):
    """Builds the generate_fn closure. object_rgb/tol: color-threshold
    parameters for locating the tracked object in raw pixels -- this
    project's own ball color (scenes/*.xml: rgba="0.8 0.1 0.1 1") by
    default, but this file (unlike cosmos.py) needs no scenario name, no
    camera, no manifest lookup -- it only ever touches the pixels it's
    handed, which is the whole point of this exercise."""
    def generate_fn(prefix_frames, n_frames):
        H, W = prefix_frames.shape[1:3]
        last = prefix_frames[-1]
        c1 = _find_blob(prefix_frames[-2], object_rgb, tol) if len(prefix_frames) >= 2 else None
        c2 = _find_blob(last, object_rgb, tol)

        out = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
        if c1 is None or c2 is None:
            # Honest "no information" fallback -- object not detectable in
            # the pixels available, same discipline as CopyLastState rather
            # than fabricating a motion estimate from nothing.
            out[:] = last
            return out

        x1, y1, _ = c1
        x2, y2, mask2 = c2
        vx, vy = x2 - x1, y2 - y1
        background = _background_color(last, mask2)
        yy, xx = np.mgrid[0:H, 0:W]

        for i in range(n_frames):
            cx, cy = x2 + vx * (i + 1), y2 + vy * (i + 1)
            frame = np.full((H, W, 3), background, dtype=np.uint8)
            if 0 <= cx < W and 0 <= cy < H:
                circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= patch_radius ** 2
                frame[circle] = object_rgb
            out[i] = frame
        return out

    return generate_fn
