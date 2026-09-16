"""Shared video-array utilities for real model adapters (AGENT.md M6) --
extracted from `cosmos.py` (2026-08, when a second model, Wan2.1, needed
the identical resampling logic) rather than duplicating it per model.

`_write_video`/`_read_video` are the ONE deliberate exception to this
project's "never compressed video" rule (`render/__init__.py`'s own
`Frames` docstring) -- needed only by models (like Cosmos) whose
documented interface is a video FILE, not raw arrays. Models that return
arrays directly (like Wan2.1's own `diffusers` pipeline) never need
these.
"""
import numpy as np


class HungCallTimeout(Exception):
    pass


def call_with_timeout(fn, timeout_s):
    """Calls `fn()` with a HARD wall-clock deadline. Raises `HungCallTimeout`
    if `fn` hasn't returned within `timeout_s` -- unlike a normal Python
    call, which blocks forever if the underlying operation never returns
    AND never raises (2026-08, found directly, three separate times, on
    the exact same real n=80 Cosmos re-run: `with_one_retry` alone was not
    enough, because it only catches EXCEPTIONS -- a Modal `.remote()` call
    that hangs at the network/container level without ever raising
    anything just blocks the calling thread forever, and `with_one_retry`
    never gets the chance to retry because its first attempt never
    returns control at all).

    Implemented via a single-use worker thread + `Future.result(timeout=)`
    rather than `signal.alarm` (which only works on the main thread, not
    inside `ThreadPoolExecutor` workers, which is exactly where every real
    call in this project runs). The worker thread is NOT forcibly killed
    if it times out -- Python cannot do that safely -- it's abandoned to
    finish or fail on its own in the background; the CALLING code is what
    actually gets unblocked, which is the actual property needed (one
    stuck episode no longer blocks the other 79 in a population from ever
    being scored/reported).

    Deliberately NOT a `with ThreadPoolExecutor(...) as pool:` block --
    found DURING testing, not assumed: the context manager's own
    `__exit__` calls `shutdown(wait=True)`, which blocks until the
    submitted (still-hung) call actually finishes -- silently defeating
    this entire function's purpose (it would still block for the full
    length of the hang, just one stack frame further down). Calling
    `shutdown(wait=False)` explicitly in both branches below is what
    actually lets the calling thread return promptly."""
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        result = fut.result(timeout=timeout_s)
        pool.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)   # NOT wait=True -- see docstring above
        raise HungCallTimeout(f"did not return within {timeout_s}s -- treated as failed, "
                              f"NOT cancelled (Python can't force-kill a thread); "
                              f"the underlying call may still be running in the background")


def with_one_retry(fn, label="call", timeout_s=None):
    """Calls `fn()` (a zero-arg callable). On failure, retries EXACTLY
    ONCE, then gives up. Returns `(result, None)` on success (first or
    second attempt) or `(None, exception)` if both attempts failed --
    NEVER raises itself, so a caller running many of these concurrently
    (e.g. `run_model_population.py`'s own episode pool) can treat one
    permanently-failed call as data (excluded from the population, n
    reported honestly smaller) rather than a crash that destroys every
    OTHER already-succeeded, already-real-GPU-cost call alongside it.

    Exactly one retry, not an unbounded loop (2026-08, found directly: a
    single remote `FunctionTimeoutError` used to crash an entire n=5
    population via an unhandled exception, losing 4 already-successful
    episodes' real GPU cost along with the 1 that failed) -- mirrors this
    project's own established precedent for transient Modal failures
    (AGENT.md M2.9's Wan retry, which succeeded immediately on its second
    attempt). A SECOND consecutive failure is treated as real and
    reported, not silently retried forever.

    timeout_s: when given, each attempt is wrapped in `call_with_timeout`
    -- see that function's own docstring for why this is NOT redundant
    with the try/except below: a call that hangs without ever raising
    would otherwise block this function (and the whole calling batch)
    forever, retry logic notwithstanding."""
    call = (lambda: call_with_timeout(fn, timeout_s)) if timeout_s else fn
    try:
        return call(), None
    except Exception as exc1:
        print(f"{label}: attempt 1 FAILED -- {exc1} -- retrying once...")
        try:
            return call(), None
        except Exception as exc2:
            print(f"{label}: attempt 2 FAILED -- {exc2} -- giving up on this call")
            return None, exc2


def save_trajectory(traj, path):
    """Persists a full (prefix+continuation) reconstructed Trajectory to
    disk -- added 2026-08 so a real model's own episodes can be re-scored
    at multiple `lam`-calibrated threshold sets later (the lambda sweep,
    AGENT.md M7) WITHOUT re-spending real GPU money per level. `lam` only
    reparameterizes the reference ensemble/thresholds, never how a
    candidate trajectory is reconstructed, so one cached trajectory per
    episode is reusable across every lam level forever -- this was the
    exact gap that made the first lambda-sweep pass Truth+baselines-only
    (nothing else, per that scope's own writeup).

    Plain `.npz`, not pickle -- `names`/`meta` (a list and a dict) are the
    only non-array fields, JSON-encoded into 0-d string arrays rather than
    pickled, so the cache is readable without trusting arbitrary code
    execution on load."""
    import json
    # Soft-body state (M9): a stitched trajectory's prefix comes from the
    # true rollout and carries the vertex cloud / extras (numpy arrays,
    # not JSON); they are state-space data that the cache (pixel-measured
    # candidates) has no use for, so they are dropped here rather than
    # pickled. `shape` (Phi's descriptors) IS kept, as its own array.
    meta = {k: v for k, v in traj.meta.items() if k not in ("flex_vertices", "flex_extras")}
    arrays = dict(t=traj.t, pos=traj.pos, quat=traj.quat, present=traj.present,
                  names=json.dumps(traj.names), meta=json.dumps(meta))
    if traj.shape is not None:
        arrays["shape"] = traj.shape
    np.savez_compressed(str(path), **arrays)


def load_trajectory(path):
    """Inverse of `save_trajectory`. `allow_pickle=False` is deliberate --
    see that function's own docstring."""
    import json
    from ..types import Trajectory
    d = np.load(str(path), allow_pickle=False)
    return Trajectory(t=d["t"], pos=d["pos"], quat=d["quat"], present=d["present"],
                      names=json.loads(str(d["names"])), meta=json.loads(str(d["meta"])),
                      shape=d["shape"] if "shape" in d.files else None)


def _write_video(frames, path, fps):
    """frames: (T,H,W,3) uint8."""
    import imageio
    imageio.mimwrite(str(path), frames, fps=fps, quality=8)


def _read_video(path, default_fps=16):
    import imageio
    reader = imageio.get_reader(str(path))
    frames = np.stack([f for f in reader])
    fps = reader.get_meta_data().get("fps", default_fps)
    reader.close()
    return frames, fps


def resample_to_target(frames, src_fps, dst_fps, n_target_frames, dst_hw):
    """Nearest-frame resampling (fps) + resize (resolution), then trim or
    hold-last-frame pad to EXACTLY n_target_frames -- the same resample-in/
    resample-out discipline applied to every model per the frozen protocol
    (AGENT.md M6, items 2-3), not something invented ad hoc for any one
    model. Padding is the HONEST stand-in for "how many frames does one
    call return" whenever a model's own docs don't specify -- flagged in
    each model's own module, not hidden."""
    from PIL import Image
    src_t = frames.shape[0]
    src_times = np.arange(src_t) / src_fps
    dst_times = np.arange(n_target_frames) / dst_fps
    idx = np.clip(np.searchsorted(src_times, dst_times), 0, src_t - 1)
    picked = frames[idx]
    h, w = dst_hw
    resized = np.stack([np.asarray(Image.fromarray(f).resize((w, h))) for f in picked])
    return resized
