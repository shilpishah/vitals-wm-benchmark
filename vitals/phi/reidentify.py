"""DINO-based appearance re-identification, wrapping segmentation.py's SAM2
tracker. Fixes the occlusion-recovery cliff measured empirically at
~22-27 frames (~1s) on occlusion_corridor: SAM2's own memory-attention
tracker discards a lost object permanently past that window, with zero
chance of recovery on its own (scripts/validate_reidentification.py has
the sweep that found this).

Design, and why (checked against 8 current video-world-model benchmark
papers -- none use this specific combination for occlusion recovery):
  - DINOv2, not CLIP. CLIP is explicitly excluded from the measurement
    path (AGENT.md 3.2, "CLIP-family excluded; DINO-family fine") -- its
    language supervision gives it the same unbounded-error-surface problem
    as a VLM judge. DINOv2 is purely self-supervised on images.
  - An off-the-shelf, independently-pretrained DINOv2 checkpoint, not a
    model trained on our own synthetic renders. Training our own would
    repeat the exact circularity trap AGENT.md already names for the P3
    relation head (10): it would memorize this repo's own narrow rendering
    domain rather than learning a generalizable notion of appearance, and
    its "error floor" would be validated against the same distribution it
    was fit to.
  - Mask-pooled patch tokens, not the CLS token. CLS is a global summary of
    the whole crop; patch tokens restricted to the object's own mask give
    an appearance descriptor of THAT OBJECT.
  - Candidates from SAM2's own class-agnostic automatic mask generator,
    matched by appearance -- not blind periodic re-prompting. Re-prompting
    without appearance verification risks locking onto a shadow, a render
    artifact, or (in busier future scenes) a different object entirely,
    which corrupts P2 exactly where P2 is supposed to be measuring
    identity, not just presence.
  - The similarity threshold is estimated from the reference ensemble
    (thresholds.py), never hardcoded -- same invariant as every other
    detector in this repo (3.3).

Lazy torch/sam2 imports throughout, same pattern as physics/runner.py and
segmentation.py, so this module is importable without CUDA installed.

UPDATED (see motion_prior.py): DINO appearance similarity was measured to
get 0/N matches on occlusion_corridor's hard velocity band -- root-caused
to genuinely low appearance separability (low-texture, rotating, rolling
object), not fixable by recalibration. motion_prior.py's kinematic
extrapolation solves the SAME hard cases cleanly (10/10 correct, ~3px
localization) because this scenario's motion is far more predictable than
its appearance is distinguishable. track_with_reidentification below now
tries the physics prior FIRST and falls back to DINO-appearance search
only when a track has too little pre-occlusion history to fit one (see
MIN_FIT_FRAMES) -- DINO stays the relevant signal for a future K>1 scene,
where position alone can't disambiguate which of several plausible nearby
objects is the right one, but for today's K=1 scenarios it is secondary,
not primary. See AGENT.md for the camera-calibration / near-constant-depth
assumptions motion_prior.py's approach depends on and does not yet handle.
"""
from __future__ import annotations
import os
import pathlib
from dataclasses import dataclass, field
import numpy as np

PATCH_SIZE = 14          # DINOv2 ViT-S/14
CROP_SIZE = 224           # 224 / 14 = 16x16 patch grid
CROP_PADDING = 0.4        # fraction of bbox size added as margin on each side
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Shared between BOTH tiers (physics gating and DINO fallback) -- see
# track_with_reidentification's search loop. A candidate near a position
# already rejected earlier in the same gap is suspected to be a persistent
# static feature, not a reappearance. Originally only gated the physics
# tier; DINO had no equivalent check at all, which is exactly what let it
# accept a known-bad static candidate (similarity 0.41, just over
# threshold) immediately after the physics tier had already rejected that
# same location four times in the same gap (AGENT.md defect #18).
STATIC_SUSPECT_RADIUS_PX = 5.0
TIGHT_GATE_PX = 4.0
# Size-consistency gate on re-identification candidates (2026-09-14):
# a candidate mask whose area is outside [1/RATIO, RATIO] x the last
# visible mask's area is not the same object. Found on soft_drop's GATE 2:
# after a planted vanish on a flat scene (no occluder, nothing else to
# find) the physics tier accepted a 42,800-px proposal -- the FLOOR, 56%
# of the frame, 60x the body -- because its centroid happened to sit
# within the position gate, and DINO then confirmed it at similarity 1.0
# (the cached embedding had been re-cached from that same acceptance).
# Existence stayed True on a floor track, and the size explosion was
# reported as R6 conservation instead of R1. 4x is generous for any real
# re-appearance (a ball at a different depth, a body squashed on impact:
# 1.2-1.5x) and rejects only the absurd; applied to BOTH tiers, before
# scoring, so the next-nearest plausible candidate is still considered.
REID_SIZE_RATIO_MAX = 4.0


def _size_consistent(candidates, last_mask, ratio=REID_SIZE_RATIO_MAX):
    if last_mask is None:
        return candidates
    ref_area = float(np.asarray(last_mask).sum())
    if ref_area <= 0:
        return candidates
    return [c for c in candidates
            if ref_area / ratio <= float(np.asarray(c["segmentation"]).sum()) <= ref_area * ratio]


def load_dino(model_name="dinov2_vits14", device="cuda"):
    import torch
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    return model.to(device).eval()


def load_mask_generator(checkpoint="facebook/sam2.1-hiera-tiny", device="cuda",
                        points_per_side=8, points_per_batch=8,
                        pred_iou_thresh=0.5, stability_score_thresh=0.5, min_mask_region_area=0):
    """A separate image-mode SAM2 instance for class-agnostic re-detection
    candidates during search mode -- distinct from the video predictor,
    which only propagates an already-prompted object, it doesn't discover
    new ones. points_per_side/points_per_batch defaults lowered well below
    SAM2's own (32, 64): this instance shares an 8GB GPU with the video
    predictor and DINO, our scenes are simple/sparse (few real objects),
    and points_per_batch in particular controls PEAK memory during a
    single generate() call, which is what was actually exhausting the card,
    not the models' resident size (all three loaded together measure well
    under 1GB).

    pred_iou_thresh/stability_score_thresh defaulted to 0.5/0.5, NOT SAM2's
    own defaults (~0.88/0.95) -- this is not a stylistic loosening, it's a
    fix for a real, previously-shipped bug found via an end-to-end search
    (AGENT.md build-sequence step 27): with SAM2's stricter defaults, the
    automatic mask generator proposed only 4 candidates per frame on
    occlusion_corridor, all large background regions (2800-42000px), and
    NEVER proposed the actual ball (~65-70px) at any frame checked from
    reappearance onward -- confirmed directly, not inferred. No candidate-
    scoring or gating logic downstream can recover from a search that never
    sees the right candidate. This was already known to matter for THIS
    object (a small, low-texture sphere) from the original DINO-calibration
    work earlier in this project, but `load_mask_generator`'s own defaults
    never carried that finding forward -- every caller that didn't
    explicitly override these two parameters was silently exposed to it,
    including `track_with_reidentification`'s own real, shipped usage."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    return SAM2AutomaticMaskGenerator.from_pretrained(
        checkpoint, device=device, points_per_side=points_per_side, points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area)


def _crop_and_resize(frame_rgb, mask, pad_frac=CROP_PADDING, size=CROP_SIZE):
    """Padded bounding-box crop around mask, resized to (size, size).
    Returns (crop_rgb, crop_mask) or (None, None) if mask is empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    H, W = mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
    x0, x1 = max(0, x0 - pad_x), min(W, x1 + pad_x + 1)
    y0, y1 = max(0, y0 - pad_y), min(H, y1 + pad_y + 1)

    from PIL import Image
    crop = Image.fromarray(frame_rgb[y0:y1, x0:x1]).resize((size, size), Image.BILINEAR)
    mask_crop = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255).resize((size, size), Image.NEAREST)
    return np.array(crop), np.array(mask_crop) > 0


def embed_region(dino_model, frame_rgb, mask, device="cuda"):
    """Mask-pooled, L2-normalized DINOv2 patch-token embedding of the
    object in `mask`. None if the mask is empty."""
    import torch
    import torch.nn.functional as F

    crop_rgb, crop_mask = _crop_and_resize(frame_rgb, mask)
    if crop_rgb is None or crop_mask.sum() == 0:
        return None

    x = torch.from_numpy(crop_rgb).float().permute(2, 0, 1)[None] / 255.0
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    x = ((x - mean) / std).to(device)

    with torch.inference_mode():
        out = dino_model.forward_features(x)
    patches = out["x_norm_patchtokens"][0]                     # (n_patches, C)
    grid = int(round(patches.shape[0] ** 0.5))                  # 16 for a 224-px, patch-14 input
    patches = patches.reshape(grid, grid, -1)

    mask_t = torch.from_numpy(crop_mask).float()[None, None]
    mask_small = F.adaptive_avg_pool2d(mask_t, (grid, grid))[0, 0] > 0.5
    if mask_small.sum() == 0:                                   # object smaller than one patch cell
        mask_small = F.adaptive_avg_pool2d(mask_t, (grid, grid))[0, 0] > 0

    selected = patches[mask_small.to(patches.device)]
    if selected.shape[0] == 0:
        return None
    emb = selected.mean(dim=0)
    emb = emb / emb.norm().clamp_min(1e-8)
    return emb.detach().cpu().numpy()


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@dataclass
class ReidentifyResult:
    masks: dict = field(default_factory=dict)      # {frame_idx: (H,W) bool} -- primary object (obj_id=1)
    reid_events: list = field(default_factory=list)
    # [(frame_idx, score, matched, method), ...] -- score is a pixel distance
    # (smaller is better) when method=="physics", a cosine similarity
    # (larger is better) when method=="dino". Don't compare scores across
    # methods directly, they're on different scales/directions.
    masks_secondary: dict = field(default_factory=dict)
    # {frame_idx: (H,W) bool} -- SECOND tracked object (obj_id=2), only
    # populated when track_with_reidentification is given
    # frame0_mask_secondary (AGENT.md M5.6, the scoped phase-2 multi-object
    # design). Deliberately gets NO search-and-reprompt machinery of its
    # own: whatever SAM2's own propagation naturally tracks (including its
    # own memory-window self-recovery for gaps under ~1s, per
    # segmentation.py's own docstring) is all this object ever gets. If
    # SAM2 loses it for longer than that and it never comes back on its
    # own, it just stays lost for the rest of the clip -- no reidentify.py
    # events are ever generated for it (reconstruct.py's own
    # _existence_mask already treats an empty reid_events list correctly:
    # every gap needs forgiveness-window or self-recovery to close, never
    # search-based closure). Reusing that existing logic unmodified, rather
    # than building a second search loop, is the entire point of this
    # scope -- see M5.6's own "scoping decision" note for why full
    # simultaneous-loss support was deliberately deferred.
    masks_tertiary: dict = field(default_factory=dict)
    # {frame_idx: (H,W) bool} -- THIRD tracked object (obj_id=3, 2026-09,
    # billiards/multi-collision scoping), populated when track_with_
    # reidentification is given frame0_mask_tertiary. Additive, not a
    # generalization of masks_secondary to an N-object list: deliberately
    # mirrors masks_secondary's own exact scope (no search-and-reprompt,
    # SAM2 propagation only) rather than refactoring the shared, GATE-2-
    # tested obj_id=2 path -- smaller diff, zero behavior change for every
    # existing single-secondary caller (GATE 2's own occluder_geometry
    # use, R2's dual-object real-model reconstruction). A genuine N>3
    # object scenario (a full billiards rack) would need a real
    # generalization of this pattern, not attempted here -- 2-3 balls
    # only, matching what was actually asked for.
    known_occluded_frames: dict = field(default_factory=dict)
    # {frame_idx: bool} -- for the PRIMARY object, whether physics's own
    # occluder_bounds check found this specific search attempt's predicted
    # position inside a KNOWN occluder (AGENT.md M5, existence-statistic
    # hardening, 2026-08). Recorded for every attempted frame regardless of
    # outcome -- unlike reid_events, which only logs frames where a tier
    # actually produced a candidate/score. This is what lets
    # reconstruct.py's _existence_mask distinguish a gap that's ENTIRELY
    # explained by a known occluder throughout (plausibly still there,
    # should be treated like ordinary invisible-in-state-space occlusion,
    # not penalized) from a gap that opens or persists somewhere with no
    # occluder explanation at all (genuinely anomalous -- see AGENT.md's
    # own writeup on why a pure duration/survival statistic turned out to
    # be insufficient on its own: a `vanish` mutant's t_star is triggered
    # independent of the object's position, so most instances are
    # spatially unexplained, but some coincidentally occur near a real
    # occluder and are philosophically, not just practically, ambiguous
    # from pixels alone -- this fix targets the former, common case, not
    # a claim of resolving the latter).


def _recent_visible_run(masks, last_visible_idx, max_frames=15):
    """Consecutive non-empty-mask frame indices walking backward from (and
    including) last_visible_idx, oldest first -- the window motion_prior's
    fit_pixel_track expects. Stops at the first empty/missing mask (a real
    occlusion gap earlier in the SAME track shouldn't be fit through)."""
    out = []
    i = last_visible_idx
    while i >= 0 and len(out) < max_frames:
        m = masks.get(i)
        if m is None or m.sum() == 0:
            break
        out.append(i)
        i -= 1
    return list(reversed(out))


def _maybe_recache_embedding(frame_idx, masks, frame_paths, dino_model, device):
    """Called the instant SAM2's mask goes empty (empty_streak==0): re-embeds
    the LAST frame the object was actually visible on (frame_idx - 1),
    retroactively, for the DINO fallback tier -- not on every visible frame
    (interleaving a DINO forward pass every single frame was the actual
    cause of an in-loop CUDA fragmentation OOM, see track_with_
    reidentification's own inline comment at its call site). Returns a new
    embedding to cache, or None if there's nothing to re-embed (no valid
    prior frame, or that frame's own mask was itself empty).

    Bounds-checked against frame_paths on BOTH ends (2026-08, found
    directly: a real `IndexError: list index out of range` crash at this
    exact line -- `frame_paths[frame_idx - 1]` with no upper-bound guard,
    unlike the `next_idx`/search-side access a few lines down which
    already checks `frame_idx + 1 < T` -- reproduced deterministically on
    a jitter-mutant/ramp_descent_high_friction episode). `frame_idx - 1`
    can go negative (the very first frame already came back empty) or, if
    `frame_paths` and whatever SAM2 actually propagated over ever
    disagree on length, land outside `frame_paths`' own range -- the exact
    live-SAM2 trigger was never pinned down (needs a real GPU repro this
    dev machine can't run), but EITHER way, an out-of-range prev_idx is a
    'nothing to re-embed yet' case, not a crash, regardless of which
    mechanism produces it. Extracted as its own function specifically so
    this bounds-checking is unit-testable without mocking SAM2's entire
    propagation protocol."""
    prev_idx = frame_idx - 1
    if not (0 <= prev_idx < len(frame_paths)):
        return None
    prev_mask = masks.get(prev_idx)
    if prev_mask is None or prev_mask.sum() == 0:
        return None
    from PIL import Image
    prev_rgb = np.array(Image.open(frame_paths[prev_idx]).convert("RGB"))
    return embed_region(dino_model, prev_rgb, prev_mask, device)


def track_with_reidentification(sam2_predictor, dino_model, mask_gen_factory, frame_paths, frame0_mask,
                                 similarity_threshold, forgiveness_frames=1, device="cuda", dt=1.0 / 30,
                                 camera=None, deceleration=None, occluder_bounds=None,
                                 frame0_mask_secondary=None, occluder_geometry=None,
                                 frame0_mask_tertiary=None):
    """frame_paths: ordered list of frame file paths, the same frames
    sam2_predictor was pointed at via frame_dir (must be a directory of
    NNNNN.jpg files -- SAM2's expected layout).

    mask_gen_factory: zero-arg callable that builds and returns a
    SAM2AutomaticMaskGenerator (e.g. functools.partial(load_mask_generator,
    device=device)), NOT an already-built instance. Loaded lazily on the
    first actual search -- most frames/trajectories never need it (SAM2's
    own tracking already handles short gaps fine), and this repo's own 8GB
    GPU can't comfortably hold the video predictor's growing memory bank
    plus DINO plus an idle mask generator for a 150-frame video at once;
    measured directly, dropping the always-loaded mask generator was what
    fixed an OOM that occurred purely inside normal SAM2 propagation, with
    no search ever triggered.

    dt: seconds per frame (default matches this repo's fps=30 convention
    used throughout scenes/manifests) -- only used to fit motion_prior's
    kinematic extrapolation; get this right for the actual video or the
    physics-prior fit's implied velocity/deceleration will be wrong even
    though its R^2 still looks fine (R^2 is scale-invariant, this bug
    wouldn't self-reveal).

    camera, deceleration: OPTIONAL. When BOTH are given, tier 1 uses
    `motion_prior.fit_metric_track`/`MetricTrackFit` (AGENT.md defect #16's
    fix) instead of the older `fit_pixel_track`/`PixelTrackFit` -- fits
    only (position, velocity) per-episode in METRIC 3D space (well-
    conditioned over a short window) and borrows `deceleration` from a
    pre-calibrated, reference-ensemble constant (e.g.
    `detect/physical_constants.py`'s corridor fit, CV=0.12%) rather than
    re-deriving it per-episode from noisy tracked pixels, which measured
    directly to be unreliable when extrapolated far (predicted a stop tens
    of pixels from where the object actually was, still moving). Falls
    back to the pixel-only path when either is omitted -- that path still
    exists deliberately, for scenarios without camera calibration or a
    pre-calibrated deceleration constant available; see motion_prior.py's
    own module docstring for its documented, narrower accuracy guarantees.
    `camera` is a dict with cam_pos, cam_mat, fovy_deg, width, height,
    plane_z -- the same fields `reconstruct.py` already needs, since this
    is exactly its own ray/plane method, reused, not a new dependency
    invented here.

    occluder_bounds: OPTIONAL, only meaningful when the metric path above
    is active. AGENT.md defect #17's fix: an accurate metric prediction
    that is still genuinely behind a KNOWN occluder (e.g. occlusion_
    corridor's wall) should refuse to offer a search target at all, not
    merely trust the distance gate downstream -- otherwise a static visual
    feature ON the occluder itself, sitting directly on the object's own
    path (since the occluder was placed there on purpose), can get
    accepted before the real object actually clears it. Passed straight
    through to `fit_metric_track`; see its docstring for the exact
    (lo, hi) convention. Known scene geometry, not privileged object
    state -- omit for scenarios without a relevant known occluder.

    frame0_mask_secondary: OPTIONAL (AGENT.md M5.6, scoped phase-2 multi-
    object support). When given, a SECOND object (SAM2 obj_id=2) is
    prompted at frame 0 and tracked inside the SAME shared propagation
    loop -- but gets NONE of the search-and-reprompt machinery above; its
    masks are recorded into `result.masks_secondary` exactly as SAM2's own
    propagation produces them, self-recovery included, reidentification
    never involved. This is a deliberate scope limit, not an oversight --
    see M5.6's own "scoping decision" note for why fully general
    simultaneous multi-object tracking (both objects independently
    mid-search at once) was not attempted. Omit for the K=1 case; behavior
    is then identical to before this parameter existed.

    occluder_geometry: OPTIONAL (occluder_plane_z, x_half_width,
    y_half_width) -- AGENT.md M5.7's dynamic, tracked-occluder case (e.g.
    occlusion_corridor_moving.xml's sliding occluder). Requires
    frame0_mask_secondary to also be given (the secondary object IS the
    occluder here). When both are present, `motion_prior.
    occluder_bounds_from_track` builds a live lookup over
    `result.masks_secondary` (the SAME dict the propagation loop is still
    filling in as it runs -- a plain dict is mutable, so the lookup always
    sees whatever's been tracked so far, no separate wiring needed) and
    that REPLACES `occluder_bounds` for every `fit_metric_track` call in
    this run, rather than being merged with it -- a scene has one relevant
    occluder or the other, never both at once, and the dynamic case is
    strictly more accurate when available. Omit for a static-occluder or
    no-occluder scenario; `occluder_bounds` (the static tuple) keeps
    working exactly as before when this is omitted.

    frame0_mask_tertiary: OPTIONAL (2026-09, billiards/multi-collision
    scoping). A THIRD object (SAM2 obj_id=3), tracked in the SAME shared
    propagation loop with the exact same no-search-and-reprompt scope as
    `frame0_mask_secondary` above -- additive, not a generalization of
    that parameter to a list, to keep this change small and leave every
    existing single-secondary caller untouched. Omit for the K<=2 case;
    behavior is then identical to before this parameter existed.

    Runs SAM2 tracking; whenever the mask has been empty for more than
    `forgiveness_frames` consecutive frames, switches to search mode:
    generate candidate object proposals on the next frame, then identify
    the match in two tiers -- (1) PRIMARY: fit motion_prior's kinematic
    extrapolation from this track's own recent pre-occlusion centroids and
    take the candidate nearest the predicted position, if enough
    pre-occlusion history exists to fit one (motion_prior.MIN_FIT_FRAMES);
    (2) FALLBACK: DINO appearance similarity against the cached pre-loss
    embedding, when there's too little history for a physics fit. See
    reidentify.py's module docstring and motion_prior.py for why physics
    is primary now, not DINO, for today's K=1 scenarios. Resumes normal
    propagation either way.
    """
    import torch
    from PIL import Image
    from . import segmentation as seg
    from . import motion_prior as mp

    # Pinned before any GPU op runs (AGENT.md defect #19): three back-to-back
    # calls of this exact function, on the exact same input frames, produced
    # two DIFFERENT reid traces -- one run's DINO similarity for a static wall
    # feature came back 0.4048 (over the 0.4 threshold, false match) where two
    # other runs of the identical code gave 0.3998 (safely under) for what
    # should be the identical comparison. cudnn's default (non-"deterministic")
    # algorithm selection can return numerically different results run-to-run
    # even at a fixed algorithm choice, and TF32's reduced mantissa widens
    # that gap further -- exactly the kind of jitter that flips a borderline
    # similarity score across a fixed threshold. This is the difference
    # between a real defect and its own instrument being unreliable, which
    # violates 3.1 (measurement symmetry) if left unpinned: the same episode
    # must not get a different verdict on different days.
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    frame_dir = str(pathlib.Path(frame_paths[0]).parent)
    T = len(frame_paths)

    # offload_video_to_cpu/offload_state_to_cpu=True confirmed necessary by a
    # clean A/B test: plain segmentation.track_video() (identical settings,
    # zero DINO/reid involvement) succeeds on this exact trajectory WITH
    # offloading. Everything SAM2-related -- init_state, add_new_mask,
    # propagate_in_video -- stays inside ONE inference_mode/autocast context,
    # matching track_video()'s structure exactly. A periodic
    # torch.cuda.empty_cache() call INSIDE that context, tried earlier, made
    # things worse, not better: it fights the caching allocator's own reuse
    # pool mid-generator and was the actual source of a fragmentation OOM
    # that plain track_video() never hit. Don't call it in this loop.
    frame0_rgb = np.array(Image.open(frame_paths[0]).convert("RGB"))
    cached_embedding = embed_region(dino_model, frame0_rgb, frame0_mask, device)

    result = ReidentifyResult()
    empty_streak = 0
    start_frame = 0
    mask_gen = None   # lazy -- built on first actual search, see docstring

    # AGENT.md M5.7: built ONCE, over the live `result.masks_secondary`
    # dict -- see occluder_geometry's own docstring above for why a plain
    # mutable dict is enough (no snapshot/refresh logic needed) and why this
    # REPLACES occluder_bounds rather than merging with it.
    if occluder_geometry is not None:
        if frame0_mask_secondary is None:
            raise ValueError("occluder_geometry requires frame0_mask_secondary -- the dynamic "
                              "occluder case needs a tracked secondary object to read positions from.")
        if camera is None:
            raise ValueError("occluder_geometry requires camera -- unprojecting the tracked "
                              "occluder's own centroid needs the same calibration fit_metric_track does.")
        occ_plane_z, occ_x_half, occ_y_half = occluder_geometry
        occluder_bounds = mp.occluder_bounds_from_track(
            result.masks_secondary, dt, cam_pos=camera["cam_pos"], cam_mat=camera["cam_mat"],
            fovy_deg=camera["fovy_deg"], width=camera["width"], height=camera["height"],
            occluder_plane_z=occ_plane_z, x_half_width=occ_x_half, y_half_width=occ_y_half)
    # Positions of physics-tier candidates rejected during the CURRENT gap.
    # A genuine reappearance is a new visual event; a candidate that keeps
    # showing up at roughly the same screen location across several rejected
    # search attempts is diagnostic of a persistent static scene feature
    # (shadow, floor/wall marking), not the object -- found the hard way
    # (AGENT.md build-sequence step 27): a size-based distance gate alone
    # still eventually accepted a static distractor once the physics
    # prediction (converging toward the object's estimated resting position)
    # drifted close enough to it, even though the "candidate" itself never
    # moved across 10 consecutive search attempts. Cleared on every
    # successful reacquisition (a new gap starts fresh).
    rejected_positions = []

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = sam2_predictor.init_state(video_path=frame_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
        sam2_predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=frame0_mask)
        track_secondary = frame0_mask_secondary is not None
        if track_secondary:
            sam2_predictor.add_new_mask(state, frame_idx=0, obj_id=2, mask=frame0_mask_secondary)
        track_tertiary = frame0_mask_tertiary is not None
        if track_tertiary:
            sam2_predictor.add_new_mask(state, frame_idx=0, obj_id=3, mask=frame0_mask_tertiary)

        while start_frame < T:
            reprompted_at = None
            # start_frame_idx=None (SAM2's own default / internal fast path) for
            # the first pass, matching the plain propagate_in_video(state) call
            # the original raw-SAM2 sweep used successfully at this exact frame
            # count; only pass an explicit index on a genuine resume-after-reprompt.
            propagate_kwargs = {} if start_frame == 0 else {"start_frame_idx": start_frame}
            for frame_idx, obj_ids, mask_logits in sam2_predictor.propagate_in_video(state, **propagate_kwargs):
                # Indexed via obj_ids.index(...) rather than the old hardcoded
                # [0, 0] -- harmless no-op for the K=1 case (obj_ids is always
                # [1] there, index 0), but required once a second object
                # (obj_id=2) can also be present in the same propagation call
                # (AGENT.md M5.6): SAM2 doesn't guarantee slot order matches
                # obj_id order.
                pred = (mask_logits[obj_ids.index(1), 0] > 0).cpu().numpy()
                result.masks[frame_idx] = pred
                if track_secondary and 2 in obj_ids:
                    result.masks_secondary[frame_idx] = (mask_logits[obj_ids.index(2), 0] > 0).cpu().numpy()
                if track_tertiary and 3 in obj_ids:
                    result.masks_tertiary[frame_idx] = (mask_logits[obj_ids.index(3), 0] > 0).cpu().numpy()

                if pred.sum() > 0:
                    empty_streak = 0
                else:
                    if empty_streak == 0:   # THIS frame is the first empty one -- (frame_idx-1)
                                             # was the last visible frame; re-embed it now,
                                             # retroactively, instead of every visible frame.
                                             # DINO's forward pass interleaved with SAM2's own
                                             # every single frame was the actual cause of an
                                             # in-loop CUDA fragmentation OOM (measured: it
                                             # disappeared once DINO calls dropped from
                                             # ~1/frame to ~1/loss-event); it's also more
                                             # principled -- we want the appearance right
                                             # before occlusion, not an arbitrary mid-video one.
                                             # See _maybe_recache_embedding's own docstring for
                                             # why this is a bounds-checked helper call, not
                                             # inline indexing (a real IndexError crash here,
                                             # fixed 2026-08).
                        emb = _maybe_recache_embedding(frame_idx, result.masks, frame_paths, dino_model, device)
                        if emb is not None:
                            cached_embedding = emb
                    empty_streak += 1

                if (empty_streak > forgiveness_frames and frame_idx + 1 < T):
                    next_idx = frame_idx + 1
                    if mask_gen is None:
                        mask_gen = mask_gen_factory()
                    frame_rgb = np.array(Image.open(frame_paths[next_idx]).convert("RGB"))
                    candidates = mask_gen.generate(frame_rgb)
                    # no torch.cuda.empty_cache() here -- calling it mid-loop, inside
                    # this inference_mode/autocast context, fights the caching
                    # allocator's own reuse pool and was the actual source of a
                    # fragmentation OOM elsewhere in this function; removed.

                    # TIER 1 (primary): physics-prior gating, see module docstring.
                    # Metric-anchored fit (defect #16's fix) when camera + a
                    # pre-calibrated deceleration are both available; otherwise
                    # the older pixel-only fit -- see track_with_reidentification's
                    # own docstring for why both paths still exist.
                    drop_frame = frame_idx - empty_streak      # last visible frame
                    # Size gate first (REID_SIZE_RATIO_MAX): neither tier ever
                    # sees a proposal that cannot be the same object.
                    candidates = _size_consistent(candidates, result.masks.get(drop_frame))
                    recent = _recent_visible_run(result.masks, drop_frame)
                    use_metric = camera is not None and deceleration is not None
                    if len(recent) >= mp.MIN_FIT_FRAMES:
                        if use_metric:
                            fit = mp.fit_metric_track(result.masks, recent, dt, deceleration=deceleration,
                                                       occluder_bounds=occluder_bounds, **camera)
                        else:
                            fit = mp.fit_pixel_track(result.masks, recent, dt)
                    else:
                        fit = None

                    matched, best_mask, best_sim, method = False, None, None, None
                    predicted_pos = None
                    gate_px = None
                    known_occluded = False
                    if fit is not None:
                        predict_t = (next_idx - drop_frame) * dt
                        if use_metric:
                            known_occluded = fit.occluded_at(predict_t)
                        predicted_pos = fit.predict_pixel(predict_t) if use_metric else fit.predict(predict_t)
                        # predicted_pos is None if EITHER the metric point projected behind
                        # the camera (degenerate) OR it's still within a known occluder's
                        # bounds (defect #17 -- genuinely not visible yet, don't offer a
                        # search target at all). Either way: skip tier 1 for this attempt.
                        # known_occluded distinguishes the second case specifically --
                        # positive evidence the object can't be visible yet, as opposed to
                        # simply "no opinion" -- used below to suppress tier 2 as well
                        # (AGENT.md defect #19).
                    result.known_occluded_frames[next_idx] = known_occluded
                    if predicted_pos is not None:
                        scored = mp.score_candidates_by_position(candidates, predicted_pos)
                        # Gate scaled to the object's own last-known size
                        # (mask_diagonal_px), not an arbitrary pixel constant --
                        # a physically-grounded, if still first-pass, choice; not
                        # yet independently calibrated via a reference-ensemble
                        # sweep the way theta_k thresholds are (3.3). Revisit if
                        # false accepts/rejects turn up in further validation.
                        # Computed here (not just inside the `if scored:` below)
                        # because the DINO tier reuses it too, see its own comment.
                        last_mask = result.masks.get(drop_frame)
                        diag = mp.mask_diagonal_px(last_mask) if last_mask is not None else None
                        gate_px = max(5.0, 1.0 * diag) if diag else 15.0
                        if os.environ.get("VITALS_REID_DEBUG"):
                            top3 = [(round(d, 1), mp.centroid(c["segmentation"])) for d, c in scored[:3]]
                            print(f"DEBUG frame={next_idx} predicted_pos={tuple(round(x,1) for x in predicted_pos)} "
                                  f"n_candidates={len(scored)} top3={top3} rejected_positions={rejected_positions}")
                        if scored:
                            dist, nearest = scored[0]
                            # Gate on distance -- do NOT accept the nearest candidate
                            # unconditionally. Found the hard way (AGENT.md build-
                            # sequence step 27): search fires as soon as empty_streak
                            # > forgiveness_frames, almost immediately after a loss and
                            # usually long before a real occlusion actually resolves,
                            # so there is often no true candidate near the prediction
                            # AT ALL -- an ungated "nearest" still returns something,
                            # and gets wrongly accepted (measured: locked onto a false
                            # match at 23px, tracked it for the rest of a 150-frame
                            # clip).
                            cand_cen = mp.centroid(nearest["segmentation"])
                            # a candidate near a position rejected earlier in THIS gap
                            # is suspected to be a persistent static feature, not a
                            # reappearance -- require it to be genuinely close to the
                            # CURRENT prediction (tight absolute bound), not just under
                            # the size-based gate, which a static object will eventually
                            # satisfy purely because the prediction drifts toward it.
                            is_suspect = cand_cen is not None and any(
                                np.hypot(cand_cen[0] - rp[0], cand_cen[1] - rp[1]) <= STATIC_SUSPECT_RADIUS_PX
                                for rp in rejected_positions)
                            effective_gate = TIGHT_GATE_PX if is_suspect else gate_px

                            if dist <= effective_gate:
                                matched, best_mask, method = True, nearest["segmentation"], "physics"
                                result.reid_events.append((next_idx, dist, matched, method))
                            else:
                                result.reid_events.append((next_idx, dist, False,
                                                            "physics_rejected_suspect" if is_suspect else "physics_rejected"))
                                if cand_cen is not None:
                                    rejected_positions.append(cand_cen)

                    # TIER 2 (fallback): DINO appearance, whenever physics didn't
                    # match -- either because it couldn't fit at all (too little
                    # pre-occlusion history) OR because it actively rejected the
                    # nearest candidate as implausible (AGENT.md defect #18). The
                    # second case matters: physics rejecting a candidate is real,
                    # informative evidence ("this location doesn't look right"),
                    # not merely "no opinion" -- and DINO used to run with zero
                    # awareness of it, meaning it could confidently accept a
                    # candidate physics would have refused on spatial grounds
                    # alone. Originally tried gating DINO on proximity to
                    # rejected_positions (the exact points physics rejected) --
                    # too narrow: SAM2 splits one static distractor (a
                    # wall/doorframe feature) into several sub-masks at
                    # different pixel offsets, so DINO's highest-similarity
                    # candidate can legitimately be a different sub-mask of the
                    # SAME static thing, 10-15px from any single rejected point,
                    # evading a small-radius check entirely (measured directly:
                    # physics rejected (95.2,118.6) four times in a gap; DINO's
                    # best match landed at (94.9,104.5), 14px away, and was
                    # accepted). Fixed properly by reusing physics's own
                    # spatial gate (gate_px, size-scaled to the object's last
                    # known extent) against DINO's candidate directly, whenever
                    # physics had a valid prediction this frame -- not just
                    # remembering exact past-rejected points. A DINO candidate
                    # far outside where the object could plausibly be is
                    # implausible regardless of appearance similarity; DINO
                    # only overrides physics within physics's own notion of
                    # "plausibly still the object." Separately: if physics
                    # affirmatively knows the object is still behind a known
                    # occluder (known_occluded), don't even let DINO try --
                    # measured directly on this same clip: with the spatial
                    # gate above in place but this check absent, DINO still
                    # false-matched a static feature at frame 40 (sim 0.4048,
                    # barely over threshold), 24 frames before the object
                    # could possibly be visible (rise=64) -- precisely because
                    # physics has no predicted_pos to gate against yet in this
                    # window (that's what makes it a *known*-occluded case,
                    # not just an unfit one), so the appearance-only fallback
                    # ran with zero constraint. known_occluded is a strictly
                    # narrower condition than "fit failed" -- it only fires
                    # when physics can extrapolate a definite position AND
                    # that position is provably inside occluder_bounds, not
                    # merely whenever there's too little history to fit at
                    # all (that case still falls through to DINO unguarded,
                    # same as before, since there's no positive evidence
                    # either way).
                    if not matched and not known_occluded and cached_embedding is not None:
                        best_sim, best_mask = -1.0, None
                        for c in candidates:
                            cand_emb = embed_region(dino_model, frame_rgb, c["segmentation"], device)
                            if cand_emb is None:
                                continue
                            sim = cosine_similarity(cached_embedding, cand_emb)
                            if sim > best_sim:
                                best_sim, best_mask = sim, c["segmentation"]
                        matched = best_sim >= similarity_threshold
                        method = "dino"
                        best_cen = mp.centroid(best_mask) if best_mask is not None else None
                        if matched and predicted_pos is not None and gate_px is not None:
                            dino_dist = np.hypot(best_cen[0] - predicted_pos[0], best_cen[1] - predicted_pos[1]) \
                                if best_cen is not None else None
                            if dino_dist is None or dino_dist > gate_px:
                                matched = False
                                method = "dino_rejected_implausible"
                        if os.environ.get("VITALS_REID_DEBUG"):
                            print(f"DEBUG frame={next_idx} dino_best_sim={round(best_sim,4)} "
                                  f"dino_best_cen={tuple(round(x,1) for x in best_cen) if best_cen else None} "
                                  f"predicted_pos={tuple(round(x,1) for x in predicted_pos) if predicted_pos else None} "
                                  f"gate_px={round(gate_px,1) if gate_px else None} matched={matched} method={method}")
                        result.reid_events.append((next_idx, best_sim, matched, method))

                    if matched:
                        sam2_predictor.add_new_mask(state, frame_idx=next_idx, obj_id=1, mask=best_mask)
                        empty_streak = 0
                        reprompted_at = next_idx
                        rejected_positions.clear()   # this gap is closed -- a future gap starts fresh
                        del candidates
                        # Free the mask generator once a search succeeds -- it isn't needed
                        # again unless another loss occurs, and holding THREE model instances
                        # resident (video predictor + DINO + mask generator) for the rest of a
                        # long video measurably exhausts VRAM on a modest (7.77GB) card: hit
                        # directly on a 150-frame occlusion_corridor clip, OOM well into the
                        # resumed propagation, not during the search itself. mask_gen_factory()
                        # reloads it lazily if a later loss needs it again -- a small latency
                        # cost, not a correctness one. Plain `del` + None (no empty_cache here)
                        # -- a single call at a rare event like this is a different usage
                        # pattern from the earlier documented mid-loop-empty_cache fragmentation
                        # issue, but staying conservative: dropping the reference alone already
                        # lets the caching allocator reuse this memory for later allocations.
                        del mask_gen
                        mask_gen = None
                        break   # exit this propagate_in_video generator; restart from next_idx below
                    del candidates

            start_frame = reprompted_at if reprompted_at is not None else T

        sam2_predictor.reset_state(state)
        del state

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result
