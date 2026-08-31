"""Promptable video segmentation via SAM2 -- prompted ONLY with the
ground-truth frame-0 mask (AGENT.md 3.2: "prompt segmentation with the
ground-truth frame-0 mask from the manifest"). Every frame after that is
the model's own propagation, never oracle data.

Requires torch + the official SAM2 package (facebookresearch/sam2) and a
CUDA device -- imported lazily inside functions, not at module load, so
this module can still be imported (just not called) on a machine without
those installed, matching physics/runner.py's pattern for its mujoco
dependency.

Measured behavior (scripts/validate_reidentification.py, ramp_descent /
occlusion_corridor): tracking IoU ~0.75-0.80 while the object is visible;
occlusion recovery has a hard cliff at ~22-27 frames (~1s) -- below that it
always re-acquires within 1-2 frames, above that it never recovers on its
own. That cliff is what reidentify.py exists to fix.
"""
from __future__ import annotations
import os
import numpy as np


def load_predictor(checkpoint="facebook/sam2.1-hiera-tiny", device="cuda"):
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    return SAM2VideoPredictor.from_pretrained(checkpoint, device=device)


def track_video(predictor, frame_dir, frame0_mask, offload_to_cpu=True):
    """frame_dir: directory of frames named NNNNN.jpg (SAM2's expected
    format). frame0_mask: (H, W) bool, the GT prompt for frame 0 ONLY.

    Returns {frame_idx: (H, W) bool mask}. A frame absent from the dict, or
    present with an all-False mask, means "object not visible" -- SAM2 does
    have a genuine absence signal (verified empirically: 100% correct empty
    predictions during real occlusion in our tests), it just can't recover
    from it on its own past the memory-window cliff above.
    """
    import torch

    n_frames = len([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frame_dir,
                                      offload_video_to_cpu=offload_to_cpu,
                                      offload_state_to_cpu=offload_to_cpu)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=frame0_mask)

        masks = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()

        predictor.reset_state(state)

    del state
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return masks


def reprompt_at_frame(predictor, state, frame_idx, mask):
    """Give the tracker a fresh mask at a specific frame (used by
    reidentify.py after a successful re-identification match) and resume
    propagation from there. Separate from track_video's one-shot loop
    because re-identification needs to interleave "propagate a few frames,
    check if still tracking, maybe re-prompt" rather than running the whole
    video in one pass."""
    import torch
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=1, mask=mask)
