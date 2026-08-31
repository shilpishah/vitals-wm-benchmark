"""Phi: video -> Trajectory (AGENT.md M4).

Built incrementally, one measured component at a time -- not as a single
monolithic pipeline written speculatively. Each piece here has a stated,
measured error floor (AGENT.md 8.1) before the next one is built on top of
it, the same discipline as everything else in this repo.

Currently implemented:
  segmentation.py   promptable video segmentation (SAM2), prompted with the
                     GT frame-0 mask only -- never oracle masks past frame 0
                     (AGENT.md 3.1/3.2)
  reidentify.py      DINO-based appearance re-identification, wrapping
                     segmentation.py, for recovering object identity after
                     occlusions too long for SAM2's own memory (empirically
                     ~22-27 frames on occlusion_corridor -- see
                     scripts/validate_reidentification.py)
  thresholds.py      leave-one-out threshold estimation for the re-ID
                     similarity cutoff, same philosophy as
                     detect/thresholds.py -- estimated, never chosen (3.3)

Everything else in AGENT.md's M4 table (point tracks, pose, 3D position,
static keypoints/camera, contact graph) is still missing.
"""
