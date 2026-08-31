"""phi/scene_geometry.py -- the canonical per-scenario reconstruction config
(AGENT.md M6, 2026-08 consolidation). Pure data/numpy, no MuJoCo or GPU
needed, so this runs unconditionally like test_motion_prior.py.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from vitals.phi import scene_geometry as sg


def test_every_scene_has_exactly_one_reconstruction_mode():
    """Each entry's reconstruct_kwargs must satisfy reconstruct_trajectory's
    own XOR contract (exactly one of plane_z / plane_pieces / ballistic) --
    a regression guard so a bad config entry fails HERE, at config-authoring
    time, rather than deep inside a real adapter's first real call."""
    for name, cfg in sg.SCENES.items():
        kw = cfg["reconstruct_kwargs"]
        n_modes = sum(k in kw for k in ("plane_z", "plane_pieces")) + bool(kw.get("ballistic"))
        assert n_modes == 1, f"{name}: reconstruct_kwargs must select exactly one mode, got {kw}"
        assert cfg["mode"] in ("plane_z", "plane_pieces", "ballistic"), f"{name}: unknown mode {cfg['mode']!r}"


def test_get_raises_for_unknown_scenario():
    try:
        sg.get("not_a_real_scenario")
        assert False, "should raise KeyError for a scenario with no registered geometry"
    except KeyError:
        pass


def test_get_returns_the_same_object_every_scene_uses():
    """get() isn't a parallel lookup path that could drift from SCENES
    itself -- it must return the literal same dict."""
    for name in sg.SCENES:
        assert sg.get(name) is sg.SCENES[name]


def test_ramp_plane_pieces_toe_boundary_matches_ramp_toe_x():
    """The ramp piece's own valid_fn boundary (RAMP_TOE_X + a small margin)
    must stay consistent with the standalone RAMP_TOE_X constant other
    callers (e.g. tests/test_reconstruct.py's own cross-scene-coverage
    assertions) import separately -- a regression guard against the two
    silently diverging if one is ever edited without the other."""
    ramp_piece, floor_piece = sg.RAMP_PLANE_PIECES
    _, _, ramp_valid_fn = ramp_piece
    _, _, floor_valid_fn = floor_piece
    just_inside = np.array([sg.RAMP_TOE_X, 0.0, 0.0])
    just_outside = np.array([sg.RAMP_TOE_X + 1.0, 0.0, 0.0])
    assert bool(ramp_valid_fn(just_inside))
    assert not bool(ramp_valid_fn(just_outside))
    assert bool(floor_valid_fn(just_outside))   # floor piece is the always-valid fallback


def test_occlusion_corridor_wall_bounds_is_not_the_naive_box_extent():
    """Regression guard for the exact drift this module exists to prevent
    (AGENT.md M6's own consolidation writeup): the wall's naive geometric
    box extent (pos=5, half-size 0.75 -> (4.25, 5.75)) undersells the true
    visual occlusion span by ~0.12-0.15m on the far edge -- if someone ever
    "simplifies" OCCLUSION_CORRIDOR_WALL_BOUNDS back to the naive value,
    this should fail loudly, not silently reintroduce the GATE 2 null
    false-positive mechanism that value caused."""
    lo, hi = sg.OCCLUSION_CORRIDOR_WALL_BOUNDS
    naive_lo, naive_hi = 4.25, 5.75
    assert (lo, hi) != (naive_lo, naive_hi)
    assert hi > naive_hi, "the empirically-measured far edge must extend past the naive box boundary"


if __name__ == "__main__":
    test_every_scene_has_exactly_one_reconstruction_mode()
    print("PASS  test_every_scene_has_exactly_one_reconstruction_mode")
    test_get_raises_for_unknown_scenario()
    print("PASS  test_get_raises_for_unknown_scenario")
    test_get_returns_the_same_object_every_scene_uses()
    print("PASS  test_get_returns_the_same_object_every_scene_uses")
    test_ramp_plane_pieces_toe_boundary_matches_ramp_toe_x()
    print("PASS  test_ramp_plane_pieces_toe_boundary_matches_ramp_toe_x")
    test_occlusion_corridor_wall_bounds_is_not_the_naive_box_extent()
    print("PASS  test_occlusion_corridor_wall_bounds_is_not_the_naive_box_extent")
