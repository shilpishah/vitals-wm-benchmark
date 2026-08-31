"""vitals/render/annotate.py -- pure numpy/PIL drawing helpers, no GPU/
MuJoCo needed. Extracted 2026-08 from render_annotated_comparison_video.py
when run_model_population.py's grid-video feature needed the same
border/marker drawing a second place."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from vitals.render.annotate import draw_border, draw_marker, tile_grid_video


def test_draw_border_paints_only_the_edge():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    out = draw_border(frame, (255, 0, 0), thickness=3)
    assert (out[0, :, :] == [255, 0, 0]).all()
    assert (out[-1, :, :] == [255, 0, 0]).all()
    assert (out[:, 0, :] == [255, 0, 0]).all()
    assert (out[:, -1, :] == [255, 0, 0]).all()
    assert (out[10, 15, :] == [0, 0, 0]).all(), "interior must be untouched"
    assert frame.sum() == 0, "must not mutate the input frame"


def test_draw_marker_skips_none_and_nan_without_raising():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    out1 = draw_marker(frame, None, None, (0, 255, 0))
    assert (out1 == frame).all()
    out2 = draw_marker(frame, float("nan"), 5.0, (0, 255, 0))
    assert (out2 == frame).all()


def test_tile_grid_video_square_grid_places_every_tile():
    episodes = [np.full((4, 10, 12, 3), fill_value=i, dtype=np.uint8) for i in range(9)]
    grid = tile_grid_video(episodes, sep=2, sep_color=(1, 1, 1))
    assert grid.shape == (4, 3 * 10 + 2 * 2, 3 * 12 + 2 * 2, 3)
    # tile 0 (row 0, col 0) top-left corner
    assert (grid[0, 0, 0] == 0).all()
    # tile 4 (row 1, col 1, the center of a 3x3) -- offset by (H+sep, W+sep)
    assert (grid[0, 10 + 2, 12 + 2] == 4).all()
    # tile 8 (row 2, col 2, bottom-right)
    assert (grid[0, 2 * (10 + 2), 2 * (12 + 2)] == 8).all()


def test_tile_grid_video_partial_grid_fills_leftover_with_sep_color():
    episodes = [np.full((2, 5, 5, 3), fill_value=i, dtype=np.uint8) for i in range(5)]
    grid = tile_grid_video(episodes, sep=1, sep_color=(9, 9, 9))
    # floor(sqrt(5))=2 rows, ceil(5/2)=3 cols -> 6 tiles, 1 leftover
    assert grid.shape == (2, 2 * 5 + 1, 3 * 5 + 2, 3)
    last_tile_y0, last_tile_x0 = 1 * (5 + 1), 2 * (5 + 1)
    assert (grid[0, last_tile_y0, last_tile_x0] == 9).all(), "leftover tile must be filled with sep_color"


def test_tile_grid_video_rejects_mismatched_shapes():
    episodes = [np.zeros((4, 10, 12, 3), dtype=np.uint8), np.zeros((4, 8, 12, 3), dtype=np.uint8)]
    try:
        tile_grid_video(episodes)
        assert False, "must reject episodes with different (T,H,W)"
    except ValueError:
        pass


if __name__ == "__main__":
    test_draw_border_paints_only_the_edge()
    print("PASS  test_draw_border_paints_only_the_edge")
    test_draw_marker_skips_none_and_nan_without_raising()
    print("PASS  test_draw_marker_skips_none_and_nan_without_raising")
    test_tile_grid_video_square_grid_places_every_tile()
    print("PASS  test_tile_grid_video_square_grid_places_every_tile")
    test_tile_grid_video_partial_grid_fills_leftover_with_sep_color()
    print("PASS  test_tile_grid_video_partial_grid_fills_leftover_with_sep_color")
    test_tile_grid_video_rejects_mismatched_shapes()
    print("PASS  test_tile_grid_video_rejects_mismatched_shapes")
