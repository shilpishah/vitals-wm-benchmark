"""occlusion_reemergence + block_stack (2026-09-11, scenario scaling) and
the ball-only perturbation mode they needed. These are the properties the
scene tuning established by simulation, pinned so a later edit to a scene
file, manifest, or the runner cannot silently change what the reference
ensemble means.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from vitals.types import EpisodeSpec
from vitals.physics.runner import rollout
from vitals.phi import scene_geometry as sg
from vitals.adapters.scenario_prompts import SCENARIO_PROMPTS

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _spec(name, **over):
    m = yaml.safe_load((ROOT / f"configs/manifests/{name}.yaml").read_text())
    kw = dict(name=m["name"], scene=str(ROOT / m["scene"]), target_property=m["target_property"], band=m["band"],
              lam=m["lam"], n_reference=m["n_reference"], horizon_s=m["horizon_s"], fps=m["fps"],
              seed=m.get("seed", 0), perturb_mode=m.get("perturb_mode", "full"))
    kw.update(over)
    return EpisodeSpec(**kw)


def _tilt_deg(quat_wxyz):
    return np.degrees(2 * np.arccos(np.clip(np.abs(quat_wxyz[..., 0]), 0, 1)))


def test_render_domain_variants_share_physics_and_differ_only_visually():
    """M10 (2026-09-16): occlusion_reemergence_domA/domB must be the base
    scene's physics exactly -- same rollout for the same seed -- with the
    render domain as the only difference (materials/lights in domA, the
    registered camera in domB)."""
    base = _spec("occlusion_reemergence", horizon_s=2.0)
    for dom in ("occlusion_reemergence_domA", "occlusion_reemergence_domB"):
        assert (ROOT / f"scenes/{dom}.xml").exists() and dom in SCENARIO_PROMPTS
        v = _spec(dom, horizon_s=2.0)
        assert (v.lam, v.n_reference, v.perturb_mode, v.target_property) == (base.lam, base.n_reference, base.perturb_mode, base.target_property)
        a, b = rollout(base, 1000, base.scene), rollout(v, 1000, v.scene)
        np.testing.assert_array_equal(a.pos, b.pos)
        cfg, cfg0 = sg.get(dom), sg.get("occlusion_reemergence")
        assert cfg["deceleration"] == cfg0["deceleration"] and cfg["mode"] == "plane_z"
    assert sg.get("occlusion_reemergence_domA")["camera"] == sg.get("occlusion_reemergence")["camera"]
    assert sg.get("occlusion_reemergence_domB")["camera"] != sg.get("occlusion_reemergence")["camera"]
    assert sg.get("occlusion_reemergence_domB")["occluder_bounds"][1] > sg.get("occlusion_reemergence")["occluder_bounds"][1]


def test_new_scenarios_are_fully_registered():
    for name in ("occlusion_reemergence", "block_stack"):
        assert (ROOT / f"configs/manifests/{name}.yaml").exists()
        assert (ROOT / f"scenes/{name}.xml").exists()
        assert name in SCENARIO_PROMPTS and len(SCENARIO_PROMPTS[name]) > 40
        cfg = sg.get(name)
        assert cfg["mode"] == "plane_z" and cfg["object_radius"] == sg.BALL_RADIUS
    # The tracker gets the span widened by one radius each side (partial
    # occlusion counts as occluded for visibility); the measured fully-
    # hidden span is kept as its own constant.
    lo, hi = sg.OCCLUSION_REEMERGENCE_WALL_BOUNDS
    assert sg.get("occlusion_reemergence")["occluder_bounds"] == (lo - sg.BALL_RADIUS, hi + sg.BALL_RADIUS)
    assert _spec("block_stack").target_property == "P3"
    assert _spec("occlusion_reemergence").target_property == "P2"


def test_obj0_perturbation_leaves_every_other_body_at_its_keyframe():
    """The runner bug-class this mode exists for: any perturbation of the
    resting stack's own initial state makes the ensemble's ambiguity about
    the tower, not the ball. Body 0 must vary across seeds; bodies 1 and 2
    must start byte-identical across seeds."""
    sp = _spec("block_stack", horizon_s=0.1)
    starts = np.stack([rollout(sp, 1000 + s, sp.scene).pos[0] for s in range(6)])   # (6, K, 3)
    assert starts.shape[1] == 3
    assert np.ptp(starts[:, 0, 0]) > 0.05, "the ball's own start must be perturbed"
    assert np.allclose(starts[:, 1], starts[0, 1]) and np.allclose(starts[:, 2], starts[0, 2])
    sp_all = _spec("block_stack", horizon_s=0.1, perturb_mode="velocity_x_only")
    starts_all = np.stack([rollout(sp_all, 1000 + s, sp_all.scene).pos[0] for s in range(6)])
    assert np.ptp(starts_all[:, 2, 0]) > 0.05, "control: the plain mode DOES move the top block"


def test_block_stack_reference_ensemble_properties():
    """What the P3 scoring relies on: the bottom cube keeps its center on
    the ball's plane (z = 0.15, it slides rather than tips), the strike
    lands after the 1.0s prefix, and the post topples in nearly every
    reference (97/100 measured; the rest is genuine ambiguity)."""
    sp = _spec("block_stack", n_reference=20)
    hits, tips, z_dev, tilt_b = [], 0, 0.0, 0.0
    for s in range(20):
        tr = rollout(sp, 1000 + s, sp.scene)
        d = np.linalg.norm(tr.pos[:, 0] - tr.pos[:, 1], axis=1)
        hits.append(tr.t[np.argmin(d)])
        assert d.min() > 0.27, "ball and cube must not interpenetrate in the reference physics"
        z_dev = max(z_dev, np.abs(tr.pos[:, 1, 2] - sg.BALL_RADIUS).max())
        tilt_b = max(tilt_b, _tilt_deg(tr.quat[:, 1]).max())
        tips += _tilt_deg(tr.quat[90, 2]) > 45
    assert min(hits) > 1.2, f"impact must land after the 1.0s prefix, got {min(hits):.2f}s"
    assert z_dev < 0.03, f"bottom cube center must stay on the z={sg.BALL_RADIUS} plane, deviated {z_dev:.3f}m"
    assert tilt_b < 10, f"bottom cube must slide, not tip: max tilt {tilt_b:.1f} deg"
    assert tips >= 18, f"post must topple by t=3s in nearly every reference, got {tips}/20"


def test_occlusion_reemergence_every_reference_reemerges_inside_the_window():
    """The whole point vs. occlusion_corridor: no reference is allowed to
    stop behind the wall. Prefix fully visible (x(1s) below the measured
    hidden span), hidden ~1s, back out before t=2.7s, for every seed."""
    sp = _spec("occlusion_reemergence")
    lo, hi = sg.OCCLUSION_REEMERGENCE_WALL_BOUNDS
    for s in range(sp.n_reference):
        x = rollout(sp, 1000 + s, sp.scene).pos[:, 0, 0]
        assert x[30] < lo, f"seed {s}: ball already hidden at the end of the prefix (x={x[30]:.2f})"
        exit_idx = np.where(x >= hi)[0]
        assert len(exit_idx) and exit_idx[0] / sp.fps < 2.7, f"seed {s}: did not re-emerge before 2.7s"
        enter_idx = np.where(x > lo)[0][0]
        assert 0.8 <= (exit_idx[0] - enter_idx) / sp.fps <= 1.5, "hidden interval should be ~1s"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
