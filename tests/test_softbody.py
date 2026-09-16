"""Soft-body modality (AGENT.md M9, 2026-09-13/14): the state path, the
renderer's flex handling, the two new channels and the four new mutants.
Pins what the groundwork established by simulation so a later edit cannot
silently change what a soft-body reference ensemble means.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import yaml
from vitals.types import EpisodeSpec, Trajectory
from vitals.physics.runner import rollout
from vitals.physics import softbody as sb
from vitals.mutants import library as mut
from vitals.detect.statistics import sigma_conservation, sigma_shape, sigma_kinematic, kinematic_axes_for
from vitals.detect.events import PRECEDENCE
from vitals.phi import scene_geometry as sg
from vitals.adapters.scenario_prompts import SCENARIO_PROMPTS

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _spec(horizon_s=1.5):
    m = yaml.safe_load((ROOT / "configs/manifests/soft_drop.yaml").read_text())
    return EpisodeSpec(name=m["name"], scene=str(ROOT / m["scene"]), target_property=m["target_property"],
                       band=m["band"], lam=m["lam"], n_reference=m["n_reference"], horizon_s=horizon_s,
                       fps=m["fps"], seed=m.get("seed", 0), perturb_mode=m.get("perturb_mode", "full"))


@pytest.fixture(scope="module")
def refs():
    spec = _spec()
    return spec, [rollout(spec, 1000 + i, spec.scene) for i in range(6)]


def test_soft_drop_is_registered():
    assert (ROOT / "scenes/soft_drop.xml").exists()
    cfg = sg.get("soft_drop")
    assert cfg["soft_body"] is True and cfg["object_radius"] == sg.SOFT_DROP_RADIUS
    assert "soft_drop" in SCENARIO_PROMPTS and len(SCENARIO_PROMPTS["soft_drop"]) > 40
    assert kinematic_axes_for("soft_drop") == (0, 2)
    assert PRECEDENCE == ["R5", "R1", "R6", "R2", "R3", "R7"]


def test_soft_ramp_is_registered_and_rolls():
    """M9.1: the slope scene -- a deformable body that keeps a scorable
    trajectory after contact (soft_drop's does not)."""
    assert (ROOT / "scenes/soft_ramp.xml").exists()
    cfg = sg.get("soft_ramp")
    assert cfg["soft_body"] is True and cfg["mode"] == "plane_pieces"
    assert cfg["reconstruct_kwargs"]["with_shape"] is True
    assert "soft_ramp" in SCENARIO_PROMPTS and kinematic_axes_for("soft_ramp") == (0, 2)
    m = yaml.safe_load((ROOT / "configs/manifests/soft_ramp.yaml").read_text())
    spec = EpisodeSpec(name=m["name"], scene=str(ROOT / m["scene"]), target_property=m["target_property"],
                       band=m["band"], lam=m["lam"], n_reference=m["n_reference"], horizon_s=4.0,
                       fps=m["fps"], seed=0, perturb_mode=m.get("perturb_mode", "full"))
    tr = rollout(spec, 1000, spec.scene)
    assert tr.K == 1 and tr.names == ["blob"]
    x = tr.pos[:, 0, 0]
    assert x[-1] - x[0] > 3.0                       # rolls the ramp and onto the floor within 4 s
    assert x[-1] - x[90] > 0.3                       # still moving after 3 s: a post-contact trajectory to score
    vol = tr.meta["flex_extras"]["blob"][:, 0]
    assert np.nanmax(np.abs(vol / vol[0] - 1)) < 0.12
    ext = tr.meta["flex_extras"]["blob"][:, 1:]
    assert np.nanmin(ext[1:, 2] / ext[1:, 0]) < 0.9  # it actually deforms


def test_runner_tracks_the_flex_as_one_object(refs):
    spec, R = refs
    tr = R[0]
    assert tr.names == ["blob"] and tr.K == 1
    V = tr.meta["flex_vertices"]["blob"]
    assert V.shape == (tr.T, 297, 3)
    np.testing.assert_allclose(tr.pos[:, 0], V.mean(axis=1))
    # falls from 1.2m, lands (centroid near its rest radius) and moves +x
    assert tr.pos[0, 0, 2] == pytest.approx(1.2, abs=0.02)
    assert tr.pos[-1, 0, 2] < 0.25 and tr.pos[-1, 0, 0] > 0.3
    # tetrahedral volume conserved by the material to a few percent
    vol = tr.meta["flex_extras"]["blob"][:, 0]
    assert np.nanmax(np.abs(vol / vol[0] - 1)) < 0.08
    # the initial-conditions-only ensemble: seeds differ in x, not in the material
    assert R[0].pos[-1, 0, 0] != R[1].pos[-1, 0, 0]


def test_rollout_is_reproducible(refs):
    spec, R = refs
    again = rollout(spec, 1000, spec.scene)
    np.testing.assert_array_equal(again.meta["flex_vertices"]["blob"], R[0].meta["flex_vertices"]["blob"])


def test_renderer_labels_flex_and_descriptors_agree(refs):
    from vitals.render.mujoco_renderer import MujocoRenderer, camera_pose
    spec, R = refs
    tr = R[0].copy()
    cam = sg.get("soft_drop")["camera"][0]
    r = MujocoRenderer(spec.scene)
    frames, gt = r.render(tr, cameras=[cam])
    mask = gt.segmentation == 0
    assert mask.sum(axis=(1, 2)).min() > 500          # visible in every frame
    assert (gt.segmentation > 0).sum() == 0            # no other object
    cam_pos, cam_mat, fovy = camera_pose(spec.scene, cam)
    np.testing.assert_allclose(cam_pos, gt.cam_pos, atol=1e-9)
    sb.attach_shape(tr, cam_pos, cam_mat, fovy, r.width, r.height)
    assert tr.shape.shape == (tr.T, 1, 2)
    pix = np.array([sb.descriptors_from_mask(m) for m in mask])
    rel_area = np.abs(tr.shape[:, 0, 0] / pix[:, 0] - 1)
    assert rel_area.max() < 0.04                       # hull-of-vertices vs mask area
    assert np.abs(tr.shape[:, 0, 1] - pix[:, 1]).max() < 0.03


def _attached(tr):
    from vitals.render.mujoco_renderer import camera_pose
    cam_pos, cam_mat, fovy = camera_pose(tr.meta["scene"], sg.get("soft_drop")["camera"][0])
    return sb.attach_shape(tr, cam_pos, cam_mat, fovy, 640, 360)


def test_channels_are_nan_for_rigid_and_fire_on_mutants(refs):
    spec, R = refs
    R = [_attached(t.copy()) for t in R]
    cand, others = R[0], R[1:]
    # rigid trajectory: undefined, never fires
    rigid = Trajectory(cand.t, cand.pos, cand.quat, cand.present, cand.names, meta={})
    assert np.all(np.isnan(sigma_conservation(rigid, others)))
    assert np.all(np.isnan(sigma_shape(rigid, others)))
    # a clean reference sits inside the band
    s_null = sigma_conservation(cand, others)
    assert np.nanmax(s_null) < 4.0
    # volume_leak: area ratio leaves the band after t*, centroid untouched
    m = mut.volume_leak(cand, t_star=0.5, rate=0.3)
    assert m.risk_expected == "R6" and m.traj.shape is None
    _attached(m.traj)
    np.testing.assert_allclose(m.traj.pos, cand.pos, atol=1e-9)
    assert np.nanmax(sigma_conservation(m.traj, others)[20:]) > 10 * np.nanmax(s_null)
    # frozen_deformation: shape leaves the band, centroid untouched
    m = mut.frozen_deformation(cand)
    assert m.risk_expected == "R7"
    _attached(m.traj)
    np.testing.assert_allclose(m.traj.pos, cand.pos, atol=1e-6)
    assert np.nanmax(sigma_shape(m.traj, others)) > np.nanmax(sigma_shape(cand, others))
    # wrong_damping: centroid z changes after contact, the cloud moves with it (shape preserved)
    m = mut.wrong_damping(cand, factor=0.5)
    assert m.risk_expected == "R3"
    V, V0 = m.traj.meta["flex_vertices"]["blob"], cand.meta["flex_vertices"]["blob"]
    np.testing.assert_allclose(V[..., :2], V0[..., :2], atol=1e-9)
    np.testing.assert_allclose(V.mean(axis=1), m.traj.pos[:, 0], atol=1e-9)
    assert np.abs(m.traj.pos[:, 0, 2] - cand.pos[:, 0, 2]).max() > 0.002   # the body barely bounces (scene comment)
    # the centroid mutants carry the cloud along: wrong_gravity in flight is the scene's R3 test
    m = mut.wrong_gravity(cand, factor=0.6)
    V = m.traj.meta["flex_vertices"]["blob"]
    np.testing.assert_allclose(V.mean(axis=1), m.traj.pos[:, 0], atol=1e-9)
    _attached(m.traj)
    assert np.nanmax(sigma_kinematic(m.traj, others, axes=(0, 2))) > 10 * np.nanmax(sigma_kinematic(cand, others, axes=(0, 2)))
    np.testing.assert_allclose(m.traj.shape[:, 0, 1], cand.shape[:, 0, 1], atol=0.08)   # shape untouched (perspective moves the ratio a little with height)


def test_phi_reconstruct_attaches_mask_shape():
    """The pixel side of the modality: reconstruct_trajectory(with_shape=True)
    fills Trajectory.shape from the tracked masks (area px, axis ratio),
    NaN where the object is not visible; merge/slice carry it along."""
    from vitals.phi import reconstruct as recon
    from vitals.adapters.video_model import _slice_continuation
    H, W = 90, 160
    yy, xx = np.mgrid[:H, :W]
    masks = {}
    for i in range(10):
        r = 12 - i                                      # shrinking disk
        masks[i] = (xx - 80) ** 2 + (yy - 45) ** 2 <= r * r
    del masks[5]                                        # one unseen frame
    cam_pos = np.array([0.0, -3.0, 1.0]); cam_mat = np.eye(3)
    tr = recon.reconstruct_trajectory(masks, fps=30, cam_pos=cam_pos, cam_mat=cam_mat, fovy_deg=45,
                                      width=W, height=H, plane_z=0.1, T=10, with_shape=True)
    assert tr.shape.shape == (10, 1, 2)
    assert np.isnan(tr.shape[5, 0]).all()
    areas = tr.shape[[0, 1, 2, 3], 0, 0]                # r = 12..9: still above half the largest mask
    assert np.all(np.diff(areas) < 0) and abs(tr.shape[0, 0, 1] - 1.0) < 0.05
    # the tracker's visibility rule: a mask under half the largest seen so
    # far is not a size measurement (r <= 8 here) -> NaN, never a tiny area
    assert np.isnan(tr.shape[[4, 6, 7, 8, 9], 0, 0]).all()
    rigid = recon.reconstruct_trajectory(masks, fps=30, cam_pos=cam_pos, cam_mat=cam_mat, fovy_deg=45,
                                         width=W, height=H, plane_z=0.1, T=10)
    assert rigid.shape is None
    merged = recon.merge_trajectories([tr, rigid])
    assert merged.shape.shape == (10, 2, 2) and np.isnan(merged.shape[:, 1]).all()
    cont = _slice_continuation(merged, 3, 1 / 30)
    assert cont.shape.shape == (7, 2, 2) and cont.shape[0, 0, 0] == tr.shape[3, 0, 0]


def test_drift_moves_the_resting_body_and_its_cloud(refs):
    spec, R = refs
    R = [_attached(t.copy()) for t in R]
    cand, others = R[0], R[1:]
    m = mut.drift(cand, t_star=0.5, velocity=(0.3, 0.0, 0.0))
    assert m.risk_expected == "R3"
    V = m.traj.meta["flex_vertices"]["blob"]
    np.testing.assert_allclose(V.mean(axis=1), m.traj.pos[:, 0], atol=1e-9)
    np.testing.assert_allclose(m.traj.pos[:15], cand.pos[:15])               # untouched before t*
    assert m.traj.pos[-1, 0, 0] - cand.pos[-1, 0, 0] == pytest.approx(0.3 * (cand.t[-1] - cand.t[15]), abs=1e-6)
    _attached(m.traj)
    np.testing.assert_allclose(m.traj.shape[:, 0, 1], cand.shape[:, 0, 1], atol=0.08)   # shape untouched
    assert np.nanmax(sigma_kinematic(m.traj, others, axes=(0, 2))) > 3 * np.nanmax(sigma_kinematic(cand, others, axes=(0, 2)))


def test_mutants_do_not_alias_the_source(refs):
    spec, R = refs
    src = R[0].meta["flex_vertices"]["blob"].copy()
    mut.volume_leak(R[0], t_star=0.2)
    mut.wrong_stiffness(R[0])
    np.testing.assert_array_equal(R[0].meta["flex_vertices"]["blob"], src)
