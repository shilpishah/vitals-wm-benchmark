"""Soft-body (MuJoCo flex) support -- the state side of AGENT.md M9, the
soft-body measurement modality. A deformable body has a configuration,
not a pose: its Trajectory row is its vertex CENTROID (so R1/R3/R4 read it
unchanged), the vertex cloud is kept in meta["flex_vertices"][name] as
(T, nvert, 3), and SHAPE descriptors are attached as `Trajectory.shape`.

The descriptor set is deliberately the one both measurement paths can
compute on identical footing:

    SHAPE_DESCRIPTORS = ("proj_area_px", "proj_axis_ratio")

  * proj_area_px  -- area, in pixels, of the body's silhouette in the fixed
                     scene camera: state space = convex hull of the
                     projected vertices; pixels = mask area. This is the
                     conservation quantity R6 reads (ratio to its own
                     frame-0 value against the reference band).
  * proj_axis_ratio -- minor/major principal-axis ratio of that silhouette
                     (PCA of the projected vertices / of the mask pixels):
                     scale-free, so it isolates SHAPE from size.

State-only extras (3-D, exact) go to meta["flex_extras"]: tetrahedral
volume and the three principal extents -- used for GATE 1 (state space)
diagnostics and for the mutants, never compared against pixels.

Why camera-plane descriptors and not 3-D ones as the canonical set: a
monocular pixel path cannot recover volume; comparing a pixel-derived
area against a state-derived volume would compare two different
quantities and call the gap "instrument error". Same-footing first; the
3-D quantities stay available where they are exact.

MuJoCo facts this relies on (verified 2026-09-13 on MuJoCo 3.12, see
AGENT.md M9): each flex vertex is a body `<flexname>_<i>` with three
SLIDE joints whose qpos is the displacement from the vertex's REST
position (model.body_pos of that body); model.flex_vertadr/vertnum index
the vertex block; model.flex_elem holds the tetrahedra (dim 3).
"""
from __future__ import annotations
import numpy as np

SHAPE_DESCRIPTORS = ("proj_area_px", "proj_axis_ratio")


def flex_objects(model):
    """[(name, vert_adr, vert_num, elems (ne,4) or None)] for every flex
    in the model, in flex-id order."""
    import mujoco
    out = []
    for f in range(model.nflex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, f) or f"flex{f}"
        adr, num = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
        elems = None
        if int(model.flex_dim[f]) == 3:
            ea, en = int(model.flex_elemdataadr[f]), int(model.flex_elemnum[f])
            elems = np.array(model.flex_elem[ea: ea + en * 4]).reshape(en, 4)   # LOCAL vertex indices
        out.append((name, adr, num, elems))
    return out


def flex_vertex_body_ids(model):
    """Set of body ids that are flex vertices -- these must never be
    treated as tracked rigid objects by the runner."""
    return set(int(b) for b in model.flex_vertbodyid[: model.nflexvert])


def free_body_count(model):
    """Number of FREE joints (rigid free bodies). The old `nq // 7` rule is
    wrong the moment a flex is present (its vertices contribute 3 dof
    each), so free bodies are counted from joint types."""
    import mujoco
    return int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))


def flex_vertex_qpos_slices(model, vert_adr, vert_num):
    """[(qpos_adr, body_id)] per vertex, in vertex order."""
    out = []
    for i in range(vert_adr, vert_adr + vert_num):
        b = int(model.flex_vertbodyid[i])
        j = int(model.body_jntadr[b])
        out.append((int(model.jnt_qposadr[j]), b))
    return out


def set_flex_vertices(model, data, vert_adr, vert_num, vertices_world):
    """Place a flex by writing each vertex body's slide-joint qpos as the
    displacement from its rest position. vertices_world: (nvert, 3)."""
    for (qadr, b), v in zip(flex_vertex_qpos_slices(model, vert_adr, vert_num), vertices_world):
        data.qpos[qadr: qadr + 3] = v - model.body_pos[b]


def shift_flex(model, data, vert_adr, vert_num, dpos, dvel):
    """Perturb a whole flex rigidly: the same displacement/velocity on
    every vertex (Sigma acts on the actor's initial state; the material
    is part of s_0 -- AGENT.md M9 decision 5)."""
    for (qadr, b) in flex_vertex_qpos_slices(model, vert_adr, vert_num):
        data.qpos[qadr: qadr + 3] += dpos
        j = int(model.body_jntadr[b]); dadr = int(model.jnt_dofadr[j])
        data.qvel[dadr: dadr + 3] += dvel


def tet_volume(vertices, elems):
    """Exact volume of a tetrahedral mesh. vertices (nvert, 3), elems (ne, 4)."""
    a, b, c, d = (vertices[elems[:, k]] for k in range(4))
    return float(np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)).sum() / 6.0)


def principal_extents(vertices):
    """sqrt of the covariance eigenvalues, descending -- the cloud's three
    principal half-extents (up to a constant), in metres."""
    c = vertices - vertices.mean(axis=0)
    w = np.linalg.eigvalsh(c.T @ c / max(len(c) - 1, 1))
    return np.sqrt(np.clip(w[::-1], 0, None))


def project_points(pts, cam_pos, cam_mat, fovy_deg, width, height):
    """(n, 3) world -> (n, 2) pixel (x, y), same pinhole model as
    reconstruct._project_raw; points behind the camera are dropped."""
    rel = (pts - cam_pos) @ cam_mat            # camera frame: columns right, up, -forward
    depth = -rel[:, 2]
    ok = depth > 1e-6
    f = height / (2.0 * np.tan(np.radians(fovy_deg) / 2.0))
    px = width / 2.0 + f * rel[ok, 0] / depth[ok]
    py = height / 2.0 - f * rel[ok, 1] / depth[ok]
    return np.stack([px, py], axis=1)


def _hull_area(xy):
    """Convex-hull area of 2-D points (monotone chain), 0 for < 3 points."""
    pts = np.unique(xy, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _axis_ratio_2d(xy):
    if len(xy) < 3:
        return float("nan")
    c = xy - xy.mean(axis=0)
    w = np.linalg.eigvalsh(c.T @ c / max(len(c) - 1, 1))
    w = np.clip(w, 0, None)
    return float(np.sqrt(w[0] / w[1])) if w[1] > 1e-12 else float("nan")


def _hull_vertices(xy):
    """Convex hull of 2-D points (monotone chain), counter-clockwise; None
    for < 3 distinct points."""
    pts = np.unique(xy, axis=0)
    if len(pts) < 3:
        return None
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1])


def hull_mask(xy, width, height):
    """Rasterise the convex hull of projected points into a (height, width)
    boolean mask -- pixel centres inside the hull. The state-side
    silhouette, on the SAME footing as a tracked mask."""
    hull = _hull_vertices(xy)
    mask = np.zeros((height, width), bool)
    if hull is None:
        return mask
    x0, y0 = np.floor(hull.min(axis=0)).astype(int); x1, y1 = np.ceil(hull.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, width - 1), min(y1, height - 1)
    if x1 < x0 or y1 < y0:
        return mask
    ys, xs = np.mgrid[y0: y1 + 1, x0: x1 + 1]
    px, py = xs + 0.5, ys + 0.5
    inside = np.ones(px.shape, bool)
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]; bx, by = hull[(i + 1) % n]
        inside &= (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0
    mask[y0: y1 + 1, x0: x1 + 1] = inside
    return mask


def descriptors_from_vertices(vertices, cam_pos, cam_mat, fovy_deg, width, height):
    """State-space SHAPE_DESCRIPTORS for one frame from the vertex cloud:
    [proj_area_px, proj_axis_ratio]. The projected vertices' convex hull is
    RASTERISED and the descriptors read from that mask with the very same
    code as the pixel path (descriptors_from_mask) -- found 2026-09-14: a
    PCA of the projected vertex cloud (a volumetric sample, denser inside)
    disagreed with the filled silhouette's PCA by up to 0.07 in axis
    ratio on a squashed body; hull-mask vs rendered mask agree within
    ~0.01."""
    xy = project_points(vertices, cam_pos, cam_mat, fovy_deg, width, height)
    return descriptors_from_mask(hull_mask(xy, width, height))


def descriptors_from_mask(mask):
    """Pixel-path SHAPE_DESCRIPTORS for one frame from a boolean mask:
    [area_px, axis_ratio]. NaN both when the mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.array([np.nan, np.nan])
    xy = np.stack([xs, ys], axis=1).astype(np.float64)
    return np.array([float(len(xs)), _axis_ratio_2d(xy)])


def phi_refs_path(root, scenario_name):
    """results/phi_refs_<scenario>.npz -- the Phi-measured reference band
    (scripts/build_phi_references.py)."""
    import pathlib
    return pathlib.Path(root) / "results" / f"phi_refs_{scenario_name}.npz"


def load_phi_refs(path, names=("ball",)):
    """The Phi-measured reference band as Trajectory objects (pos, present
    and SHAPE all measured through the tracker), for calibrating and
    scoring the pixel-side R6/R7 (AGENT.md T10). None if absent. A
    reference whose frame-0 shape is undefined is dropped (no ratio)."""
    import pathlib
    from ..types import Trajectory
    path = pathlib.Path(path)
    if not path.exists():
        return None
    z = np.load(path)
    t, shape, pos, present = z["t"], z["shape"], z["pos"], z["present"]
    out = []
    for i in range(shape.shape[0]):
        if not np.isfinite(shape[i, 0, 0]):
            continue
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(t), 1, 1))
        out.append(Trajectory(t=t.copy(), pos=pos[i][:, None, :].copy(), quat=quat,
                              present=present[i][:, None].copy(), names=list(names),
                              shape=shape[i][:, None, :].copy()))
    return out or None


def attach_shape(traj, cam_pos, cam_mat, fovy_deg, width, height):
    """Fill `traj.shape` (T, K, 2) for every object that has a vertex cloud
    in meta["flex_vertices"]; rigid objects get NaN rows (no shape state).
    Idempotent; returns traj."""
    verts = traj.meta.get("flex_vertices", {})
    if not verts:
        return traj
    T, K = traj.T, traj.K
    shape = np.full((T, K, len(SHAPE_DESCRIPTORS)), np.nan)
    for k, name in enumerate(traj.names):
        if name in verts:
            V = verts[name]
            for t in range(min(T, V.shape[0])):
                if traj.present[t, k]:
                    shape[t, k] = descriptors_from_vertices(V[t], cam_pos, cam_mat, fovy_deg, width, height)
    traj.shape = shape
    return traj
