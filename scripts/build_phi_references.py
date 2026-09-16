"""Phi-measured reference band for the soft-body channels (AGENT.md M9,
2026-09-14).

The rigid channels compare a pixel-measured candidate against STATE-SPACE
references and accept the centroid's small reconstruction floor. The
soft-body descriptors cannot: measured on soft_drop's own GATE 2 nulls,
SAM2's tracked mask is ~13% larger than the rendered silhouette once the
body rests on the floor (the contact shadow is absorbed into the mask;
the renderer's own segmentation matches the state-side hull to 1-2%), so
an area ratio against state references fires R6 on every null at first
contact. That is a systematic instrument error, and 8.3's answer to
systematic error is that the REFERENCE absorbs it: this script renders
the scenario's own reference rollouts, runs each through the identical
Phi path (`run_gate2_episode` on the GATE 2 app) and stores the
pixel-measured shape (and centroid) traces. `run_gate2.py` then
calibrates the L1 thresholds for R6/R7 on THIS band and scores L1
candidates against it -- same instrument on both sides of the comparison.

    VITALS_MANIFEST=configs/manifests/soft_drop.yaml VITALS_PHI_APP=vitals-phi-gate2 \\
        python scripts/build_phi_references.py            # -> results/phi_refs_soft_drop.npz

Seeds are GATE 1/2's own reference seeds (1000+i), rendered at the GATE 2
renderer size; episodes are uploaded under `phiref_<scenario>_<i>`.
"""
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import modal

import run_gate2 as g

OUT = g.ROOT / "results" / f"phi_refs_{g.SPEC.name}.npz"


def main():
    M = int(os.environ.get("VITALS_PHI_REF_M", g.M))
    refs = [g.roll(g.SPEC, 1000 + i) for i in range(M)]
    names = []
    print(f"rendering {M} references of {g.SPEC.name} at {g.renderer.width}x{g.renderer.height} ...")
    for i, r in enumerate(refs):
        name = f"phiref_{g.SPEC.name}_{i}"
        g.render_episode(g.truncate(r, r.T), name)
        names.append(name)
    g.upload_all(names)
    fn = modal.Function.from_name(g.APP_NAME, "run_gate2_episode")
    calls = [(i, fn.spawn(name=n)) for i, n in enumerate(names)]
    T = refs[0].T
    shape = np.full((M, T, 2), np.nan)
    pos = np.full((M, T, 3), np.nan)
    present = np.zeros((M, T), bool)
    ok = 0
    for i, c in calls:
        try:
            res = c.get()
        except Exception as e:
            print(f"  ref {i} FAILED: {e}")
            continue
        tr = g.traj_from_modal_result(res)
        n = min(T, tr.T)
        pos[i, :n] = tr.pos[:n, 0]
        present[i, :n] = tr.present[:n, 0]
        if tr.shape is not None:
            shape[i, :n] = tr.shape[:n, 0]
        ok += 1
    np.savez(OUT, t=refs[0].t, shape=shape, pos=pos, present=present, seeds=np.arange(1000, 1000 + M),
             width=g.renderer.width, height=g.renderer.height, app=g.APP_NAME)
    st = refs[0].shape[:, 0, 0]
    ratio_state = np.stack([r.shape[:, 0, 0] / r.shape[0, 0, 0] for r in refs])
    ratio_phi = shape[..., 0] / shape[:, :1, 0]
    print(f"{ok}/{M} references measured -> {OUT}")
    for t in (0.4, 1.0, 2.0, 3.0):
        k = int(t * g.SPEC.fps)
        print(f"  t={t}s  area ratio: state {np.nanmean(ratio_state[:, k]):.3f}±{np.nanstd(ratio_state[:, k]):.3f}"
              f"   phi {np.nanmean(ratio_phi[:, k]):.3f}±{np.nanstd(ratio_phi[:, k]):.3f}"
              f"   axis ratio: state {np.nanmean([r.shape[k, 0, 1] for r in refs]):.3f}  phi {np.nanmean(shape[:, k, 1]):.3f}±{np.nanstd(shape[:, k, 1]):.3f}")


if __name__ == "__main__":
    main()
