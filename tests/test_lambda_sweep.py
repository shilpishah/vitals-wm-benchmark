"""scripts/run_lambda_sweep.py -- fast smoke tests, synthetic backend + tiny
n throughout (2026-08, M7's lambda-sweep pass). Not a substitute for
actually running the real sweep (results/lambda_sweep_*.json, AGENT.md's
own writeup) -- this just guards the plumbing (calibration, GATE 1a/1b,
Truth population, both baselines, one full level) against silently
breaking on a future change, the same "smoke test the script, not just
its numbers" discipline as tests/test_validity_analysis.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import math
import run_lambda_sweep as sweep
from vitals.physics import make_backend

SPEC = sweep.load_spec("configs/manifests/occlusion_corridor.yaml")
ROLL = make_backend("synthetic", scene=SPEC.scene)


def test_calibrate_returns_finite_thresholds_for_both_channels():
    refs, stats, thetas, stats_video, thetas_video, dt, t_max = sweep.calibrate(ROLL, SPEC)
    assert len(refs) == SPEC.n_reference
    assert set(thetas) == {"R1", "R3"}
    assert math.isfinite(float(thetas["R1"][0]))
    assert t_max > 0


def test_build_demo_cases_occlusion_corridor_uses_duplicate_not_wrong_gravity():
    base = ROLL(SPEC, 9000)
    cases = sweep.build_demo_cases(SPEC, base)
    labels = [label for label, _ in cases]
    assert "duplicate" in labels
    assert "wrong_gravity" not in labels
    assert any(label.startswith("jitter sig=0.05") for label in labels)


def test_gate1a_and_gate1b_run_end_to_end_small_n():
    refs, stats, thetas, stats_video, thetas_video, dt, t_max = sweep.calibrate(ROLL, SPEC)
    gate_1a, held_out = sweep.run_gate1a(ROLL, SPEC, refs, stats, thetas, dt, t_max, n=8)
    assert 0.0 <= gate_1a["false_termination_rate"] <= 1.0
    assert len(held_out) == 8

    gate_1b = sweep.run_gate1b(SPEC, refs, stats, thetas, dt, t_max, held_out[0])
    assert len(gate_1b) >= 4
    assert all("ok" in c for c in gate_1b)
    # null must never fire anything -- the one case with a deterministic answer
    null_case = next(c for c in gate_1b if c["label"] == "null")
    assert null_case["risk_fired"] is None


def test_truth_and_baseline_populations_produce_valid_events():
    refs, stats, thetas, stats_video, thetas_video, dt, t_max = sweep.calibrate(ROLL, SPEC)
    truth = sweep.truth_population(ROLL, SPEC, refs, stats, thetas, dt, t_max, n=6)
    assert len(truth) == 6
    assert all(e.censored or e.time <= t_max + 1e-9 for e in truth)

    cv = sweep.baseline_population(ROLL, SPEC, refs, stats, thetas, dt, t_max,
                                   "constant_velocity", n=4)
    cls = sweep.baseline_population(ROLL, SPEC, refs, stats, thetas, dt, t_max,
                                    "copy_last_state", n=4)
    assert len(cv) == 4 and len(cls) == 4


def test_run_one_level_end_to_end_and_json_serializable():
    import json
    level = sweep.run_one_level(ROLL, SPEC, lam_multiplier=1.0,
                                n_gate1a=6, n_truth=6, n_baseline=4)
    assert set(sweep.BASE_CONDITIONS) <= set(level["vi50"])
    assert level["lam"] == SPEC.lam
    assert ("constant_velocity", "copy_last_state") == \
        tuple(next(iter(level["deltas"])).split("_vs_"))
    json.dumps(level)   # must not raise -- everything persisted must be plain JSON types


def test_real_model_population_reuses_cached_trajectories_no_regen():
    """The whole point of caching: a real-model 'population' here is just
    scoring already-reconstructed trajectories against fresh thetas -- no
    generate_fn/phi_fn call, no network. Round-trips a couple of synthetic
    Trajectories through save/load_trajectory then confirms real_model_
    population produces one Event per cached episode, sliced to their own
    (shorter-than-reference) length. Uses tempfile (not pytest's tmp_path)
    so this file stays runnable standalone via its own __main__ block, the
    same convention every other test file in this project follows."""
    import tempfile
    from vitals.adapters.video_utils import save_trajectory
    from vitals.types import Trajectory

    refs, stats, thetas, stats_video, thetas_video, dt, t_max = sweep.calibrate(ROLL, SPEC)
    n_short = 20   # shorter than the reference ensemble's own horizon
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = pathlib.Path(tmp) / "occlusion_corridor_fakemodel"
        cache_dir.mkdir()
        for seed in (1, 2, 3):
            r = refs[0]
            traj = Trajectory(t=r.t[:n_short].copy(), pos=r.pos[:n_short].copy(),
                              quat=r.quat[:n_short].copy(), present=r.present[:n_short].copy(),
                              names=list(r.names), meta={})
            save_trajectory(traj, cache_dir / f"seed{seed}.npz")

        trajs = sweep.load_cached_population(cache_dir)
        assert len(trajs) == 3 and all(t.T == n_short for t in trajs)

        events = sweep.real_model_population(stats, thetas, dt, refs, trajs)
        assert len(events) == 3
        assert all(e.censored or e.time <= n_short * dt + 1e-9 for e in events)

        # a nonexistent cache dir is a normal empty result, not an error
        assert sweep.load_cached_population(pathlib.Path(tmp) / "does_not_exist") == []


if __name__ == "__main__":
    import inspect
    fns = [obj for name, obj in sorted(globals().items())
           if name.startswith("test_") and inspect.isfunction(obj)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
