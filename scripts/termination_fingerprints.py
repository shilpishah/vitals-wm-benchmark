"""Termination fingerprints with uncertainty (AGENT.md M10.1, 2026-09-16).

The cross-model finding M10 established -- cosmos3super dies by kinematic
divergence (R3), cosmos3nano by re-framing the scene (R5), in every render
domain -- was read off point estimates. This script quantifies it:

  1. Per model x domain cell: the cause-specific CUMULATIVE INCIDENCE at
     t_max (Aalen-Johansen; with no censoring before t_max it equals the
     terminal-cause proportion) with a bootstrap-over-episodes 95% CI.
  2. Within each matched domain: the model x channel association
     (chi-square on the R3/R5/other counts, exact permutation p-value).
  3. Across the repeated domains: a multinomial logit of terminal cause
     on model, domain and their interaction, fit by maximum likelihood
     (numpy/scipy only), with likelihood-ratio tests for each term -- the
     competing-risks analogue of  cause ~ model + domain + model x domain.
  4. The effect-size contrast the claim rests on: the model effect on
     P(R5) versus the render-domain effect on P(R5).

Reads results/l0_demo_<scenario>_<model>.json; writes
results/termination_fingerprints.json and prints the tables.

    python3 scripts/termination_fingerprints.py
    python3 scripts/termination_fingerprints.py --models cosmos3nano cosmos3super \\
        --domains occlusion_reemergence_domA occlusion_reemergence_domB
"""
import sys, pathlib, json, argparse, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAUSES = ["R1", "R3", "R5", "other"]


def load(scenario, model):
    d = json.load(open(ROOT / "results" / f"l0_demo_{scenario}_{model}.json"))["survival"]
    ev = [(e["time"], (e["risk"] if e["risk"] in ("R1", "R3", "R5") else "other") if not e["censored"] else None)
          for e in d["events"]]
    return ev, float(json.load(open(ROOT / "results" / f"l0_demo_{scenario}_{model}.json"))["t_max"])


def aalen_johansen_at(events, t_max):
    """Cause-specific cumulative incidence at t_max for each cause."""
    ev = sorted(events, key=lambda e: e[0])
    n = len(ev); at_risk = n; S = 1.0
    cif = {c: 0.0 for c in CAUSES}
    times = sorted({t for t, c in ev if c is not None and t <= t_max})
    for t in times:
        d_by = {c: sum(1 for tt, cc in ev if tt == t and cc == c) for c in CAUSES}
        d = sum(d_by.values())
        for c in CAUSES:
            cif[c] += S * d_by[c] / at_risk
        S *= 1 - d / at_risk
        at_risk -= sum(1 for tt, cc in ev if tt == t)   # events and censorings at t leave the risk set
    return cif


def bootstrap_cif(events, t_max, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(events)
    draws = {c: [] for c in CAUSES}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cif = aalen_johansen_at([events[i] for i in idx], t_max)
        for c in CAUSES:
            draws[c].append(cif[c])
    point = aalen_johansen_at(events, t_max)
    return {c: dict(point=point[c], lo=float(np.percentile(draws[c], 2.5)), hi=float(np.percentile(draws[c], 97.5))) for c in CAUSES}


def cause_counts(events):
    return np.array([sum(1 for _, c in events if c == k) for k in CAUSES], dtype=float)


def chi2_stat(table):
    table = np.asarray(table, dtype=float)
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    exp = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    return float(((table - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())


def permutation_p(cells, n_perm=20000, seed=0):
    """cells: list of event lists (one per model, same domain). Permute
    cause labels across models; p = P(chi2_perm >= chi2_obs)."""
    rng = np.random.default_rng(seed)
    labels = [c for ev in cells for _, c in ev]
    sizes = [len(ev) for ev in cells]
    obs = chi2_stat([cause_counts(ev) for ev in cells])
    labels = np.array(labels, dtype=object)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(labels)
        parts, k = [], 0
        for s in sizes:
            parts.append([(0.0, l) for l in labels[k:k + s]]); k += s
        if chi2_stat([cause_counts(p) for p in parts]) >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def multinomial_fit(X, y, n_classes):
    """Multinomial logit by L-BFGS on the negative log-likelihood (class 0
    reference). Returns (max log-likelihood, coefficient matrix)."""
    from scipy.optimize import minimize
    n, p = X.shape
    def nll(w):
        W = w.reshape(p, n_classes - 1)
        z = np.concatenate([np.zeros((n, 1)), X @ W], axis=1)
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(n), y].sum() + 1e-4 * (w ** 2).sum()   # tiny ridge: separable cells otherwise diverge
    w0 = np.zeros(p * (n_classes - 1))
    res = minimize(nll, w0, method="L-BFGS-B")
    return -res.fun, res.x.reshape(p, n_classes - 1)


def lrt(models_domains_events, causes_present):
    """cause ~ model + domain + model:domain. Likelihood-ratio tests via
    nested fits; chi-square p-values from scipy."""
    from scipy.stats import chi2
    cells = models_domains_events
    models = sorted({m for m, _ in cells}); domains = sorted({d for _, d in cells})
    rows, y = [], []
    cidx = {c: i for i, c in enumerate(causes_present)}
    for (m, d), ev in cells.items():
        for _, c in ev:
            if c is None: continue
            rows.append((models.index(m), domains.index(d))); y.append(cidx[c])
    rows = np.array(rows); y = np.array(y)
    def design(terms):
        cols = [np.ones(len(rows))]
        if "model" in terms:
            cols += [(rows[:, 0] == i).astype(float) for i in range(1, len(models))]
        if "domain" in terms:
            cols += [(rows[:, 1] == j).astype(float) for j in range(1, len(domains))]
        if "inter" in terms:
            cols += [((rows[:, 0] == i) & (rows[:, 1] == j)).astype(float) for i in range(1, len(models)) for j in range(1, len(domains))]
        return np.stack(cols, axis=1)
    K = len(causes_present)
    fits = {t: multinomial_fit(design(t), y, K)[0] for t in [(), ("model",), ("domain",), ("model", "domain"), ("model", "domain", "inter")]}
    def test(full, reduced):
        stat = 2 * (fits[full] - fits[reduced]); df = (design(full).shape[1] - design(reduced).shape[1]) * (K - 1)
        return dict(lr_stat=float(stat), df=int(df), p=float(chi2.sf(stat, df)))
    return dict(
        model_given_domain=test(("model", "domain"), ("domain",)),
        domain_given_model=test(("model", "domain"), ("model",)),
        interaction=test(("model", "domain", "inter"), ("model", "domain")),
        n=int(len(y)), causes=causes_present, models=models, domains=domains)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cosmos3nano", "cosmos3super"])
    ap.add_argument("--domains", nargs="+", default=["occlusion_reemergence_domA", "occlusion_reemergence_domB"])
    ap.add_argument("--extra", nargs="*", default=["occlusion_reemergence:cosmos3super"],
                    help="scenario:model cells reported (CIF + CI) but outside the matched design")
    args = ap.parse_args()
    out = dict(cells={}, within_domain={}, lrt=None, contrast={})
    cells = {}
    print("CAUSE-SPECIFIC CUMULATIVE INCIDENCE at t_max (Aalen-Johansen), bootstrap 95% CI over episodes")
    print(f"  {'cell':46s} {'n':>3}  " + "  ".join(f"{c:>18s}" for c in CAUSES))
    for d in args.domains:
        for m in args.models:
            ev, t_max = load(d, m); cells[(m, d)] = ev
            b = bootstrap_cif(ev, t_max); out["cells"][f"{d}:{m}"] = dict(n=len(ev), t_max=t_max, cif=b)
            print(f"  {d + ':' + m:46s} {len(ev):>3}  " + "  ".join(f"{b[c]['point']:.2f} [{b[c]['lo']:.2f},{b[c]['hi']:.2f}]" for c in CAUSES))
    for x in args.extra:
        s, m = x.split(":"); ev, t_max = load(s, m); b = bootstrap_cif(ev, t_max)
        out["cells"][f"{s}:{m}"] = dict(n=len(ev), t_max=t_max, cif=b, matched=False)
        print(f"  {s + ':' + m + ' (unmatched)':46s} {len(ev):>3}  " + "  ".join(f"{b[c]['point']:.2f} [{b[c]['lo']:.2f},{b[c]['hi']:.2f}]" for c in CAUSES))
    print("\nWITHIN-DOMAIN model x channel association (chi-square, permutation p over 20000 label shuffles)")
    for d in args.domains:
        stat, p = permutation_p([cells[(m, d)] for m in args.models])
        out["within_domain"][d] = dict(chi2=stat, p_perm=p)
        print(f"  {d:36s} chi2={stat:6.1f}  p={p:.4f}")
    causes_present = [c for c in CAUSES if any(cc == c for ev in cells.values() for _, cc in ev)]
    res = lrt(cells, causes_present); out["lrt"] = res
    print("\nMULTINOMIAL LOGIT  cause ~ model + domain + model:domain  (likelihood-ratio tests)")
    for k in ("model_given_domain", "domain_given_model", "interaction"):
        r = res[k]; print(f"  {k:20s} LR={r['lr_stat']:6.1f}  df={r['df']}  p={r['p']:.2e}")
    # effect-size contrast on P(R5)
    p5 = {(m, d): out["cells"][f"{d}:{m}"]["cif"]["R5"]["point"] for (m, d) in cells}
    model_eff = np.mean([abs(p5[(args.models[0], d)] - p5[(args.models[1], d)]) for d in args.domains])
    dom_eff = np.mean([abs(p5[(m, args.domains[0])] - p5[(m, args.domains[1])]) for m in args.models])
    out["contrast"] = dict(p_R5=p5 and {f"{m}:{d}": v for (m, d), v in p5.items()},
                           model_effect_on_P_R5=float(model_eff), domain_effect_on_P_R5=float(dom_eff))
    print(f"\nEFFECT SIZES on P(terminal cause = R5): |model effect| (mean over domains) = {model_eff:.2f}   "
          f"|render-domain effect| (mean over models) = {dom_eff:.2f}   ratio = {model_eff / max(dom_eff, 1e-9):.1f}x")
    (ROOT / "results" / "termination_fingerprints.json").write_text(json.dumps(out, indent=1))
    print(f"-> results/termination_fingerprints.json")


if __name__ == "__main__":
    main()
