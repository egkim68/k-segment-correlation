"""Simulations added in revision (seed 1107). Usage: python3 revision_sim.py OUTDIR [CORES]
bias.csv         mean and median bias of corrected segment r, and share of samples with a range above 0.10 (Shapes A, E)
ci.csv           coverage and width of 95% percentile intervals (Shapes A, E; rho = 0.4)
equal_slope.csv  rejection rates of the equal-slope F test with K = 3 (Shapes A to E)"""
import sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
from shapes import gen, calibrate
from kseg_boot import kseg_boot, segment_ci, equal_slope_test

def rc_only(x, y, K):
    return kseg_boot(x, y, K, 1, np.random.default_rng(0))["rc"]

def bias_cell(a):
    shape, rho, n, K, reps, cell = a; rng = np.random.default_rng([1107, 1, cell]); b = calibrate(shape, rho)
    r = np.array([rc_only(*gen(shape, b, n, rng), K) for _ in range(reps)]); rg = r.max(1) - r.min(1)
    return [dict(shape=shape, rho=rho, n=n, K=K, seg=k + 1, mean_bias=r[:, k].mean() - rho,
                 median_bias=np.median(r[:, k]) - rho, range_over_010=(rg > 0.10).mean()) for k in range(K)]

def ci_cell(a):
    shape, rho, n, K, reps, cell = a; rng = np.random.default_rng([1107, 2, cell]); b = calibrate(shape, rho); out = []
    for _ in range(reps):
        c = segment_ci(*gen(shape, b, n, rng), K, 399, rng=rng)
        out += [dict(shape=shape, rho=rho, n=n, K=K, seg=k + 1, coverage=c["lo"][k] <= rho <= c["hi"][k],
                     width=c["hi"][k] - c["lo"][k]) for k in range(K)]
    return pd.DataFrame(out).groupby(["shape", "rho", "n", "K", "seg"]).mean().reset_index().to_dict("records")

def es_cell(a):
    shape, rho, n, reps, cell = a; rng = np.random.default_rng([1107, 3, cell]); b = calibrate(shape, rho)
    rej = [equal_slope_test(*gen(shape, b, n, rng), 3) < .05 for _ in range(reps)]
    return [dict(shape=shape, rho=rho, n=n, K=3, rate=np.mean(rej), reps=reps)]

if __name__ == "__main__":
    O = sys.argv[1]; P = int(sys.argv[2]) if len(sys.argv) > 2 else 2; os.makedirs(O, exist_ok=True)
    jobs = [(bias_cell, [(s, r, n, K, 2000) for s in "AE" for r in (.2, .4, .6) for n in (200, 500, 1000, 2000, 5000) for K in (3, 4)], "bias.csv"),
            (ci_cell, [(s, .4, n, K, 300) for s in "AE" for n in (500, 1000, 5000) for K in (3, 4)], "ci.csv"),
            (es_cell, [(s, r, n, 1000 if s in "AE" else 500) for s in "ABCDE" for r in (.2, .4, .6) for n in (200, 500, 1000, 2000, 5000)], "equal_slope.csv")]
    with Pool(P) as pool:
        for fn, cells, name in jobs:
            cells = [c + (i,) for i, c in enumerate(cells)]
            pd.DataFrame(sum(pool.map(fn, cells), [])).to_csv(os.path.join(O, name), index=False); print("done", name, flush=True)
