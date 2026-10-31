"""Confirmatory run (protocol v1.0, seed 303). Usage: python3 confirm.py OUTCSV [NPROC]"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from shapes import gen, calibrate
from kseg_boot import kseg_boot, _segments

B = 399
DESIGN = [(p, r, 1000) for p in "AE" for r in (0.2, 0.4, 0.6)] + [(p, r, 500) for p in "BCD" for r in (0.2, 0.4, 0.6)]


def run_cell(args):
    pattern, rho, reps, n, cell = args
    rng = np.random.default_rng([303, cell]); beta = calibrate(pattern, rho); rows = []
    for rep in range(reps):
        x, y = gen(pattern, beta, n, rng)
        X = sm.add_constant(x); fit = sm.OLS(y, X).fit()
        base = dict(pattern=pattern, rho=rho, n=n, rep=rep, p_bp=het_breuschpagan(fit.resid, X)[1],
                    p_reset2=linear_reset(fit, power=2, use_f=True).pvalue,
                    p_reset3=linear_reset(fit, power=3, use_f=True).pvalue)
        for K in (2, 3, 4, 5, 10):
            r = kseg_boot(x, y, K, B, rng); seg = _segments(x, K)
            rr = [np.corrcoef(x[seg == k], y[seg == k])[0, 1] for k in range(K)]
            rows.append(dict(base, K=K, p_boot=r["p_boot"], p_chi2=r["p_chi2"], range_rc=r["range_rc"],
                             mean_rc=float(np.mean(r["rc"])), mean_r=float(np.mean(rr)),
                             top_is_max=int(np.argmax(r["rc"]) == K - 1)))
    return rows


if __name__ == "__main__":
    out = sys.argv[1]; nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cells = [(p, r, reps, n, i * 10 + j) for i, (p, r, reps) in enumerate(DESIGN) for j, n in enumerate((200, 500, 1000, 5000))]
    with Pool(nproc) as pool:
        res = []
        for k, rows in enumerate(pool.imap_unordered(run_cell, cells)):
            res.extend(rows); print(f"{k + 1}/{len(cells)} cells", flush=True)
    pd.DataFrame(res).to_csv(out, index=False)
