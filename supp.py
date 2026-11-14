"""Supplementary run (Amendment 4; reported in Document S2): (a) null conditions with non-normal errors; (b) a combined standard test
(Breusch-Pagan or RESET cubic, Bonferroni) and the location of the departure, with all corrected values saved.
Seed 505. Usage: python3 supp.py OUTCSV [NPROC]"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from shapes import gen, calibrate
from kseg_boot import kseg_boot

B = 399; NS = (200, 500, 1000, 2000, 5000)


def errors(kind, n, rng):
    """errors with mean 0 and variance 1"""
    if kind == "skewed": return (rng.chisquare(2, n) - 2) / 2                     # skewness 2
    if kind == "heavy": return rng.standard_t(3, n) / np.sqrt(3)                   # t with 3 df
    if kind == "outliers":                                                          # 5% with SD 5
        e = rng.standard_normal(n) * np.where(rng.random(n) < .05, 5, 1); return e / np.sqrt(.95 + .05 * 25)


def run_cell(args):
    part, shape, n, reps, cell = args
    rng = np.random.default_rng([505, cell]); rows = []
    if part == "errors": beta = 0.4 / np.sqrt(1 - 0.16)
    else: beta = calibrate(shape, 0.4)
    for rep in range(reps):
        if part == "errors":
            x = rng.standard_normal(n); y = beta * x + errors(shape, n, rng); Ks = (2, 3, 4, 5, 10)
        else:
            x, y = gen(shape, beta, n, rng); Ks = (3, 4)
        X = sm.add_constant(x); fit = sm.OLS(y, X).fit()
        pbp = het_breuschpagan(fit.resid, X)[1]; pr3 = linear_reset(fit, power=3, use_f=True).pvalue
        for K in Ks:
            r = kseg_boot(x, y, K, B, rng)
            rows.append(dict(part=part, shape=shape, n=n, rep=rep, K=K, p_boot=r["p_boot"], p_chi2=r["p_chi2"],
                             p_bp=pbp, p_reset3=pr3, rc=";".join(f"{v:.4f}" for v in r["rc"])))
    return rows


if __name__ == "__main__":
    out = sys.argv[1]; nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cells = [("errors", s, n, 1000) for s in ("skewed", "heavy", "outliers") for n in NS]
    cells += [("compare", s, n, 1000 if s in "AE" else 500) for s in "ABCDE" for n in NS]
    cells = [c + (i,) for i, c in enumerate(cells)]
    cells.sort(key=lambda c: -c[2] * c[3])   # largest first
    with Pool(nproc) as pool:
        res = []
        for k, rows in enumerate(pool.imap_unordered(run_cell, cells)):
            res.extend(rows); print(f"{k + 1}/{len(cells)} cells", flush=True)
    pd.DataFrame(res).to_csv(out, index=False)
