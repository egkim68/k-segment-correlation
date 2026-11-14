"""Rejection rates of the K-segment test with segments formed on Y instead of X (seed 1110; Document S2).
Usage: python3 ydirection.py OUT.csv [CORES]"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from shapes import gen, calibrate
from kseg_boot import kseg_boot

def cell(a):
    shape, n, reps, i = a; rng = np.random.default_rng([1110, i]); b = calibrate(shape, 0.4)
    rej = []
    for _ in range(reps):
        x, y = gen(shape, b, n, rng); rej.append(kseg_boot(y, x, 3, 199, rng)["p_boot"] < .05)   # segments of Y
    return dict(shape=shape, n=n, K=3, rho=0.4, rate_y=np.mean(rej), reps=reps)

if __name__ == "__main__":
    cells = [(s, n, 500) for s in "ABCDE" for n in (200, 500, 1000, 2000, 5000)]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as p:
        res = p.map(cell, [c + (i,) for i, c in enumerate(cells)])
    pd.DataFrame(res).to_csv(sys.argv[1], index=False); print("done")
