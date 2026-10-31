"""Amendment 3: the confirmatory design run at N = 2,000 (seeds [303, i * 10 + 4], unused in the main run).
Usage: python3 confirm_n2000.py OUTCSV [NPROC]"""
import sys, pandas as pd
from multiprocessing import Pool
from confirm import DESIGN, run_cell

if __name__ == "__main__":
    out = sys.argv[1]; nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cells = [(p, r, reps, 2000, i * 10 + 4) for i, (p, r, reps) in enumerate(DESIGN)]
    with Pool(nproc) as pool:
        res = []
        for k, rows in enumerate(pool.imap_unordered(run_cell, cells)):
            res.extend(rows); print(f"{k + 1}/{len(cells)} cells", flush=True)
    pd.DataFrame(res).to_csv(out, index=False)
