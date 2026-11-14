"""Summaries of the supplementary run (Amendment 4; Document S2). Usage: python3 analysis_supp.py supp.csv OUTDIR"""
import sys, os, json, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); O = sys.argv[2]
d["rb"] = d.p_boot < .05; d["rc2"] = d.p_chi2 < .05; d["comb"] = (d.p_bp < .025) | (d.p_reset3 < .025)
e = d[d.part == "errors"]
err = e.groupby(["shape", "K", "n"])[["rb", "rc2"]].mean().reset_index()
err.to_csv(os.path.join(O, "supp_errors.csv"), index=False)
c = d[d.part == "compare"].copy()
rc = c.rc.str.split(";").apply(lambda v: np.array(v, float))
def loc(row, v):
    K = len(v)
    if row.shape == "B": return int(np.argmax(v) == 0 and np.argmin(v) == K - 1)
    if row.shape == "C": return int(v[1:-1].min() > max(v[0], v[-1]))
    if row.shape == "D": return int(np.argmax(v) == K - 1)
    return np.nan
c["loc"] = [loc(r, v) for r, v in zip(c.itertuples(), rc)]
comp = c.groupby(["shape", "K", "n"]).agg(rb=("rb", "mean"), comb=("comb", "mean")).reset_index()
locs = c[c.rb].groupby(["shape", "K", "n"])["loc"].agg(["mean", "count"]).reset_index()
comp.to_csv(os.path.join(O, "supp_compare.csv"), index=False); locs.to_csv(os.path.join(O, "supp_location.csv"), index=False)
print(err.groupby(["shape"]).rb.agg(["min", "max"]))
print(err[err.n == 1000].pivot_table(index="shape", columns="K", values=["rb", "rc2"]).round(3))
print(err.groupby("shape").rc2.max())
print(comp.pivot_table(index=["shape", "K"], columns="n", values=["rb", "comb"]).round(2))
print(locs[locs["shape"].isin(list("BCD"))].pivot_table(index=["shape", "K"], columns="n", values="mean").round(2))
