"""Analysis of the confirmatory run: hypotheses H1-H6 and summary tables.
Usage: python3 analysis.py confirm.csv OUTDIR [EXTRA.csv]
Hypotheses use the main run only; the descriptive tables also include EXTRA (Amendment 3, N = 2,000)."""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv(sys.argv[1]); O = sys.argv[2]; os.makedirs(O, exist_ok=True)
d["rb"] = d.p_boot < .05; d["rc2"] = d.p_chi2 < .05
for t in ("bp", "reset2", "reset3"): d["r_" + t] = d["p_" + t] < .05
H = {}

A = d[d.pattern == "A"]
x = A[A.K >= 3]; g = x.groupby(["K", "n", "rho"]).mean_r.mean(); cell = g / g.index.get_level_values("rho")
t = stats.ttest_1samp(cell, 0.85, alternative="less")
H["H1"] = dict(criterion="Uncorrected segment r / rho, pattern A, K >= 3", type="Mean ratio over conditions",
               threshold="<= 0.85", estimate=float(cell.mean()), max=float(cell.max()), p=float(t.pvalue), conditions=len(cell))

x = A[(A.K <= 4) & (A.n / A.K >= 250)]
bias = (x.mean_rc - x.rho).groupby([x.K, x.n, x.rho]).mean()
H["H2"] = dict(criterion="Bias of corrected segment r, pattern A, K <= 4, n/K >= 250", type="Largest absolute bias",
               threshold="<= 0.03", estimate=float(bias.abs().max()), mean=float(bias.mean()), conditions=len(bias),
               p=float(stats.ttest_1samp(bias.abs(), 0.03, alternative="less").pvalue))

ae = d[d.pattern.isin(["A", "E"])]
te = ae.groupby(["pattern", "K", "n", "rho"]).rb.agg(["mean", "sum", "count"])
pmax = max(stats.binomtest(int(r["sum"]), int(r["count"]), .07, alternative="less").pvalue for _, r in te.iterrows())
pmin = max(stats.binomtest(int(r["sum"]), int(r["count"]), .03, alternative="greater").pvalue for _, r in te.iterrows())
H["H3"] = dict(criterion="Bootstrap Type I error, patterns A and E, every condition", type="Range of rates",
               threshold="0.03 to 0.07", estimate=[float(te["mean"].min()), float(te["mean"].max())],
               conditions=len(te), p=float(max(pmax, pmin)))

e = d[d.pattern == "E"].groupby(["K", "n", "rho"]).rc2.agg(["mean", "sum", "count"]); top = e["mean"].idxmax()
H["H4"] = dict(criterion="Chi-square reference Type I error, pattern E, largest condition", type="Largest rate",
               threshold="> 0.10", estimate=float(e["mean"].max()), at=[int(top[0]), int(top[1]), float(top[2])],
               p=float(stats.binomtest(int(e.loc[top, "sum"]), int(e.loc[top, "count"]), .10, alternative="greater").pvalue))

for p in "BCD":
    x = d[(d.pattern == p) & (d.K == 3) & (d.n == 1000) & (d.rho == 0.4)]
    H["H5" + p] = dict(criterion=f"Bootstrap power, pattern {p}, K = 3, n = 1,000, rho = 0.4", type="Share",
                       threshold=">= 0.80", estimate=float(x.rb.mean()),
                       p=float(stats.binomtest(int(x.rb.sum()), len(x), .80, alternative="greater").pvalue))
x = d[(d.pattern == "D") & (d.K == 4) & (d.n == 1000) & (d.rho == 0.4) & d.rb]
H["H6"] = dict(criterion="Top segment highest among rejections, pattern D, K = 4, n = 1,000, rho = 0.4", type="Share",
               threshold=">= 0.90", estimate=float(x.top_is_max.mean()), rejections=len(x),
               p=float(stats.binomtest(int(x.top_is_max.sum()), len(x), .90, alternative="greater").pvalue))
for k, h in H.items():
    est = h["estimate"]
    if k == "H1": ok = est <= .85
    elif k == "H2": ok = est <= .03
    elif k == "H3": ok = est[0] >= .03 and est[1] <= .07
    elif k == "H4": ok = est > .10
    else: ok = est >= (.90 if k == "H6" else .80)
    h["decision"] = ("Supported" if h["p"] < .05 else "Threshold met, not significant") if ok else "Not supported"
json.dump(H, open(os.path.join(O, "hypotheses.json"), "w"), indent=1)

if len(sys.argv) > 3:
    e = pd.read_csv(sys.argv[3]); e["rb"] = e.p_boot < .05; e["rc2"] = e.p_chi2 < .05
    for t in ("bp", "reset2", "reset3"): e["r_" + t] = e["p_" + t] < .05
    d = pd.concat([d, e], ignore_index=True); A = d[d.pattern == "A"]

d.groupby(["pattern", "K", "n", "rho"])[["rb", "rc2", "r_bp", "r_reset2", "r_reset3"]].mean().reset_index().to_csv(os.path.join(O, "rates.csv"), index=False)
A.assign(bias_r=A.mean_r - A.rho, bias_rc=A.mean_rc - A.rho).groupby(["K", "n"])[["bias_r", "bias_rc"]].mean().reset_index().to_csv(os.path.join(O, "bias.csv"), index=False)
x = d[d.pattern.isin(list("BCD")) & (d.rho == .4)]
pw = x.groupby(["pattern", "K", "n"]).rb.mean().reset_index(); pw["per_seg"] = pw.n / pw.K
pw[pw.rb >= .8].groupby("pattern").per_seg.min().to_csv(os.path.join(O, "min_per_segment.csv"))
for k, h in H.items(): print(k, h["estimate"], h["decision"], round(h["p"], 4))
