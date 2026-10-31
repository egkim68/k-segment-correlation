"""Figures (PNG) for the K-segment correlation paper, separate from the simulation code.
Usage: python3 make_figures.py OUTDIR DATADIR FIGDIR [DPI]"""
import sys, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

O, D, F = sys.argv[1:4]; DPI = int(sys.argv[4]) if len(sys.argv) > 4 else 600
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C = {"boot": "#1f4e79", "chi2": "#c0392b", "unc": "#7f7f7f", "bp": "#e08e0b", "reset3": "#4a7c59"}
Ks = [2, 3, 4, 5, 10]

# Figure 5: gapminder, raw and log GDP, K = 3 segments with corrected correlations
g = pd.read_csv(os.path.join(D, "gapminder.csv")); ill = json.load(open(os.path.join(O, "illustration.json")))
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8))
for a, (xv, res, lab, logax) in zip(ax, [(g.gdpPercap, ill[0], "GDP per capita (US$, log axis)", True), (np.log(g.gdpPercap), ill[2], "Log GDP per capita", False)]):
    a.scatter(xv, g.lifeExp, s=2, color="#9db4c0", alpha=.6, linewidths=0)
    e = res["edges"]
    for b in e[1:-1]: a.axvline(b, color="#555555", lw=.6, ls="--")
    for k in range(len(e) - 1):
        mid = np.sqrt(e[k] * e[k + 1]) if logax else (e[k] + e[k + 1]) / 2
        a.text(mid, 88, f"{res['rc'][k]:.2f}", ha="center", fontsize=7)
    if logax: a.set_xscale("log")
    a.set_ylim(20, 92); a.set_xlabel(lab); a.set_ylabel("Life expectancy (years)")
    a.set_title(f"r = {res['r_all']:.2f}, bootstrap p = {res['p_boot']:.3f}", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_5.png"), dpi=DPI); plt.close(fig)

# Figure 2: bias of uncorrected and corrected segment correlations (pattern A)
b = pd.read_csv(os.path.join(O, "bias.csv"))
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for n, ls in [(1000, "-"), (5000, "--")]:
    x = b[b.n == n]
    ax.plot(x.K, x.bias_r, marker="o", color=C["unc"], ls=ls, lw=1.2, label=f"Uncorrected, N = {n:,}")
    ax.plot(x.K, x.bias_rc, marker="s", color=C["boot"], ls=ls, lw=1.2, label=f"Corrected, N = {n:,}")
ax.axhline(0, color="black", lw=.6); ax.set_xticks(Ks); ax.set_xlabel("Number of segments K")
ax.set_ylabel("Mean segment r minus true correlation"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_2.png"), dpi=DPI); plt.close(fig)

# Figure 3: Type I error rates, bootstrap and chi-square references (patterns A and E, N = 1,000)
r = pd.read_csv(os.path.join(O, "rates.csv"))
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
for a, p, t in zip(ax, "AE", ["(a) Normal X", "(b) Skewed X"]):
    x = r[(r.pattern == p) & (r.n == 1000)].groupby("K")[["rb", "rc2"]].mean()
    a.plot(x.index, x.rb, marker="s", color=C["boot"], lw=1.2, label="Bootstrap reference")
    a.plot(x.index, x.rc2, marker="o", color=C["chi2"], lw=1.2, label="Chi-square reference")
    a.axhspan(.03, .07, color="#dddddd", alpha=.6, lw=0); a.axhline(.05, color="black", lw=.6)
    a.set_xticks(Ks); a.set_xlabel("Number of segments K"); a.set_title(t, loc="left", fontsize=8)
ax[0].set_ylabel("Type I error rate"); ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_3.png"), dpi=DPI); plt.close(fig)

# Figure 4: power by K at N = 1,000, rho = 0.4, with standard tests as reference lines
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
for a, p, t in zip(ax, "BCD", ["(a) Growing scatter", "(b) S-shaped curve", "(c) Top quarter only"]):
    x = r[(r.pattern == p) & (r.n == 1000) & (r.rho == .4)].set_index("K")
    a.plot(x.index, x.rb, marker="s", color=C["boot"], lw=1.3, label="K-segment test")
    a.axhline(x.r_bp.mean(), color=C["bp"], ls="--", lw=1, label="Breusch–Pagan")
    a.axhline(x.r_reset3.mean(), color=C["reset3"], ls=":", lw=1.2, label="RESET (cubic)")
    a.set_ylim(0, 1.03); a.set_xticks(Ks); a.set_xlabel("Number of segments K"); a.set_title(t, loc="left", fontsize=8)
ax[0].set_ylabel("Power"); ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_4.png"), dpi=DPI); plt.close(fig)
print("figures saved")
