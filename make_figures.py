"""Figures (PNG) for the K-segment correlation paper, separate from the simulation code.
Usage: python3 make_figures.py OUTDIR DATADIR FIGDIR [DPI]"""
import sys, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

O, D, F = sys.argv[1:4]; DPI = int(sys.argv[4]) if len(sys.argv) > 4 else 600
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C = {"boot": "#1f4e79", "chi2": "#c0392b", "unc": "#7f7f7f", "bp": "#e08e0b", "reset3": "#4a7c59"}
Ks = [2, 3, 4, 5, 10]

# Figure 5: two reasons a segment correlation falls (gapminder: slope flattens; diamonds: scatter grows), K = 3
g = pd.read_csv(os.path.join(D, "gapminder.csv")); di = pd.read_csv(os.path.join(D, "diamonds.csv"))
ill = json.load(open(os.path.join(O, "illustration.json")))
plt.rcParams["mathtext.fontset"] = "dejavuserif"
fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.0))
panels = [(g.gdpPercap, g.lifeExp, ill[0], "GDP per capita (US$, log axis)", "Life expectancy (years)", "(a) Gapminder:", 2, .6, (20, 93), 88),
          (di.carat, di.price, ill[4], "Weight (carat, log axis)", "Price (US$)", "(b) Diamonds:", .4, .15, (0, 20500), 19300)]
for a, (xv, yv, res, xl, yl, t, sz, al, yl_, ytxt) in zip(ax, panels):
    a.scatter(xv, yv, s=sz, color="#9db4c0", alpha=al, linewidths=0, rasterized=True)
    e = res["edges"]
    for b in e[1:-1]: a.axvline(b, color="#555555", lw=.6, ls="--")
    for k in range(len(e) - 1):
        a.text(np.sqrt(e[k] * e[k + 1]), ytxt, f"$r$ = {res['rc'][k]:.2f}", ha="center", fontsize=7)
    a.set_xscale("log"); a.set_ylim(*yl_); a.set_xlabel(xl); a.set_ylabel(yl)
    ptxt = "$p$ \u2264 0.001" if res["p_boot"] <= 0.001 else f"$p$ = {res['p_boot']:.3f}"
    a.set_title(f"{t} $r$ = {res['r_all']:.2f}, bootstrap {ptxt}", loc="left", fontsize=8)
ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10000 else f"{v:.0f}"))
ax[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax[1].set_xticks([0.25, 0.5, 1, 2, 4])
ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_5.png"), dpi=DPI); plt.close(fig)

# Figure 2: bias of uncorrected and corrected segment correlations (pattern A)
# N = 5000 is drawn first with open markers, N = 1000 on top with filled markers, so both series stay visible
b = pd.read_csv(os.path.join(O, "bias.csv"))
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for n, ls, filled in [(5000, "--", False), (1000, "-", True)]:
    x = b[b.n == n]; ms = 4 if filled else 8
    for col, mk, key, lab in [("unc", "o", "bias_r", "Uncorrected"), ("boot", "s", "bias_rc", "Corrected")]:
        ax.plot(x.K, x[key], marker=mk, color=C[col], ls=ls, lw=1.2, ms=ms, markerfacecolor=C[col] if filled else "white",
                markeredgewidth=1.0, label=f"{lab}, $N$ = {n}", zorder=3 if filled else 2)
h, l = ax.get_legend_handles_labels()
order = [2, 0, 3, 1]  # N = 1000 entries first
ax.axhline(0, color="black", lw=.6); ax.set_xticks(Ks); ax.set_xlabel("Number of segments $K$")
ax.set_ylabel("Mean segment $r$ minus true correlation")
ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_2.png"), dpi=DPI); plt.close(fig)

# Figure 3: Type I error rates, bootstrap and chi-square references (patterns A and E, N = 1000)
r = pd.read_csv(os.path.join(O, "rates.csv"))
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
for a, p, t in zip(ax, "AE", ["(a) Normal X", "(b) Skewed X"]):
    x = r[(r.pattern == p) & (r.n == 1000)].groupby("K")[["rb", "rc2"]].mean()
    a.plot(x.index, x.rb, marker="s", color=C["boot"], lw=1.2, label="Bootstrap reference")
    a.plot(x.index, x.rc2, marker="o", color=C["chi2"], lw=1.2, label="Chi-square reference")
    a.axhspan(.03, .07, color="#dddddd", alpha=.6, lw=0); a.axhline(.05, color="black", lw=.6)
    a.set_xticks(Ks); a.set_xlabel("Number of segments $K$"); a.set_title(t, loc="left", fontsize=8)
ax[0].set_ylabel("Type I error rate"); ax[0].set_yticks([i / 10 for i in range(8)])
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))  # one legend for both panels
fig.tight_layout(rect=[0, 0.09, 1, 1]); fig.savefig(os.path.join(F, "Figure_3.png"), dpi=DPI); plt.close(fig)

# Figure 4: power by K at N = 1000, rho = 0.4, with standard tests as reference lines
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
for a, p, t in zip(ax, "BCD", ["(a) Growing scatter", "(b) S-shaped curve", "(c) Top quarter only"]):
    x = r[(r.pattern == p) & (r.n == 1000) & (r.rho == .4)].set_index("K")
    a.plot(x.index, x.rb, marker="s", ms=4.5, color=C["boot"], lw=1.3, label="K-segment test", zorder=2)
    a.axhline(x.r_bp.mean(), color=C["bp"], ls="--", lw=1.1, label="Breusch\u2013Pagan", zorder=3)
    a.axhline(x.r_reset3.mean(), color=C["reset3"], ls=":", lw=1.4, label="RESET", zorder=4)
    a.set_ylim(0, 1.03); a.set_xticks(Ks); a.set_xlabel("Number of segments $K$"); a.set_title(t, loc="left", fontsize=8)
ax[0].set_ylabel("Power")
h, l = ax[0].get_legend_handles_labels(); l[0] = "$K$-segment test"
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))  # one legend for all panels
fig.tight_layout(rect=[0, 0.09, 1, 1]); fig.savefig(os.path.join(F, "Figure_4.png"), dpi=DPI); plt.close(fig)
print("figures saved")
