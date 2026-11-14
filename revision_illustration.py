"""Real-data results added in revision (seed 1108): 95% intervals for segment r, ordinary and wild cluster
bootstrap p-values, the equal-slope test, and the Breusch-Pagan and RESET tests.
Usage: python3 revision_illustration.py DATADIR OUT.json"""
import sys, os, json, numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from kseg_boot import kseg_boot, segment_ci, kseg_wild_cluster, equal_slope_test

D, out = sys.argv[1:3]; rng = np.random.default_rng(1108)
g = pd.read_csv(os.path.join(D, "gapminder.csv")); di = pd.read_csv(os.path.join(D, "diamonds.csv"))
f = pd.read_csv(os.path.join(D, "faithful.csv")); ga = pd.read_csv(os.path.join(D, "galton.csv")); ga = ga[ga.gender == "male"]
sets = [("Gapminder, GDP", g.gdpPercap.values, g.lifeExp.values, g.country.values),
        ("Gapminder, log GDP", np.log(g.gdpPercap.values), g.lifeExp.values, g.country.values),
        ("Diamonds", di.carat.values, di.price.values.astype(float), None),
        ("Diamonds, log scales", np.log(di.carat.values), np.log(di.price.values.astype(float)), None),
        ("Old Faithful", f.eruptions.values, f.waiting.values.astype(float), None),
        ("Galton heights", ga.midparentHeight.values, ga.childHeight.values.astype(float), ga.family.astype(str).values)]
res = []
for lab, x, y, cl in sets:
    X = sm.add_constant(x); fit = sm.OLS(y, X).fit()
    for K in (3, 4):
        r = kseg_boot(x, y, K, 999, rng); ci = segment_ci(x, y, K, 999, rng=rng)
        row = dict(label=lab, K=K, n=len(x), r_all=float(np.corrcoef(x, y)[0, 1]), rc=r["rc"].tolist(),
                   range=float(r["range_rc"]), lo=ci["lo"].tolist(), hi=ci["hi"].tolist(), p_boot=float(r["p_boot"]),
                   p_es=float(equal_slope_test(x, y, K)), p_bp=float(het_breuschpagan(fit.resid, X)[1]),
                   p_reset=float(linear_reset(fit, power=3, use_f=True).pvalue))
        if cl is not None: row["p_wild"] = float(kseg_wild_cluster(x, y, K, cl, 999, rng)["p_boot"])
        res.append(row)
        print(lab, K, np.round(r["rc"], 2), np.round(ci["lo"], 2), np.round(ci["hi"], 2), "p", row["p_boot"],
              "wild", row.get("p_wild"), "es", round(row["p_es"], 3), "reset", round(row["p_reset"], 3))
json.dump(res, open(out, "w"), indent=1)
