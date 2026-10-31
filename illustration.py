"""Real-data illustrations (gapminder, diamonds, faithful, Galton) with the tie rule. Usage: python3 illustration.py DATADIR OUTDIR"""
import sys, os, json, numpy as np, pandas as pd
from kseg_boot import kseg_boot

D, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(404)


def tie_ok(x, K, share=0.05, min_count=10):
    """True if no internal quantile boundary is shared by more than max(share * n / K, min_count) cases (Amendment 1)."""
    edges = np.quantile(x, np.linspace(0, 1, K + 1))[1:-1]
    return all((x == b).sum() <= max(share * len(x) / K, min_count) for b in edges)


def run(label, x, y, K):
    r = kseg_boot(x, y, K, 999, rng)
    edges = np.quantile(x, np.linspace(0, 1, K + 1))
    return dict(label=label, n=len(x), K=K, ties_flag=bool(not tie_ok(x, K)),
                r_all=float(np.corrcoef(x, y)[0, 1]), rc=[round(float(v), 3) for v in r["rc"]],
                range=round(float(r["range_rc"]), 3), p_boot=float(r["p_boot"]),
                edges=[round(float(v), 3) for v in edges])


g = pd.read_csv(os.path.join(D, "gapminder.csv")); di = pd.read_csv(os.path.join(D, "diamonds.csv")); f = pd.read_csv(os.path.join(D, "faithful.csv"))
ga = pd.read_csv(os.path.join(D, "galton.csv")); ga = ga[ga.gender == "male"]  # sons only (Amendment 2)
data = [("gapminder: GDP per capita vs life expectancy", g.gdpPercap.values, g.lifeExp.values),
        ("gapminder: log GDP per capita vs life expectancy", np.log(g.gdpPercap.values), g.lifeExp.values),
        ("diamonds: carat vs price", di.carat.values, di.price.values.astype(float)),
        ("faithful: eruption length vs waiting time", f.eruptions.values, f.waiting.values.astype(float)),
        ("galton: midparent height vs son height", ga.midparentHeight.values, ga.childHeight.values.astype(float))]
res = [run(lab, x, y, K) for lab, x, y in data for K in (3, 4)]
# check of the tie rule with fathers' heights (recorded mostly in whole or half inches); reported in the text only
res += [run("galton: father height vs son height", ga.father.values, ga.childHeight.values.astype(float), K) for K in (3, 4)]
json.dump(res, open(os.path.join(O, "illustration.json"), "w"), indent=1)
for r in res: print(r["label"], "K", r["K"], "ties", r["ties_flag"], "r", round(r["r_all"], 3), r["rc"], "range", r["range"], "p", r["p_boot"])
