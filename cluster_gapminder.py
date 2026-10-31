"""Cluster bootstrap by country for the gapminder illustration (seed 606).
Usage: python3 cluster_gapminder.py DATADIR"""
import sys, os, numpy as np, pandas as pd
from kseg_boot import kseg_boot, kseg_boot_cluster

D = sys.argv[1]; rng = np.random.default_rng(606)
g = pd.read_csv(os.path.join(D, "gapminder.csv")).sort_values(["country", "year"])
for lab, x in [("GDP per capita", g.gdpPercap.values), ("log GDP per capita", np.log(g.gdpPercap.values))]:
    for K in (3, 4):
        p = kseg_boot_cluster(x, g.lifeExp.values, K, g.country.values, 999, rng)["p_boot"]
        kseg_boot(g.lifeExp.values, x, K, 999, rng)   # keeps the random stream of the original run
        print(f"{lab}, K = {K}: cluster bootstrap p = {p:.3f}")
