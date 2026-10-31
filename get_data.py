"""Download the four public datasets as CSV files. Usage: python3 get_data.py DATADIR"""
import sys, os
import statsmodels.api as sm
D = sys.argv[1] if len(sys.argv) > 1 else "."; os.makedirs(D, exist_ok=True)
for name, pkg, out in [("gapminder", "gapminder", "gapminder.csv"), ("diamonds", "ggplot2", "diamonds.csv"),
                       ("faithful", "datasets", "faithful.csv"), ("GaltonFamilies", "HistData", "galton.csv")]:
    sm.datasets.get_rdataset(name, pkg).data.to_csv(os.path.join(D, out), index=False); print("saved", out)
