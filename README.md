# K-segment correlation

Code and protocol for "Is One Correlation Enough? K-Segment Correlations for Relationships That Change Across the Data".

The method splits the cases into K segments at the quantiles of X, corrects each segment's correlation for range restriction (Thorndike's Case II), and tests whether the corrected values differ with a fixed-X residual bootstrap. All functions are in `kseg_boot.py`.

## Use

```python
from kseg_boot import kseg_boot, segment_ci, kseg_wild_cluster, equal_slope_test
res = kseg_boot(x, y, K=3, B=999)            # corrected segment correlations, Q, bootstrap p-value
ci  = segment_ci(x, y, K=3, B=999)           # 95% intervals for each segment correlation
wc  = kseg_wild_cluster(x, y, 3, clusters)   # p-value for linked cases (e.g., countries, families)
p   = equal_slope_test(x, y, K=3)            # F test of equal slopes across the segments
```


A step-by-step guide to calculating segment correlations, with a worked example, is in [HOW_TO.md](HOW_TO.md).

## Reproduce

Python 3.13 with NumPy, SciPy, pandas, statsmodels, and matplotlib.

```
python3 get_data.py .
python3 confirm.py confirm.csv            # main run (seed 303)
python3 confirm_n2000.py confirm_n2000.csv
mkdir -p out
python3 analysis.py confirm.csv out confirm_n2000.csv
python3 illustration.py . out
python3 make_figures.py out . figures 600
python3 supp.py supp.csv                   # Document S2 (seed 505)
python3 analysis_supp.py supp.csv out
python3 revision_sim.py rev                # added in revision: bias, interval coverage, equal-slope test (seed 1107)
python3 revision_illustration.py . rev/illustration_rev.json   # intervals, wild cluster bootstrap (seed 1108)
python3 ydirection.py rev/ydirection.csv  # segments formed on Y, Document S2 (seed 1110)
```

`PROTOCOL.md` gives the design, hypotheses, seeds, and amendments.

## License

MIT

