# K-segment correlation

Code and protocol for "Is One Correlation Enough? K-Segment Correlations for Relationships That Change Across the Data".

The method splits the cases into K segments at the quantiles of X, corrects each segment's correlation for range restriction (Thorndike's Case II), and tests whether the corrected values differ with a fixed-X residual bootstrap.

## Quick start

Needs Python with NumPy and SciPy. The functions are in `kseg_boot.py`.

**Inputs**
- `x`: the variable used to form the segments, one number per case.
- `y`: the other variable, for the same cases in the same order. The two must have the same length and no missing values.
- `K`: the number of segments (3 by default; choose it before looking at the results).
- `B`: the number of bootstrap resamples (999 by default).

```python
from kseg_boot import kseg_boot, segment_ci
res = kseg_boot(x, y, K=3, B=999)
ci  = segment_ci(x, y, K=3, B=999)
```

**Outputs**
- `res["rc"]`: the corrected correlation in each segment, from the lowest to the highest values of `x`.
- `res["p_boot"]`: the bootstrap p-value. A small value means one correlation does not describe the whole range.
- `res["Q"]`: the heterogeneity statistic behind the p-value. `res["range_rc"]` is the gap between the highest and lowest corrected correlation.
- `ci["lo"]` and `ci["hi"]`: the 95% interval for each segment's corrected correlation.

For linked cases (e.g., countries, families), use `kseg_wild_cluster(x, y, 3, clusters)`. `equal_slope_test(x, y, K=3)` gives an F test of equal slopes across the segments.

A step-by-step guide with a worked example is in [HOW_TO.md](HOW_TO.md). The design, hypotheses, seeds, and amendments are in [PROTOCOL.md](PROTOCOL.md).

## Reproducing the paper

Needed only to rerun the simulations and figures. Python 3.13 with NumPy, SciPy, pandas, statsmodels, and matplotlib. Run the commands in this order:

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
python3 ydirection.py rev/ydirection.csv   # segments formed on Y, Document S2 (seed 1110)
```

## License

MIT
