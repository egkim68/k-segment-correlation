# K-segment correlation

Code and protocol for "Is One Correlation Enough? K-Segment Correlations for Relationships That Change Across the Data".

The method splits the cases into K equal groups at the quantiles of X, corrects each group's correlation for range restriction (Thorndike's Case II), and tests whether the corrected values differ with a fixed-X residual bootstrap (`kseg_boot.py`).

## Use

```python
from kseg_boot import kseg_boot
res = kseg_boot(x, y, K=3, B=999)   # corrected segment correlations, Q, bootstrap p value
```

## Reproduce

Python 3.13 with NumPy, SciPy, pandas, statsmodels, and matplotlib.

```
python3 get_data.py .
python3 confirm.py confirm.csv            # main run (seed 303)
python3 confirm_n2000.py confirm_n2000.csv
mkdir -p out
python3 analysis.py confirm.csv out confirm_n2000.csv
python3 illustration.py . out
python3 cluster_gapminder.py .
python3 make_figures.py out . figures 600
```

`PROTOCOL.md` gives the design, hypotheses, seeds, and amendments.

## License

MIT

