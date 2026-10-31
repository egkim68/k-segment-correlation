# How to calculate segment correlations

This page shows how the segment correlations in the paper are calculated. The same example appears in Document S1 of the paper.

**Inputs**
- `x`: the segmenting variable, one value per case (in the example, eruption length)
- `y`: the other variable, for the same cases in the same order (in the example, waiting time)
- `K`: the number of segments (3 by default)

**Output**: one row per segment: segment number, number of cases, raw r, range ratio u, and corrected segment r.

**Steps**
1. Cut `x` at its K quantiles. A case on a cut point goes to the higher segment.
2. Compute the Pearson r within each segment.
3. Compute u = SD(x, full sample) / SD(x, segment).
4. Correct r for the narrower range (Thorndike's Case II): r_c = r u / sqrt(1 - r^2 + r^2 u^2).
5. Test whether the corrected values differ with `kseg_boot` (bootstrap test).

```python
import numpy as np

def segment_r(x, y, K=3):
    x, y = np.asarray(x, float), np.asarray(y, float)
    # Step 1: cut x at its quantiles (a case on a cut point goes up)
    edges = np.quantile(x, np.linspace(0, 1, K + 1))
    seg = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, K - 1)
    S = x.std(ddof=1)                       # SD of x in the full sample
    out = []
    for k in range(K):
        xs, ys = x[seg == k], y[seg == k]
        r = np.corrcoef(xs, ys)[0, 1]                   # Step 2: raw r
        u = S / xs.std(ddof=1)                          # Step 3: range ratio
        rc = r * u / np.sqrt(1 - r**2 + r**2 * u**2)    # Step 4: corrected r
        out.append((k + 1, len(xs), round(float(r), 2),
                    round(float(u), 2), round(float(rc), 2)))
    return out

# Example (replace the file and column names with your own data)
import pandas as pd
data = pd.read_csv("faithful.csv")
for row in segment_r(data["eruptions"], data["waiting"], K=3):
    print(row)

# Output:
# (1, 91, 0.23, 5.61, 0.8)
# (2, 89, 0.6, 2.76, 0.9)
# (3, 92, 0.12, 5.88, 0.58)
```

Step 5, the bootstrap test:

```python
from kseg_boot import kseg_boot
res = kseg_boot(data["eruptions"].values, data["waiting"].values.astype(float), K=3, B=999)
print(res["rc"], res["p_boot"])   # same segment correlations; p near 0.16 (varies slightly with the random seed)
```

Run `python3 get_data.py .` first to download faithful.csv.
