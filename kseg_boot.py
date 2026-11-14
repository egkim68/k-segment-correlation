"""K-segment correlation with a fixed-X residual bootstrap test (vectorized).
The bootstrap keeps the observed X values and their segments, refits a straight line, and resamples its residuals,
which gives the null distribution of Q for 'linear relationship with constant scatter' for any K and any shape of X."""
import numpy as np
from scipy import stats


def _segments(x, K):
    edges = np.quantile(x, np.linspace(0, 1, K + 1))
    seg = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, K - 1)
    small = np.bincount(seg, minlength=K) < 3
    if small.any():   # heavy ties can leave a segment (almost) empty
        raise ValueError(f"segment(s) {[int(i) + 1 for i in np.flatnonzero(small)]} have fewer than 3 cases; use a smaller K")
    return seg


def _stats(x, Y, seg, K, S):
    """Y: (B, n). Returns corrected r (B, K) and Q (B,)."""
    rc_all, z_all, w_all = [], [], []
    for k in range(K):
        m = seg == k; xs = x[m]; n = m.sum()
        xc = xs - xs.mean(); ys = Y[:, m]; yc = ys - ys.mean(1, keepdims=True)
        r = (yc @ xc) / np.sqrt((yc ** 2).sum(1) * (xc ** 2).sum())
        u = S / xs.std(ddof=1)
        den = 1 - r ** 2 + r ** 2 * u ** 2
        rc = np.clip(r * u / np.sqrt(den), -0.999999, 0.999999)
        var_z = (u / den ** 1.5) ** 2 * (1 - r ** 2) ** 2 / (n - 1) / (1 - rc ** 2) ** 2
        rc_all.append(rc); z_all.append(np.arctanh(rc)); w_all.append(1 / var_z)
    rc = np.stack(rc_all, 1); z = np.stack(z_all, 1); w = np.stack(w_all, 1)
    zbar = (w * z).sum(1, keepdims=True) / w.sum(1, keepdims=True)
    return rc, (w * (z - zbar) ** 2).sum(1)


def kseg_boot(x, y, K, B=199, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    seg = _segments(x, K); S = x.std(ddof=1)
    rc, Q = _stats(x, y[None, :], seg, K, S)
    b1, b0 = np.polyfit(x, y, 1); fit = b0 + b1 * x; res = y - fit
    Ystar = fit[None, :] + res[rng.integers(0, n, (B, n))]
    _, Qb = _stats(x, Ystar, seg, K, S)
    p_boot = (1 + (Qb >= Q[0]).sum()) / (B + 1)
    return dict(rc=rc[0], Q=Q[0], p_chi2=stats.chi2.sf(Q[0], K - 1), p_boot=p_boot, range_rc=rc[0].max() - rc[0].min())



def segment_ci(x, y, K, B=999, level=0.95, rng=None):
    """Percentile intervals for each corrected segment correlation.
    Cases are resampled within each segment; the segments and the full-sample SD of x are held fixed."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float); y = np.asarray(y, float)
    seg = _segments(x, K); S = x.std(ddof=1); rcs = np.empty((B, K))
    for k in range(K):
        idx = np.flatnonzero(seg == k); draw = idx[rng.integers(0, len(idx), (B, len(idx)))]
        xs, ys = x[draw], y[draw]
        xc = xs - xs.mean(1, keepdims=True); yc = ys - ys.mean(1, keepdims=True)
        r = (xc * yc).sum(1) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
        u = S / xs.std(1, ddof=1)
        rcs[:, k] = r * u / np.sqrt(1 - r ** 2 + r ** 2 * u ** 2)
    a = (1 - level) / 2
    return dict(lo=np.quantile(rcs, a, 0), hi=np.quantile(rcs, 1 - a, 0))


def kseg_wild_cluster(x, y, K, groups, B=999, rng=None):
    """Wild cluster bootstrap for linked cases (e.g., countries or families): every case keeps its own residual,
    and the residuals of one cluster are multiplied by the same random sign (+1 or -1)."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float); y = np.asarray(y, float)
    seg = _segments(x, K); S = x.std(ddof=1)
    rc, Q = _stats(x, y[None, :], seg, K, S)
    b1, b0 = np.polyfit(x, y, 1); fit = b0 + b1 * x; res = y - fit
    _, gi = np.unique(np.asarray(groups), return_inverse=True)
    signs = rng.choice([-1.0, 1.0], (B, gi.max() + 1))
    _, Qb = _stats(x, fit[None, :] + signs[:, gi] * res[None, :], seg, K, S)
    return dict(rc=rc[0], Q=Q[0], p_boot=(1 + (Qb >= Q[0]).sum()) / (B + 1))


def equal_slope_test(x, y, K):
    """F test of one common slope against separate slopes in the K segments (separate intercepts in both)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    seg = _segments(x, K); D = np.eye(K)[seg]
    rss = lambda X: ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum()
    r0, r1 = rss(np.column_stack([D, x])), rss(np.column_stack([D, D * x[:, None]]))
    df1, df2 = K - 1, len(x) - 2 * K
    return stats.f.sf(((r0 - r1) / df1) / (r1 / df2), df1, df2)
