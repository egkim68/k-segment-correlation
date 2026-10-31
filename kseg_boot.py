"""K-segment correlation with a fixed-X residual bootstrap test (vectorized).
The bootstrap keeps the observed X values and their segments, refits a straight line, and resamples its residuals,
which gives the null distribution of Q for 'linear relationship with constant scatter' for any K and any shape of X."""
import numpy as np
from scipy import stats


def _segments(x, K):
    edges = np.quantile(x, np.linspace(0, 1, K + 1))
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, K - 1)


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


def kseg_boot_cluster(x, y, K, groups, B=999, rng=None):
    """Cluster version: residuals are resampled as whole blocks, one block per cluster (e.g., country or family).
    Blocks are drawn with replacement among clusters of the same size and keep their internal order."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float); y = np.asarray(y, float); groups = np.asarray(groups)
    seg = _segments(x, K); S = x.std(ddof=1)
    rc, Q = _stats(x, y[None, :], seg, K, S)
    b1, b0 = np.polyfit(x, y, 1); fit = b0 + b1 * x; res = y - fit
    idx = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    by_size = {}
    for g, ix in idx.items(): by_size.setdefault(len(ix), []).append(ix)
    Ystar = np.empty((B, len(x)))
    for b in range(B):
        e = np.empty(len(x))
        for size, blocks in by_size.items():
            pick = rng.integers(0, len(blocks), len(blocks))
            for target, src in zip(blocks, pick): e[target] = res[blocks[src]]
        Ystar[b] = fit + e
    _, Qb = _stats(x, Ystar, seg, K, S)
    return dict(rc=rc[0], Q=Q[0], p_boot=(1 + (Qb >= Q[0]).sum()) / (B + 1))
