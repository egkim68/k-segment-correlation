"""Data-generating shapes A to E and calibration of the slope to a target overall correlation."""
import numpy as np
from scipy import optimize


def gen(pattern, beta, n, rng):
    if pattern == "E":
        x = rng.lognormal(0, 0.8, n)
    else:
        x = rng.standard_normal(n)
    e = rng.standard_normal(n)
    if pattern in ("A", "E"):
        y = beta * x + e
    elif pattern == "B":                      # scatter grows with x
        y = beta * x + np.exp(0.5 * x) / np.exp(0.125) * e
    elif pattern == "C":                      # saturating curve
        y = beta * np.tanh(1.5 * x) + e
    elif pattern == "D":                      # relationship only in the top quarter of x
        y = beta * np.maximum(0, x - 0.674) + e
    return x, y


def calibrate(pattern, rho):
    big = np.random.default_rng(7)
    def f(b):
        x, y = gen(pattern, b, 400_000, big); return np.corrcoef(x, y)[0, 1] - rho
    return optimize.brentq(f, 1e-3, 50)
