# Protocol v1.0: K-segment correlation

Frozen 2026-10-30, after two exploratory pilot runs (pilot1.py, pilot2.py; not reported as evidence) and before the confirmatory run.

## Aim
To examine whether a simple K-segment correlation, with each segment's correlation corrected for its narrower range, shows when a single Pearson r is enough to describe the relationship between two variables and where the relationship changes when it is not.

## Method under study
1. Choose a segmenting variable X and a number of segments K.
2. Split the cases into K equal-sized groups at the quantiles of X. The boundaries are never fitted to the data.
3. In each segment k, compute the Pearson correlation r_k and the standard deviation s_k of X.
4. Correct each r_k for range restriction on X (Thorndike Case II), with u_k = S / s_k and S the standard deviation of X in the full sample:
   r_ck = r_k u_k / sqrt(1 − r_k² + r_k² u_k²).
5. Heterogeneity statistic on the Fisher z scale: Q = Σ w_k (z_k − z̄)², with z_k = atanh(r_ck) and w_k the inverse delta-method variance of z_k.
6. Main test: fixed-X residual bootstrap. The observed X values and segments are kept, a straight line is refitted, and its residuals are resampled to create B tables under the hypothesis of a linear relationship with constant scatter. The p value is the share of bootstrap Q values at least as large as the observed Q. The chi-square reference (K − 1 degrees of freedom) is reported for comparison only.
7. Report the K corrected correlations, the range (largest minus smallest), and the bootstrap p value.

Interpretation. A non-significant test means that one r is enough. A significant test means that the relationship bends or that its scatter changes along X, and the segment values show where.

Use rules
- K. Three or four segments by default. Larger K only for large samples (guidance from H6 below).
- Ties. If a quantile boundary falls on a value shared by more than 5% of a segment's cases, the data are flagged, and K is reduced until the boundaries are distinct.
- Size. In very large samples, a difference is reported as practically relevant only if the range of corrected correlations is at least 0.10.

## Simulation design
| Factor | Levels |
|---|---|
| Sample size n | 200, 500, 1,000, 5,000 |
| Number of segments K | 2, 3, 4, 5, 10 |
| Overall correlation ρ | 0.2, 0.4, 0.6 |
| Pattern | A: bivariate normal (null); B: scatter of Y grows with X; C: S-shaped relationship flattening at both ends; D: relationship only in the top quarter of X; E: skewed X (lognormal), linear with constant scatter (null) |
| Replications | 1,000 per condition for A and E; 500 for B, C, D |
| Bootstrap resamples | B = 399 |

Patterns B to D are scaled to the same overall ρ. Seed 303.

## Methods compared
- K-segment test with bootstrap reference (main) and with chi-square reference.
- Uncorrected segment correlations (bias only).
- Breusch–Pagan test (changing scatter); RESET with quadratic and with quadratic plus cubic terms (curvature).

## Hypotheses
- H1. Under A with K ≥ 3, uncorrected segment correlations average at most 0.85 ρ.
- H2. Under A with K ≤ 4 and n / K ≥ 250, the mean corrected segment correlation lies within ±0.03 of ρ.
- H3. Under A and E, the bootstrap K-segment test has a Type I error rate between 0.03 and 0.07 in every condition.
- H4. Under E, the chi-square reference exceeds a Type I error rate of 0.10 in at least one condition (the reason for the bootstrap).
- H5. With K = 3, n = 1,000, and ρ = 0.4, the bootstrap test has power of at least 0.80 under each of B, C, and D.
- H6. Under D with K = 4 and n = 1,000, the top segment has the highest corrected correlation in at least 90% of the rejections (the test shows where the relationship changes).

The comparison with the Breusch–Pagan and RESET tests is descriptive. No superiority is claimed.

Decision rule. A hypothesis is supported if its criterion is met. One-sided p values against each threshold are reported, with a separate label for estimates that meet the threshold without p < 0.05.

## Rule for choosing K
Reported descriptively: for each pattern, the smallest number of cases per segment at which power reaches 0.80 at ρ = 0.4.

## Real-data illustrations
- gapminder (1,704 country-years): GDP per capita and log GDP per capita against life expectancy.
- diamonds (53,940 diamonds): carat against price (tie rule applies).
- faithful (272 eruptions): eruption length against waiting time (small sample).
- Galton family heights (481 sons): midparent height against son height (added by Amendment 2).

## Software and reproducibility
Python with NumPy, SciPy, and statsmodels. Figures as PNG files from a separate script. Code, seeds, and this protocol in a minimal public repository.

## Amendments

Amendment 1 (2026-10-30, after the confirmatory run; real-data illustration only). The tie rule counted a boundary as tied when its value was shared by more than 5% of a segment's cases. In the small faithful data (68 cases per segment), this flagged values shared by only four eruptions. The rule now also requires at least 10 tied cases. The simulation, which uses continuous data, is unaffected.

Amendment 2 (2026-10-30, after the confirmatory run; real-data illustration only). A fourth illustration was added, Galton's family heights (GaltonFamilies in the R package HistData, sons only, n = 481), to show a real case in which one correlation may be enough. Fathers' heights against sons' heights are also analyzed as a check of the tie rule. The simulation and the first three illustrations are unchanged, as the new analyses run after them with the same random number stream.

Amendment 3 (2026-10-31, after the confirmatory run; descriptive results only). The design was run once more at n = 2,000 for all 15 pattern-by-correlation conditions and all K (confirm_n2000.py; seeds [303, i * 10 + 4], not used in the main run), to refine the guidance on cases per segment between n = 1,000 and n = 5,000. Hypotheses H1 to H6 are judged on the original four sample sizes only.

Amendment 4 (2026-10-31, after the confirmatory run; real-data illustration only). As gapminder links the 12 years of each country, a cluster bootstrap by country was added (cluster_gapminder.py, seed 606), in which the residuals of each country are resampled as one block. Further analyses run at the same time (non-normal errors, combined standard tests, segments of Y) are reported in a separate companion study and are not part of this paper.
