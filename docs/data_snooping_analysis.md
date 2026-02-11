# Data Snooping Analysis: A Critical Review

This document provides a critical examination of potential data snooping issues in both the synthetic data analysis and the asset pricing application of GIPCA.

## Executive Summary

**Key Finding**: The asset pricing results contain a metric that can be misinterpreted. The "OOS R² (Realized)" uses factors estimated from OOS returns, which is an upper bound but not a genuine prediction. The **OOS R² (Predicted)** is the honest out-of-sample metric.

**Table: IPCA vs GIPCA — Complete Comparison**

| Metric | IPCA | GIPCA | Notes |
|--------|------|-------|-------|
| In-Sample $R^2$ | 52.08% | 9.51% | GIPCA trades fit for predictability |
| OOS $R^2$ (Realized)* | 42.40% | 36.94% | Upper bound (uses OOS returns) |
| **OOS $R^2$ (Predicted)** | **-0.59%** | **+1.51%** | **Genuinely out-of-sample** |

*\* Factors estimated from OOS returns—not a prediction*

The comparison that matters: **GIPCA (+1.51%) vs IPCA (-0.59%)** for OOS Predicted $R^2$.

---

## 1. Asset Pricing Application: Critical Issues

### 1.1 The "OOS Total R²" Metric is NOT Out-of-Sample

**Problem**: The notebook computes "OOS Total R² with realized factors" as follows:

```python
# OOS with realized factors - estimate via cross-sectional regression
test_factors = np.zeros((T_test, K), dtype=np.float32)
for t in range(T_test):
    loadings_t = test_chars[t] @ Gamma_arr  # (N, K)
    f_t, *_ = np.linalg.lstsq(loadings_t, test_returns_filled[t], rcond=None)
    test_factors[t] = f_t
```

This estimates factors by regressing **OOS returns** on loadings. The factors are chosen to minimize the cross-sectional residuals in the test period—this is **look-ahead bias**.

**Why this is problematic**:
- At time $t$ in the OOS period, we use returns $r_t$ to estimate $f_t$
- Then we compute $\hat{r}_t = Z_t \Gamma f_t$ and measure fit
- This is circular: we're fitting to data we're trying to predict

**Why OOS > IS for GIPCA**: The apparent paradox (36.94% OOS vs 9.51% IS) occurs because:
- IS factors come from the ALS optimization with macro constraint
- OOS factors are estimated purely to minimize cross-sectional error (no macro constraint)
- The unconstrained OOS regression naturally achieves higher fit

**The honest metric**: The **Predictive R²** (1.51% OOS) uses macro-predicted factors:
```python
test_factors_macro = test_macro @ Lambda_arr  # No look-ahead
```
This is genuinely out-of-sample because factors are predicted from macro variables known at time $t$.

### 1.2 Hyperparameter Selection Without Cross-Validation

**Number of factors (K=4)**:
- Chosen without documented justification
- Kelly, Pruitt, and Su (2020) use K=5 in their main specification
- No sensitivity analysis to K was performed
- Risk: K may have been chosen after observing results with multiple values

**Regularization weight (α=1.0)**:
- No cross-validation or tuning procedure documented
- This parameter critically affects the IS/OOS trade-off
- Different α values could yield very different results

**Recommendation**: Perform cross-validation on the training period:
- Split training data into estimation (1965-1994) and validation (1995-2004)
- Select K and α that maximize validation predictive R²
- Then evaluate on true OOS period (2005-2016)

### 1.3 Stock Universe Selection

**Current approach**: Top 100 stocks by average market capitalization

**Concerns**:
- Selection criterion uses full-sample average market cap (includes OOS period)
- The number 100 is arbitrary—could have been chosen after experimentation
- Large-cap stocks may be easier to predict (more analyst coverage, more efficient)

**Recommendation**:
- Use only IS data to select stocks (average market cap 1965-2004)
- Report results for multiple universe sizes (50, 100, 250, 500, all stocks)
- Or use the full cross-section with appropriate weighting

### 1.4 Macro Variable Standardization

**Current approach**:
```python
macro_means = macro_df[available_macro].mean()  # Full sample
macro_stds = macro_df[available_macro].std()    # Full sample
macro_df[available_macro] = (macro_df[available_macro] - macro_means) / macro_stds
```

**Problem**: Standardization uses **full-sample** statistics, including OOS data.

**Impact**: Likely minor, but technically constitutes information leakage. The OOS macro values are transformed using statistics that include their own values.

**Recommendation**: Use expanding-window standardization:
```python
# For each t, standardize using only data up to t-1
```

### 1.5 Train/Test Split

**Current split**: 1965-2004 (train) / 2005-2016 (test)

**Concerns**:
- This is a single split—results may be sensitive to the cutoff date
- 2005-2016 includes the 2008 financial crisis, which may have unique dynamics
- No robustness check with alternative splits

**Recommendation**:
- Report results for multiple split dates (2000, 2005, 2010)
- Or use rolling/expanding window estimation

### 1.6 Characteristic and Macro Variable Selection

**Characteristics**: All 94 Gu et al. (2020) characteristics used—**this is good** (no selection)

**Macro variables**: All 13 Welch-Goyal predictors used—**this is good** (standard set from literature)

These choices follow established literature and do not appear to involve snooping.

---

## 2. Synthetic Data Analysis: Assessment

The synthetic analysis has **fewer data snooping concerns** because the DGP is known. However, some issues remain:

### 2.1 DGP Designed to Favor GIPCA

**Issue**: The synthetic data is generated according to the GIPCA model:
- $f_t = \Lambda' m_t + \nu_t$ (factors driven by macro)
- $r_{i,t} = z_{i,t}' \Gamma f_t + \epsilon_{i,t}$ (returns driven by characteristic-based loadings)

This is the exact model GIPCA estimates, so good performance is expected by construction.

**Implication**: The synthetic results validate that GIPCA works when its assumptions hold, but don't prove it works in general.

**Recommendation**: Test on DGPs that violate GIPCA assumptions:
- Factors not driven by macro (pure noise)
- Non-linear factor-macro relationship
- Time-varying Gamma

### 2.2 Noise Parameters

**Current values**: $\sigma_\epsilon = 0.05$, $\sigma_\nu = 0.3$

**Issue**: These were chosen to produce reasonable signal-to-noise ratios, but the sensitivity analysis shows results are highly dependent on $\sigma_\epsilon$.

**Positive**: The sensitivity analysis in Table 5 is valuable and shows GIPCA's limitations at high noise.

### 2.3 OOS Evaluation in Synthetic Data

**Same issue as asset pricing**: The "OOS Total R²" in the synthetic comparison also uses factors estimated via cross-sectional regression on OOS data.

From the notebook:
```python
# Out-of-sample: estimate factors via cross-sectional regression
gipca_test_pred = np.zeros_like(test_returns)
for t in range(T_test):
    loadings_t = test_Z[t] @ gipca_Gamma
    f_t, *_ = np.linalg.lstsq(loadings_t, test_returns[t], rcond=None)
    gipca_test_pred[t] = loadings_t @ f_t
```

This inflates OOS performance for both IPCA and GIPCA equally, so the relative comparison is fair, but the absolute numbers are optimistic.

---

## 3. What the Results Actually Show

### Honest Interpretation of Asset Pricing Results

**Table: Complete Picture**

| Metric | IPCA | GIPCA |
|--------|------|-------|
| In-Sample $R^2$ | 52.08% | 9.51% |
| OOS $R^2$ (Realized)* | 42.40% | 36.94% |
| **OOS $R^2$ (Predicted)** | **-0.59%** | **+1.51%** |

**Key insight**: GIPCA's 1.51% predictive R² is modest but **positive**, which is rare in return forecasting. The comparison to IPCA's -0.59% is meaningful.

**However**, we cannot conclude:
- That 1.51% is statistically significant (no standard errors computed)
- That results generalize to other time periods
- That hyperparameter choices didn't inflate results

### Honest Interpretation of Synthetic Results

| Metric | PCA | IPCA | GIPCA |
|--------|-----|------|-------|
| In-Sample $R^2$ | 67.87% | 99.20% | 72.98% |
| OOS $R^2$ (Realized)* | 45.37% | 98.63% | 98.25% |

**Key insight**: IPCA and GIPCA both recover the DGP well. GIPCA's slightly lower in-sample performance reflects the macro constraint, but both achieve similar OOS fit and far outperform PCA.

**Parameter recovery**: IPCA recovers Gamma better (distance 0.026 vs 0.187) because it's optimized purely for cross-sectional fit, while GIPCA trades some Gamma accuracy for macro-predictable factors.

---

## 4. Recommendations for Rigorous Evaluation

### 4.1 Immediate Fixes

1. **Remove misleading "OOS Total R²" metric** from documentation
2. **Clarify that Predictive R² is the honest metric**
3. **Document hyperparameter choices** and their justification

### 4.2 Cross-Validation Protocol

```
Training: 1965-1994 (360 months)
Validation: 1995-2004 (120 months)  <- Select K, α here
Test: 2005-2016 (144 months)        <- Final evaluation
```

### 4.3 Statistical Inference

- Compute standard errors for R² using bootstrap
- Test H₀: Predictive R² ≤ 0 vs H₁: Predictive R² > 0
- Compare GIPCA vs IPCA with Diebold-Mariano test

### 4.4 Robustness Checks

- Multiple train/test splits
- Multiple stock universes (by size, by coverage)
- Varying K (2, 3, 4, 5, 6 factors)
- Varying α (0.1, 0.5, 1.0, 2.0, 5.0)

### 4.5 Expanding Window Evaluation

Instead of single train/test split:
```
t=1995: Train on 1965-1994, predict 1995
t=1996: Train on 1965-1995, predict 1996
...
t=2016: Train on 1965-2015, predict 2016
```
This provides 22 years of genuine OOS predictions.

---

## 5. Conclusion

The GIPCA results are **promising but not conclusive**. The key findings hold up to scrutiny:

**Valid claims**:
- GIPCA achieves positive predictive R² (1.51%) while IPCA achieves negative (-0.59%)
- This difference suggests macro variables contain information about future factor realizations
- The portfolio analysis shows monotonic quintile returns (Sharpe 0.72 for long-short)

**Claims requiring more evidence**:
- That 1.51% is statistically significant
- That results are robust to hyperparameter choices
- That results generalize to other periods and stock universes

**Invalid metrics in current documentation**:
- The "OOS Total R² = 36.94%" should not be reported as an out-of-sample metric
- This number uses look-ahead information and is misleading

The honest comparison is:
- **GIPCA Predictive R²: +1.51%** (genuinely out-of-sample)
- **IPCA Predictive R²: -0.59%** (genuinely out-of-sample)

This 2.1 percentage point improvement is economically meaningful if it holds up to further testing.
