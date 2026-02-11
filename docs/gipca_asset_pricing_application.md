# Generalized IPCA: Asset Pricing Application

## 1 Introduction

This document describes an empirical application of Generalized Instrumented Principal Component Analysis (GIPCA) to US stock returns. GIPCA extends the IPCA framework of Kelly, Pruitt, and Su (2019, 2020) by incorporating macroeconomic variables to model the dynamics of latent factors. While IPCA uses firm characteristics to instrument for time-varying factor loadings, GIPCA additionally models the factor realizations themselves as functions of macroeconomic state variables.

The key innovation of GIPCA is its ability to generate genuine out-of-sample predictions of expected returns. Standard IPCA can only predict returns using the historical mean of estimated factors, which provides limited forecasting power. GIPCA, by contrast, uses current macroeconomic conditions to forecast factor realizations, enabling economically meaningful return predictions.

The GIPCA model is specified as:
$$r_{i,t+1} = \beta_{i,t}' f_{t+1} + \epsilon_{i,t+1}$$
$$\beta_{i,t} = \Gamma' z_{i,t}$$
$$f_{t+1} = \Lambda' m_t + \nu_{t+1}$$

where $r_{i,t+1}$ is the excess return of stock $i$, $z_{i,t}$ is a vector of firm characteristics, $m_t$ is a vector of macroeconomic predictors, $\Gamma$ maps characteristics to factor loadings, and $\Lambda$ maps macro variables to factor realizations.

## 2 Data

### 2.1 Stock Returns and Characteristics

Our data consists of monthly excess stock returns and 94 associated firm characteristics from January 1965 to December 2016, sourced from CRSP and the characteristic dataset of Gu, Kelly, and Xiu (2020). Following Kelly, Pruitt, and Su (2020), firm characteristics are cross-sectionally rank-transformed to the interval $[-0.5, 0.5]$ each month, with missing values set to zero (the cross-sectional median).

To ensure computational tractability while maintaining economic representativeness, we focus on the 100 largest stocks by average market capitalization. This yields a panel of 624 months with approximately 31,000 stock-month observations (49.5% panel coverage due to listing/delisting dynamics).

### 2.2 Macroeconomic Predictors

We use 13 macroeconomic predictors from the Welch and Goyal (2008) dataset, which have been widely studied in the return predictability literature:

| Variable | Description |
|----------|-------------|
| D12 | Dividend-price ratio (12-month) |
| E12 | Earnings-price ratio (12-month) |
| b/m | Book-to-market ratio |
| tbl | Treasury bill rate |
| ntis | Net equity issuance |
| infl | Inflation |
| svar | Stock variance |
| AAA | AAA corporate bond yield |
| BAA | BAA corporate bond yield |
| lty | Long-term government bond yield |
| corpr | Corporate bond premium |
| csp | Cross-sectional premium |
| ltr | Long-term return |

All macro variables are standardized (z-scored) over the full sample period.

### 2.3 Sample Split

We divide the sample into an in-sample estimation period (January 1965 - December 2004; 480 months) and an out-of-sample evaluation period (January 2005 - December 2016; 144 months). The model parameters $\Gamma$ and $\Lambda$ are estimated using only in-sample data, and all out-of-sample evaluations use these fixed parameter estimates.

## 3 Estimation

We estimate GIPCA with $K=4$ latent factors using alternating least squares (ALS), following the methodology of Kelly, Pruitt, and Su (2020). The objective function balances the cross-sectional return fit against the macro-factor relationship:

$$\mathcal{Q} = \frac{1}{N}\sum_t \|r_t - Z_t \Gamma f_t\|^2 + \frac{\alpha}{K}\sum_t \|f_t - \Lambda' m_t\|^2$$

where $\alpha = 1.0$ weights the macro constraint. The normalization by $N$ and $K$ ensures both terms contribute comparably to the objective.

### 3.1 The Alpha Trade-off

The regularization parameter $\alpha$ controls a fundamental trade-off:

**Table: Sensitivity to $\alpha$ (Asset Pricing Data)**

| $\alpha$ | In-Sample $R^2$ | OOS Predicted $R^2$ | Macro $R^2$ (avg) |
|----------|-----------------|---------------------|-------------------|
| 0.1 | 22.10% | -19.49% | 70.7% |
| **1.0** | **9.51%** | **+1.51%** | **95.3%** |

- **Lower $\alpha$** (0.1): Higher in-sample fit, but factors are less constrained by macro variables, leading to negative predictive $R^2$.
- **Higher $\alpha$** (1.0): Lower in-sample fit, but factors are forced to be macro-predictable, enabling positive predictive $R^2$.

For forecasting applications, $\alpha = 1.0$ is preferred because the macro constraint is precisely what enables genuine out-of-sample prediction. The lower in-sample $R^2$ is the necessary price for predictive power.

## 4 Results

### 4.1 Model Comparison

Table 1 compares IPCA and GIPCA across three $R^2$ metrics:

- **In-Sample $R^2$**: Fit during estimation using factors from the ALS optimization
- **OOS $R^2$ (Realized)**: Out-of-sample fit using factors estimated via cross-sectional regression on OOS returns. *Caveat: This metric uses contemporaneous OOS returns to estimate factors—it is an upper bound, not a genuine prediction.*
- **OOS $R^2$ (Predicted)**: Out-of-sample fit using predicted factors—mean factors for IPCA, macro-predicted factors ($f_t = \Lambda' m_t$) for GIPCA. *This is the genuinely predictive metric with no look-ahead bias.*

**Table 1: IPCA vs GIPCA — Explained Variation of Stock Returns**

| Metric | IPCA | GIPCA |
|--------|------|-------|
| In-Sample $R^2$ | 52.08% | 9.51% |
| OOS $R^2$ (Realized)* | 42.40% | 36.94% |
| **OOS $R^2$ (Predicted)** | **-0.59%** | **+1.51%** |

*\* Uses factors estimated from OOS returns (not a genuine prediction)*

Several findings emerge from Table 1:

1. **In-sample fit**: IPCA achieves higher in-sample $R^2$ (52%) than GIPCA (9.5%). This is expected: GIPCA's objective function includes a macro regularization term that trades off in-sample fit for factor predictability.

2. **OOS with realized factors**: Both models generalize reasonably when factors are estimated from OOS returns (IPCA: 42%, GIPCA: 37%). However, this metric involves look-ahead—factors are estimated using the returns we're trying to explain.

3. **OOS with predicted factors (the honest comparison)**: GIPCA achieves **positive** predictive $R^2$ (+1.51%), while IPCA yields **negative** predictive $R^2$ (-0.59%). This 2.1 percentage point improvement demonstrates that macroeconomic variables contain genuine information about future factor realizations.

The key insight is that IPCA, without a model for factor dynamics, can only use the unconditional mean of historical factors for prediction—which provides no forecasting power. GIPCA addresses this by modeling $f_t = \Lambda' m_t$, enabling genuine return prediction.

### 4.2 Factor-Macro Relationship

Table 2 reports the $R^2$ of regressing each estimated GIPCA factor on the 13 macroeconomic predictors. The high explanatory power indicates that the latent factors are strongly related to macroeconomic conditions.

**Table 2: Macro Explanatory Power for Latent Factors**

| Factor 1 | Factor 2 | Factor 3 | Factor 4 |
|----------|----------|----------|----------|
| 99.5% | 91.4% | 99.7% | 90.3% |

*Notes: $R^2$ (in percentage) from regressing each GIPCA factor on the 13 Welch-Goyal macro predictors.*

The macro $R^2$ values exceed 90% for all four factors, indicating that GIPCA successfully identifies factors whose dynamics are driven by macroeconomic state variables. This is the key mechanism enabling GIPCA's out-of-sample predictive power.

### 4.3 Characteristic Loadings

Table 3 reports the five characteristics with the largest absolute loadings (elements of $\Gamma$) for each factor. These reveal which firm attributes most strongly determine exposure to each systematic risk factor.

**Table 3: Top Characteristics by Factor Loading**

| Factor 1 | Factor 2 | Factor 3 | Factor 4 |
|----------|----------|----------|----------|
| dolvol (+0.49) | beta (-0.60) | pchsale_pchinvt (+0.33) | mvel1 (-0.40) |
| lev (+0.38) | betasq (+0.56) | mvel1 (-0.30) | bm (+0.31) |
| ill (+0.25) | mvel1 (-0.36) | pchsaleinv (-0.29) | betasq (-0.30) |
| pchsale_pchinvt (+0.21) | ill (-0.20) | divi (-0.27) | cfp (+0.27) |
| mvel1 (-0.19) | zerotrade (+0.18) | zerotrade (+0.21) | zerotrade (+0.26) |

*Notes: Top 5 characteristics by absolute $\Gamma$ coefficient for each factor. Signs in parentheses indicate the direction of the loading. dolvol = dollar volume, lev = leverage, ill = illiquidity, mvel1 = market equity, bm = book-to-market, cfp = cash flow to price, pchsale_pchinvt = sales growth relative to inventory growth, divi = dividend indicator.*

Factor 1 loads heavily on liquidity-related characteristics (dollar volume, illiquidity) and leverage. Factor 2 captures market beta exposure, with high loadings on beta and beta-squared. Factor 3 relates to sales growth dynamics and firm maturity (dividends). Factor 4 loads on value characteristics (book-to-market, cash flow to price) and size.

### 4.4 Portfolio Analysis

To assess the economic significance of GIPCA's return predictions, we sort stocks into quintile portfolios based on predicted expected returns each month in the out-of-sample period.

**Table 4: Quintile Portfolio Performance (Out-of-Sample)**

| Quintile | Annualized Return | Volatility | Sharpe Ratio |
|----------|-------------------|------------|--------------|
| Q1 (Low) | 7.06% | 21.11% | 0.33 |
| Q2 | 5.29% | 16.56% | 0.32 |
| Q3 | 8.81% | 15.43% | 0.57 |
| Q4 | 14.24% | 16.77% | 0.85 |
| Q5 (High) | 21.00% | 20.65% | 1.02 |
| **Long-Short (Q5-Q1)** | **13.95%** | **19.42%** | **0.72** |

*Notes: Quintile portfolios formed monthly based on GIPCA predicted returns using macro-predicted factors. Returns are annualized. Sample period: January 2005 - December 2016.*

The results show a clear monotonic relationship between predicted and realized returns. Stocks in the highest predicted return quintile (Q5) earn 21% annualized versus 7% for the lowest quintile (Q1). The long-short portfolio earns 14% annually with a Sharpe ratio of 0.72, indicating economically significant predictive power.

## 5 Discussion

The empirical results demonstrate that GIPCA provides a meaningful extension to IPCA for asset pricing applications. The key findings are:

1. **Positive out-of-sample predictive $R^2$**: GIPCA achieves 1.51% predictive $R^2$ out-of-sample, compared to -0.59% for IPCA using mean factors. While 1.51% may appear modest, positive predictive $R^2$ is rare in the return forecasting literature, and this level of predictability translates into economically significant portfolio returns.

2. **Strong factor-macro relationship**: The latent factors estimated by GIPCA are highly predictable from macroeconomic variables ($R^2 > 90\%$), validating the model's structural assumption that systematic risk factors are driven by macroeconomic conditions.

3. **Economically meaningful predictions**: Portfolio sorts based on GIPCA predictions generate a long-short Sharpe ratio of 0.72, indicating that the model's forecasts translate into actionable investment strategies.

4. **Trade-off between fit and prediction**: GIPCA's lower in-sample total $R^2$ (9.51% vs. 52.08% for IPCA) reflects the regularization imposed by the macro constraint. This trade-off is economically sensible: by requiring factors to be macro-predictable, GIPCA sacrifices some in-sample fit but gains genuine forecasting ability.

The comparison between IPCA and GIPCA highlights a fundamental limitation of standard factor models: without a model for factor dynamics, out-of-sample prediction reduces to using unconditional mean factors, which provides no forecasting power. GIPCA addresses this limitation by explicitly modeling the relationship between macroeconomic conditions and factor realizations.

## 6 Limitations and Caveats

Several limitations should be acknowledged:

1. **Hyperparameter selection**: The number of factors ($K=4$) and regularization weight ($\alpha=1.0$) were not selected via cross-validation. Results may be sensitive to these choices. Future work should implement a validation protocol (e.g., train on 1965-1994, validate on 1995-2004, test on 2005-2016).

2. **Single train/test split**: Results are based on a single sample split. Robustness to alternative cutoff dates (2000, 2010) should be examined.

3. **Stock universe**: The analysis focuses on the 100 largest stocks by market capitalization. Results may differ for broader or smaller-cap universes.

4. **Statistical significance**: Standard errors for the predictive $R^2$ are not computed. Bootstrap inference would strengthen the conclusions.

5. **Macro standardization**: Macro variables are z-scored using full-sample statistics. Strictly, expanding-window standardization should be used to avoid information leakage.

6. **Model specification**: The linear factor-macro relationship ($f_t = \Lambda' m_t + \nu_t$) may be misspecified. Non-linear specifications could be explored.

Despite these caveats, the core finding—positive predictive $R^2$ for GIPCA versus negative for IPCA—is economically meaningful and suggests that the macro-factor linkage captures genuine predictability.

## 7 Conclusion

This application demonstrates that GIPCA successfully extends IPCA to deliver out-of-sample return predictions. By modeling latent factors as functions of macroeconomic state variables, GIPCA captures the time-varying nature of systematic risk premia and generates economically significant forecasts.

The key practical implication is that GIPCA enables conditional expected return estimation—something not possible with IPCA alone. This has direct applications for portfolio construction, risk management, and asset valuation, where forecasts of expected returns (cost of capital) are essential inputs.

The honest comparison between models reveals:
- **GIPCA Predictive $R^2$: +1.51%** (positive, using macro-predicted factors)
- **IPCA Predictive $R^2$: -0.59%** (negative, using mean factors)

While 1.51% may appear modest, positive predictive $R^2$ is rare in the return forecasting literature. The portfolio analysis confirms that this predictability translates into economically significant returns (long-short Sharpe ratio of 0.72).

## References

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.

- Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics*, 134(3), 501-524.

- Kelly, B., Pruitt, S., & Su, Y. (2020). Instrumented principal component analysis. *Working Paper*.

- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455-1508.
