# IPCA Asset Pricing Experiment

## 1. Model

Instrumented Principal Component Analysis (IPCA), following Kelly, Pruitt & Su (2019, 2020):

$$
x_{i,t+1} = \beta_{i,t}' f_{t+1} + \varepsilon_{i,t+1}, \qquad \beta_{i,t} = \Gamma' z_{i,t}
$$

where:

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $x_{i,t+1}$ | Excess return of asset $i$ in month $t+1$ | scalar |
| $z_{i,t}$ | Observable characteristics of asset $i$ at end of month $t$ | $L \times 1$ |
| $\Gamma$ | Characteristic-to-loading map (estimated) | $L \times K$ |
| $\beta_{i,t}$ | Time-varying factor loadings | $K \times 1$ |
| $f_{t+1}$ | Latent factor returns in month $t+1$ | $K \times 1$ |

Characteristics at time $t$ predict returns at time $t+1$. This timing convention is enforced by lagging $Z$ by one month relative to returns when constructing the panel arrays.

### Estimation objective

Both estimators solve the profiled IPCA loss over the Grassmannian:

$$
\min_{\Gamma:\;\Gamma'\Gamma = I_K} \frac{1}{T} \sum_{t=1}^{T} \min_{f_t} \| r_t - Z_t \Gamma f_t \|^2
$$

For a given $\Gamma$, the inner minimization has a closed-form OLS solution:

$$
\hat{f}_t = ((\Lambda_t'\Lambda_t)^{-1} \Lambda_t' r_t, \qquad \Lambda_t = Z_t \Gamma
$$

## 2. Data

### Returns

- **Source**: CRSP monthly stock returns (`data/crsp_monthly_returns.csv`)
- **Fields used**: `PERMNO` (stock identifier), `YYYYMM` (date), `MthRet` (monthly return), `MthCap` (market capitalization)
- **Sample period**: January 1965 -- December 2018

### Characteristics

- **Source**: Gu, Kelly & Xiu (2020) firm characteristics (`data/datashare.csv`)
- **Number of characteristics**: 94
- **Preprocessing**:
  1. Cross-sectional rank transform within each month
  2. Rescaled to $[-0.5, 0.5]$
  3. Missing values filled with 0

### Risk-free rate

- **Source**: Welch & Goyal (2008) predictor data (`data/macro/PredictorData2021 - Monthly.csv`)
- **Used for**: Computing excess returns $x_{i,t} = r_{i,t} - r_{f,t}$

### Universe

- Top 500 stocks by average market capitalization over the full sample
- After filtering: ~500 stocks per month (unbalanced panel due to listing/delisting)
- Missing returns filled with 0 for estimation; NaN mask retained for proper $R^2$ computation

### Panel alignment (lagging)

After merging returns and characteristics on `YYYYMM`, characteristics are lagged by one month:

```
returns  = returns[1:]     # months 2 to T
chars    = chars[:-1]      # months 1 to T-1
```

This ensures $z_{i,t}$ (end of month $t$) is paired with $x_{i,t+1}$ (return during month $t+1$).

## 3. Estimators

### ALS IPCA (`src/models/als_ipca.py`)

Alternating Least Squares:

1. **Initialize** $\Gamma$ randomly on the Grassmannian (best of $5 \times L \times K$ random orthonormal candidates)
2. **Update factors**: $\hat{f}_t = (\Gamma' W_t \Gamma)^{-1} \Gamma' X_t$ using managed portfolios $X_t = Z_t' r_t / N$ and second moments $W_t = Z_t' Z_t / N$
3. **Update $\Gamma$**: Pooled regression via Kronecker products, then QR orthonormalization
4. Repeat until convergence ($\max |\Delta\Gamma|, |\Delta f| < \text{tol}$)

Settings: `max_iter=500`, `tol=1e-6`

### Grassmannian IPCA (`src/models/grassmanian_ipca.py`)

Riemannian optimization via pymanopt:

1. Define cost function using autograd-compatible profiled IPCA loss
2. Optimize on `Grassmann(L, K)` manifold using Riemannian Conjugate Gradient
3. Post-estimation: recover $\hat{f}_t$ via cross-sectional OLS

Settings: `optimizer=ConjugateGradient`, `max_iterations=300`

### Number of factors

$K = 4$ (following Kelly, Pruitt & Su)

## 4. Train / Test Split

| Set | Period | Months |
|-----|--------|--------|
| Train (in-sample) | 1965/01 -- 2004/12 | ~480 |
| Test (out-of-sample) | 2005/01 -- 2018/12 | ~168 |

Split date: `200412` (last in-sample month).

## 5. Evaluation Metrics

### $R^2$ measures

All $R^2$ are computed using only non-NaN return observations. The denominator is $\text{SS}_\text{tot} = \sum x_{i,t}^2$ (no demeaning, since excess returns have mean approximately zero).

The two $R^2$ measures answer fundamentally different questions and differ in what information is available at the time of prediction:

#### Total $R^2$ (in-sample and out-of-sample)

Uses realized factor estimates $\hat{f}_t$:

$$
R^2_\text{total} = 1 - \frac{\sum_{i,t} (x_{i,t} - \hat{\beta}_{i,t}' \hat{f}_t)^2}{\sum_{i,t} x_{i,t}^2}
$$

- **In-sample**: $\hat{f}_t$ from the training estimation
- **Out-of-sample**: $\hat{f}_t$ re-estimated via cross-sectional OLS at each test month:
  $\hat{f}_t = (\Lambda_t' \Lambda_t)^{-1} \Lambda_t' r_t$, using $\Gamma$ fixed from training

**Interpretation**: Total $R^2$ measures how well the model explains the cross-section of returns *after observing* what happened in the market that month. The realized factor $\hat{f}_t$ is extracted from the returns themselves (via cross-sectional regression of $r_t$ on $\Lambda_t = Z_t \Gamma$), so it captures the common component of returns at time $t$. This is an *ex-post* measure: it tells you how much of the cross-sectional variation in returns is explained by the factor structure, but it cannot be used for forecasting because $\hat{f}_t$ is only known after observing $r_t$.

Think of it as asking: *"Given that we know what the market did this month, how well does the model attribute returns to the right stocks?"*

#### Predictive $R^2$

Uses the mean factor $\hat{\lambda} = \bar{f} = \frac{1}{T_\text{train}} \sum_t \hat{f}_t$ instead of realized $\hat{f}_t$:

$$
R^2_\text{pred} = 1 - \frac{\sum_{i,t} (x_{i,t} - \hat{\beta}_{i,t}' \hat{\lambda})^2}{\sum_{i,t} x_{i,t}^2}
$$

**Interpretation**: Predictive $R^2$ measures the model's ability to forecast expected returns *before* the period begins, using only information available at time $t$. The predicted return $\hat{E}[x_{i,t+1}] = z_{i,t}' \Gamma \hat{\lambda}$ depends only on (i) the stock's current characteristics $z_{i,t}$ and (ii) the historical average factor return $\hat{\lambda}$. No future information is used. This is an *ex-ante* measure and the one that matters for portfolio construction.

Think of it as asking: *"Can I use last month's characteristics to predict which stocks will outperform next month?"*

#### Why the gap matters

- **Total $R^2 \gg$ Predictive $R^2$** is expected: most of the return variation comes from time-varying factor realizations (market shocks), which the mean factor $\hat{\lambda}$ cannot capture. The predictive $R^2$ only captures the *expected return* component.
- A positive **OOS Predictive $R^2$** is economically meaningful -- it implies the model identifies characteristics that genuinely predict the cross-section of expected returns, not just in-sample overfitting.
- The portfolio sorts (Section 5.2) are the economic translation of the predictive $R^2$: stocks sorted into quintiles by $z_{i,t}' \Gamma \hat{\lambda}$ should produce a monotonic pattern in realized returns and a profitable long-short spread.

### Portfolio sorts (out-of-sample)

1. At each test month $t$, compute predicted expected returns: $\hat{E}[x_{i,t+1}] = z_{i,t}' \Gamma \hat{\lambda}$
2. Sort stocks into quintiles (Q1 = lowest, Q5 = highest predicted return)
3. Compute equal-weighted monthly returns for each quintile portfolio
4. Long-short portfolio: Q5 $-$ Q1

Reported metrics:
- Annualized average return (monthly mean $\times$ 12)
- Annualized volatility (monthly std $\times \sqrt{12}$)
- Sharpe ratio (annualized return / annualized volatility)

## 6. Summary Table

The notebook produces a table comparing ALS vs Grassmannian:

```
                               ALS IPCA   Grassmann
==========================================================
Final objective                   ...          ...
IS Total R² (%)                   ...          ...
IS Predictive R² (%)              ...          ...
OOS Total R² (%)                  ...          ...
OOS Predictive R² (%)             ...          ...
----------------------------------------------------------
L/S Sharpe (OOS)                  ...          ...
Iterations                        ...          ...
Wall time (s)                     ...          ...
Parameters (L x K)                ...          ...
```

## 7. References

- Kelly, B., Pruitt, S., & Su, Y. (2019). *Characteristics are covariances: A unified model of risk and return.* Journal of Financial Economics, 134(3), 611--632.
- Kelly, B., Pruitt, S., & Su, Y. (2020). *Instrumented principal component analysis.* Working paper.
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical asset pricing via machine learning.* Review of Financial Studies, 33(5), 2223--2273.
- Welch, I., & Goyal, A. (2008). *A comprehensive look at the empirical performance of equity premium prediction.* Review of Financial Studies, 21(4), 1455--1508.
