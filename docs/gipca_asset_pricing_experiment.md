# GIPCA Asset Pricing Experiment

## 1. Model

Generalized Instrumented Principal Component Analysis (GIPCA) extends IPCA by decomposing latent factors into a macro-predicted component and a residual:

$$
x_{i,t+1} = \beta_{i,t}' f_{t+1} + \varepsilon_{i,t+1}, \qquad \beta_{i,t} = \Gamma' z_{i,t}, \qquad f_t = f^0_t + \Delta' m_t, \qquad f^0_t \perp m_t
$$

where:

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $x_{i,t+1}$ | Excess return of asset $i$ in month $t+1$ | scalar |
| $z_{i,t}$ | Observable characteristics of asset $i$ at end of month $t$ | $L \times 1$ |
| $\Gamma$ | Characteristic-to-loading map | $L \times K$ |
| $f_t$ | Latent factor returns | $K \times 1$ |
| $\Delta$ | Macro-to-factor loading matrix | $K \times R$ |
| $m_t$ | Macroeconomic variables at time $t$ | $R \times 1$ |
| $f^0_t$ | Residual factor component orthogonal to macro | $K \times 1$ |

### Penalised objective

$$
\min_{\Gamma,\Delta} \frac{1}{T} \sum_{t=1}^{T} \min_{f_t} \left[ \| r_t - Z_t \Gamma f_t \|^2 + \alpha \| f_t - \Delta' m_t \|^2 \right]
$$

The penalty parameter $\alpha$ controls the role of macro variables:

| $\alpha$ | Regime | Behaviour |
|----------|--------|-----------|
| $= 0$ | Pure IPCA | $f^0_t$ is free; $\Delta$ is post-hoc OLS; macro has no effect on $\Gamma$ estimation |
| $> 0$ | Soft GIPCA | Penalises $f^0_t$; macro influences $\Gamma$ estimation through factors |
| $\to \infty$ | Hard GIPCA | Forces $f_t \approx \Delta' m_t$ (hard constraint) |

All experiments use $\alpha = 0.1$.

### Identification

The decomposition $f_t = f^0_t + \Delta' m_t$ requires $f^0_t \perp m_t$ for identification. Without this, signal can shift freely between $\Delta' m_t$ and $f^0_t$.

### GIPCA predictive signal vs IPCA

The key advantage of GIPCA over IPCA lies in the predictive signal:

- **IPCA** predicts expected returns using the unconditional mean factor: $\hat{E}_t[x_{i,t+1}] = z_{i,t}' \Gamma \hat{\lambda}$, where $\hat{\lambda} = \bar{f}$ is constant over time.
- **GIPCA** predicts expected returns using time-varying macro-conditioned factors: $\hat{E}_t[x_{i,t+1}] = z_{i,t}' \Gamma (\Delta' m_t)$, which adapts to the current macro environment.

GIPCA's predictive signal varies over time through $m_t$, allowing conditional expected returns to respond to changing macroeconomic conditions. IPCA's signal only varies cross-sectionally through $z_{i,t}$.

## 2. Data

### Returns

- **Source**: CRSP monthly stock returns (`data/crsp_monthly_returns.csv`)
- **Fields**: `PERMNO`, `YYYYMM`, `MthRet`, `MthCap`
- **Sample period**: January 1965 -- December 2018

### Characteristics

- **Source**: Gu, Kelly & Xiu (2020) firm characteristics (`data/datashare.csv`)
- **Number of characteristics**: 94--95
- **Preprocessing**: Cross-sectional rank transform to $[-0.5, 0.5]$ within each month; missing values filled with 0

### Risk-free rate

- **FF5 experiment**: Compounded daily RF from Fama-French daily data, converted to decimal
- **WG experiment**: Monthly `Rfree` from Welch & Goyal predictor dataset

### Universe

Two configurations were tested:

| Experiment | Universe | Stocks |
|------------|----------|--------|
| Fama-French 5 | Top 500 by average market cap | ~500 |
| Welch-Goyal | Top 3000 by average market cap | ~3000 |

Missing returns filled with 0 for estimation; NaN mask retained for $R^2$ computation.

## 3. Macro Variable Sets

### Experiment 1: Fama-French 5 Factors

**Source**: `data/F-F_Research_Data_5_Factors_2x3_daily.csv` (daily, compounded to monthly)

| Variable | Description |
|----------|-------------|
| `Mkt-RF` | Market excess return |
| `SMB` | Small minus Big (size) |
| `HML` | High minus Low (value) |
| `RMW` | Robust minus Weak (profitability) |
| `CMA` | Conservative minus Aggressive (investment) |

**Compounding**: Daily returns aggregated to monthly via $r_\text{monthly} = \left(\prod_{d \in \text{month}} (1 + r_d / 100) - 1\right) \times 100$.

$R = 5$ macro predictors.

### Experiment 2: Welch & Goyal Macro Predictors

**Source**: `data/macro/PredictorData2021 - Monthly.csv` (monthly)

| Variable | Description | Construction |
|----------|-------------|--------------|
| `tms` | Term spread | `lty - tbl` |
| `dfy` | Default spread | `BAA - AAA` |
| `svar` | Stock variance | Direct from dataset |
| `infl` | Inflation | Direct from dataset |
| `CRSP_SPvw` | Market return (value-weighted) | Direct from dataset |
| `ltr` | Long-term bond return | Direct from dataset |
| `cs_disp` | Cross-sectional return dispersion | Constructed: $\text{std}(r_{i,t})$ across stocks at each $t$ |

All variables are stationary (spreads, returns, rates) -- no differencing needed.

$R = 7$ macro predictors.

### Macro preprocessing

- **Z-scoring**: Macro variables standardised using training-period mean and standard deviation only. Test-period macro is standardised with the same training statistics. This prevents look-ahead bias.
- **Forward-filling**: Missing macro values forward-filled before z-scoring.

## 4. Timing and Lagging

### Characteristics lag

Same as the IPCA experiment: characteristics at month $t$ are paired with returns at month $t+1$.

### Macro lag (critical for GIPCA)

To ensure no look-ahead bias in the predictive signal, GIPCA is trained on **lagged macro**:

```
train_rets_lag   = returns[1:split]      # returns from month 2 to T_train
train_Z_lag      = chars[1:split]        # chars from month 2 to T_train
train_macro_lag  = macro[0:split-1]      # macro from month 1 to T_train-1
```

This means $\Delta$ natively learns the mapping $f_t \sim m_{t-1}$, so at prediction time:

$$
\hat{E}_t[f_{t+1}] = \Delta' m_t
$$

uses only macro information available at time $t$.

**Out-of-sample lagged macro**: `test_macro_lag[t]` = macro from the previous month. For the first test month, this is the last training-period macro observation.

**Impact on sample size**: Lagging costs one observation. GIPCA trains on $T_\text{train} - 1$ months (479 vs 480 for IPCA).

## 5. Estimators

Four models are compared in each experiment:

| Model | Estimator | Predictive signal | Parameters |
|-------|-----------|-------------------|------------|
| IPCA | ALS | $\Gamma' z_{i,t} \cdot \hat{\lambda}$ (constant) | $L \times K$ |
| IPCA | Grassmannian CG | $\Gamma' z_{i,t} \cdot \hat{\lambda}$ (constant) | $L \times K$ |
| GIPCA | ALS | $\Gamma' z_{i,t} \cdot \Delta' m_t$ (time-varying) | $L \times K + K \times R$ |
| GIPCA | Grassmannian CG | $\Gamma' z_{i,t} \cdot \Delta' m_t$ (time-varying) | $L \times K + K \times R$ |

### ALS GIPCA (`src/models/als_gipca.py`)

Alternates three steps, each decreasing the joint penalised objective:

1. **f-step**: $f_t = (\Lambda_t'\Lambda_t + \alpha I)^{-1} (\Lambda_t' r_t + \alpha \Delta' m_t)$ where $\Lambda_t = Z_t \Gamma$
2. **$\Gamma$-step**: Kronecker normal equations, then SVD orthonormalization
3. **$\Delta$-step**: Time-series OLS regression of $f_t$ on $m_t$

After convergence, enforces $f^0_t \perp m_t$ (verified via $\| f_0^T m \| \approx 0$).

Settings: `max_iter=500`, `min_iter=100`, `tol=1e-6`, `alpha=0.1`

### Grassmannian GIPCA (`src/models/grassmanian_gipca.py`)

Riemannian Conjugate Gradient on `Grassmann(L, K)` via pymanopt, with $\Delta$ and $f^0_t$ recovered in post-estimation.

Settings: `optimizer=ConjugateGradient`, `max_iterations=300`

### Number of factors

$K = 4$ in all experiments (following Kelly, Pruitt & Su).

## 6. Train / Test Split

| Set | Period | Months |
|-----|--------|--------|
| Train (in-sample) | 1965/01 -- 2004/12 | 480 (479 for GIPCA due to lagging) |
| Test (out-of-sample) | 2005/01 -- 2016/12 | 144 |

## 7. Evaluation Metrics

### Total $R^2$

Uses realized factor estimates $\hat{f}_t$ (same as IPCA experiment):

$$
R^2_\text{total} = 1 - \frac{\sum_{i,t} (x_{i,t} - \hat{\beta}_{i,t}' \hat{f}_t)^2}{\sum_{i,t} x_{i,t}^2}
$$

Out-of-sample factors re-estimated via cross-sectional OLS with $\Gamma$ fixed from training.

### Predictive $R^2$

Different predictive signals for IPCA vs GIPCA:

- **IPCA**: $\hat{x}_{i,t+1} = z_{i,t}' \Gamma \hat{\lambda}$, where $\hat{\lambda} = \bar{f}$ (constant)
- **GIPCA**: $\hat{x}_{i,t+1} = z_{i,t}' \Gamma (\Delta' m_t)$ (time-varying, lagged macro)

$$
R^2_\text{pred} = 1 - \frac{\sum_{i,t} (x_{i,t} - \hat{x}_{i,t})^2}{\sum_{i,t} x_{i,t}^2}
$$

### Portfolio sorts (out-of-sample)

Same methodology as the IPCA experiment, but with different predictive signals:

- **IPCA**: Sort by $z_{i,t}' \Gamma \hat{\lambda}$
- **GIPCA**: Sort by $z_{i,t}' \Gamma (\Delta' m_t)$ using lagged macro

Equal-weighted quintile portfolios. Long-short = Q5 $-$ Q1.

## 8. Results

### Experiment 1: Fama-French 5 Factors (Top 500 stocks)

```
                          ALS IPCA  Grass IPCA  ALS GIPCA  Grass GIPCA
Final objective             1.3003      1.3004     1.3099       1.3026
IS Total R² (%)              35.51       35.51      35.49        35.50
IS Predictive R² (%)          1.63        1.63       0.51         0.50
OOS Total R² (%)             35.71       35.71      35.70        35.71
OOS Predictive R² (%)         0.80        0.80       0.10         0.12
L/S Sharpe (OOS)              0.38        0.37       0.41         0.43
Parameters                     380         380        400          400
```

### Experiment 2: Welch & Goyal Predictors (Top 3000 stocks)

```
                          ALS IPCA  Grass IPCA  ALS GIPCA  Grass GIPCA
Final objective             8.8424      8.8425     8.8630       8.8579
IS Total R² (%)              26.46       26.46      26.46        26.46
IS Predictive R² (%)          0.87        0.87       0.82         0.82
OOS Total R² (%)             27.00       27.01      27.00        27.01
OOS Predictive R² (%)         0.52        0.52       0.31         0.31
L/S Sharpe (OOS)             -0.41       -0.41       0.08         0.08
Parameters                     380         380        408          408
```

### Key observations

1. **Total $R^2$** is nearly identical across all four models in both experiments (~35.5% for FF5, ~26.5% for WG). The factor structure is dominated by the characteristics loading $\Gamma$, which all models learn similarly.

2. **Predictive $R^2$** is lower for GIPCA than IPCA in both experiments. This is because the mean factor $\hat{\lambda}$ (IPCA) captures the unconditional average, while GIPCA's time-varying $\Delta' m_t$ introduces additional noise when the macro-factor relationship is weak.

3. **Portfolio L/S Sharpe ratios** tell a different story. Despite lower predictive $R^2$:
   - **FF5**: GIPCA improves the L/S Sharpe from 0.37--0.38 (IPCA) to 0.41--0.43 (GIPCA), with L/S return rising from ~5.3% to ~7.2%.
   - **WG**: IPCA produces a *negative* L/S Sharpe (-0.41), meaning the quintile sort is inverted. GIPCA corrects this to a small positive Sharpe (0.08), suggesting the macro conditioning rescues the directional signal.

4. **ALS vs Grassmannian**: Both optimizers converge to essentially the same solution within each model class. The Grassmannian optimizer is faster in wall time for both IPCA and GIPCA.

## 9. Notebooks

| Notebook | Macro set | Universe |
|----------|-----------|----------|
| `notebooks/ipca_vs_gipca_ff5_comparison.ipynb` | Fama-French 5 | Top 500 |
| `notebooks/ipca_vs_gipca_wg_comparison.ipynb` | Welch & Goyal | Top 3000 |

## 10. References

- Kelly, B., Pruitt, S., & Su, Y. (2019). *Characteristics are covariances: A unified model of risk and return.* Journal of Financial Economics, 134(3), 611--632.
- Kelly, B., Pruitt, S., & Su, Y. (2020). *Instrumented principal component analysis.* Working paper.
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical asset pricing via machine learning.* Review of Financial Studies, 33(5), 2223--2273.
- Fama, E. F., & French, K. R. (2015). *A five-factor asset pricing model.* Journal of Financial Economics, 116(1), 1--22.
- Welch, I., & Goyal, A. (2008). *A comprehensive look at the empirical performance of equity premium prediction.* Review of Financial Studies, 21(4), 1455--1508.
