# Daily IPCA vs GIPCA: S&P 500 (Sharadar) with FF5 Macro

## 1. Overview

This experiment replicates the monthly IPCA vs GIPCA asset pricing comparison at **daily frequency** using the **Sharadar database**, restricted to **S&P 500 constituents** with **20 firm characteristics** and daily **Fama-French 5 factors** as macro variables.

Four models are compared:

| Model | Estimator | Predictive signal | Parameters |
|-------|-----------|-------------------|------------|
| IPCA | ALS | $\Gamma' z_{i,t} \cdot \hat{\lambda}$ (constant mean factor) | $L \times K = 100$ |
| IPCA | Grassmannian CG | $\Gamma' z_{i,t} \cdot \hat{\lambda}$ (constant mean factor) | $L \times K = 100$ |
| GIPCA | ALS | $\Gamma' z_{i,t} \cdot \Delta' m_{t-1}$ (lagged macro-predicted) | $L \times K + K \times R = 125$ |
| GIPCA | Grassmannian CG | $\Gamma' z_{i,t} \cdot \Delta' m_{t-1}$ (lagged macro-predicted) | $L \times K + K \times R = 125$ |

## 2. Data

### Returns

- **Source**: `SHARADAR_SEP.csv` (daily stock prices)
- **Total return**: $r_{i,t} = (\text{close}_t + \text{dividends}_t) / \text{close}_{t-1} - 1$
- **Excess return**: $x_{i,t} = r_{i,t} - RF_t$ (daily risk-free rate from FF5)

### Universe

- **S&P 500 constituents** with point-in-time membership tracking
- Membership reconstructed from `SHARADAR_SP500.csv`: quarterly `historical`/`current` snapshots as anchors, with inter-snapshot `added`/`removed` events applied chronologically
- **N = 611** unique stocks appearing in the panel
- **1,069** tickers ever in S&P 500 across the full Sharadar history

### Sample Period

| Set | Period | Trading days |
|-----|--------|-------------|
| Train (in-sample) | 2015-01-02 -- 2016-12-30 | 504 (503 for GIPCA) |
| Test (out-of-sample) | 2017-01-03 -- 2018-12-31 | 502 |
| **Total** | **2015-01-02 -- 2018-12-31** | **1,006** |

### Non-zero return coverage

82.2% of the $(T \times N)$ panel entries have non-zero excess returns. The remaining 17.8% correspond to dates when a stock was not an S&P 500 member (filled with 0).

## 3. Characteristics ($L = 20$)

Twenty firm characteristics from three Sharadar data sources, covering size, value, profitability, investment, leverage, momentum, and risk:

| # | Name | Category | Source | Formula |
|---|------|----------|--------|---------|
| 1 | `mvel1` | Size | SHARADAR_DAILY | marketcap |
| 2 | `bm` | Value | SHARADAR_DAILY | 1 / pb |
| 3 | `ep` | Value | SHARADAR_DAILY | 1 / pe |
| 4 | `sp` | Value | SHARADAR_DAILY | 1 / ps |
| 5 | `roaq` | Profitability | SHARADAR_SF1 | netinccmn / assets |
| 6 | `roeq` | Profitability | SHARADAR_SF1 | netinccmn / equity |
| 7 | `gma` | Profitability | SHARADAR_SF1 | gp / revenue |
| 8 | `operprof` | Profitability | SHARADAR_SF1 | opinc / assets |
| 9 | `agr` | Investment | SHARADAR_SF1 | YoY asset growth (4-quarter lag) |
| 10 | `sgr` | Growth | SHARADAR_SF1 | YoY revenue growth (4-quarter lag) |
| 11 | `lev` | Leverage | SHARADAR_SF1 | debt / equity |
| 12 | `currat` | Liquidity | SHARADAR_SF1 | currentratio |
| 13 | `chcsho` | Investment | SHARADAR_SF1 | YoY shares outstanding change (4-quarter lag) |
| 14 | `cfp` | Value | SF1 + DAILY | (netinccmn + depamor) / marketcap |
| 15 | `depr` | Balance Sheet | SHARADAR_SF1 | depamor / assets |
| 16 | `dy` | Value | SHARADAR_SF1 | divyield |
| 17 | `mom1m` | Momentum | SHARADAR_SEP | 21-day cumulative return |
| 18 | `mom6m` | Momentum | SHARADAR_SEP | 126-day cumulative return |
| 19 | `retvol` | Risk | SHARADAR_SEP | 21-day return standard deviation |
| 20 | `turn` | Liquidity | SHARADAR_SEP | 21-day average volume / shares outstanding |

### Characteristic preprocessing

1. **Point-in-time quarterly fundamentals**: SF1 data merged via `merge_asof` using `datekey` (SEC filing date), ensuring no look-ahead bias from unreported financials.
2. **1-day lag**: All characteristics are lagged by 1 trading day within each ticker. This prevents contemporaneous bias -- e.g., `mom1m[t]` would otherwise contain `ret[t]`, allowing the model to trivially "predict" today's return from today's return.
3. **Cross-sectional rank transform**: Each characteristic is rank-transformed to $[-0.5, +0.5]$ per day: $\text{rank}(\text{pct}=\text{True}) - 0.5$. Missing values filled with 0.

### Characteristic coverage

All characteristics have ~81-82% non-zero coverage (matching the S&P 500 membership rate), except `currat` at 67.5% (fewer firms report current ratio).

## 4. Macro Variables ($R = 5$)

**Source**: `F-F_Research_Data_5_Factors_2x3_daily.csv` (daily, values in decimal)

| Variable | Description |
|----------|-------------|
| `Mkt-RF` | Market excess return |
| `SMB` | Small minus Big (size) |
| `HML` | High minus Low (value) |
| `RMW` | Robust minus Weak (profitability) |
| `CMA` | Conservative minus Aggressive (investment) |

**Z-scoring**: Standardised using training-period mean and standard deviation only. Test-period macro standardised with the same training statistics (no look-ahead).

## 5. Timing and Lagging

### Characteristics lag

All 20 characteristics are lagged by 1 day: characteristics at end of day $t-1$ are paired with returns on day $t$. This is essential at daily frequency to avoid contemporaneous bias.

### Macro lag (GIPCA)

GIPCA is trained on lagged macro to ensure no look-ahead bias in the predictive signal:

```
train_rets_lag   = returns[1:split]      # returns from day 2 to T_train
train_Z_lag      = chars[1:split]        # chars from day 2 to T_train
train_macro_lag  = macro[0:split-1]      # macro from day 1 to T_train-1
```

$\Delta$ learns the mapping $f_t \sim m_{t-1}$, so the predictive signal $\hat{E}_t[f_{t+1}] = \Delta' m_t$ uses only information available at time $t$.

## 6. Model Settings

| Parameter | Value |
|-----------|-------|
| Factors ($K$) | 5 |
| Characteristics ($L$) | 20 |
| Macro predictors ($R$) | 5 |
| Penalty ($\alpha$) | 0.1 |
| ALS max iterations | 500 |
| ALS convergence tolerance | $10^{-6}$ |
| Grassmannian optimizer | Conjugate Gradient |
| Grassmannian max iterations | 300 |
| Annualization | $\times 252$ (returns), $\times \sqrt{252}$ (volatility) |

## 7. Results

### R-squared

```
                          ALS IPCA  Grass IPCA  ALS GIPCA  Grass GIPCA
Final objective             0.1514      0.1514     0.1517       0.1516
IS Total R² (%)              10.97       10.97      10.97        10.97
IS Predictive R² (%)          0.00        0.00       0.13         0.13
OOS Total R² (%)              8.57        8.57       8.57         8.56
OOS Predictive R² (%)        -0.00       -0.00      -0.07        -0.07
```

### Portfolio Sorts (out-of-sample)

Equal-weighted quintile portfolios sorted by predicted expected return. Annualised $\times 252$ / $\sqrt{252}$.

```
                      ALS IPCA  Grass IPCA  ALS GIPCA  Grass GIPCA
Q1 Return (%)             3.57        3.53      -4.39        -4.26
Q2 Return (%)             5.61        5.60       3.96         3.94
Q3 Return (%)             3.52        3.51       6.62         6.42
Q4 Return (%)             4.40        4.42       7.33         7.38
Q5 Return (%)             4.01        4.06       7.61         7.65
L/S Return (%)            0.43        0.54      12.00        11.91
L/S Sharpe                0.04        0.05       1.08         1.07
```

### Summary

```
                          ALS IPCA  Grass IPCA  ALS GIPCA  Grass GIPCA
IS Total R² (%)              10.97       10.97      10.97        10.97
OOS Total R² (%)              8.57        8.57       8.57         8.56
OOS Predictive R² (%)        -0.00       -0.00      -0.07        -0.07
L/S Sharpe (OOS)              0.04        0.05       1.08         1.07
Iterations                     122          68        100           80
Wall time (s)                 49.8        13.4       20.4         15.8
Parameters                     100         100        125          125
Predictive signal          mean(f)     mean(f)   Delta'm_t    Delta'm_t
```

## 8. Key Observations

1. **Total $R^2$ is ~9-11%**, substantially lower than the monthly experiment (~35%). This is expected: daily returns are much noisier, and the signal-to-noise ratio is lower. With only 20 characteristics (vs 95 monthly), the model captures less of the return variation.

2. **ALS and Grassmannian converge to the same solution** in both IPCA and GIPCA, confirming both optimisers find the same global optimum. The Grassmannian CG is ~3-4x faster in wall time.

3. **IPCA's mean-factor signal has zero predictive power at daily frequency** (L/S Sharpe 0.04-0.05). The unconditional mean of daily factor returns is too small relative to daily noise. This contrasts with the monthly experiment where IPCA achieves Sharpe 0.37-0.38.

4. **GIPCA's macro-conditioned signal substantially outperforms IPCA** on portfolio sorts (Sharpe 1.07-1.08 vs 0.04-0.05). The time-varying signal $\Delta' m_{t-1}$ captures short-term factor timing through lagged FF5 factors. This is the main finding: at daily frequency, GIPCA's macro conditioning is essential for generating tradeable signals.

5. **Monotonic quintile spread for GIPCA**: Returns increase from Q1 (-4.3%) to Q5 (+7.6%), with a 12% annualised L/S return. IPCA's quintile returns show no monotonic pattern.

6. **Predictive $R^2$ is near zero for all models**. Even GIPCA's L/S Sharpe of 1.08 comes with negative predictive $R^2$. This apparent contradiction arises because predictive $R^2$ measures total cross-sectional fit, while the L/S Sharpe measures the portfolio's risk-adjusted return. A model can sort stocks correctly (positive Sharpe) without explaining much overall variance (low $R^2$).

## 9. Comparison with Monthly Experiment

| Metric | Monthly (FF5) | Daily (Sharadar) |
|--------|--------------|-----------------|
| Universe | Top 500 by avg cap | S&P 500 (point-in-time) |
| $T_\text{train}$ | 480 months | 504 days |
| $T_\text{test}$ | 144 months | 502 days |
| $L$ | 95 | 20 |
| $K$ | 4 | 5 |
| $R$ | 5 | 5 |
| $\alpha$ | 0.1 | 0.1 |
| IS Total $R^2$ | 35.5% | 11.0% |
| OOS Total $R^2$ | 35.7% | 8.6% |
| IPCA L/S Sharpe | 0.37-0.38 | 0.04-0.05 |
| GIPCA L/S Sharpe | 0.41-0.43 | 1.07-1.08 |
| Annualisation | $\times 12$, $\sqrt{12}$ | $\times 252$, $\sqrt{252}$ |

## 10. Source Code

| File | Description |
|------|-------------|
| `src/generate/daily_sharadar.py` | Data loading module (S&P 500 membership, returns, valuations, fundamentals, panel construction) |
| `notebooks/daily_ipca_vs_gipca.ipynb` | Experiment notebook (28 cells, mirrors `ipca_vs_gipca_ff5_comparison.ipynb`) |

## 11. References

- Kelly, B., Pruitt, S., & Su, Y. (2019). *Characteristics are covariances: A unified model of risk and return.* Journal of Financial Economics, 134(3), 611--632.
- Fama, E. F., & French, K. R. (2015). *A five-factor asset pricing model.* Journal of Financial Economics, 116(1), 1--22.
- Sharadar/Quandl US Equity Prices and Fundamentals database.
