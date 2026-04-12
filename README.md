# Generalized Instrumented Principal Component Analysis (GIPCA)

A Python implementation of **Generalized Instrumented Principal Component Analysis (GIPCA)** — an extension of IPCA (Kelly, Pruitt & Su, 2019, 2020) that incorporates macroeconomic variables to partially explain latent factors in asset pricing models.

## Authors

- Oualid Missaoui
- Andrew Lesniewski

## Model

Standard IPCA links asset returns to latent factors via characteristic-dependent loadings. GIPCA adds a third equation that decomposes factors into a macro-explained component and a residual:

```
r_t = Z_t Gamma f_t + eps_t          (returns)
f_t = f0_t + Delta' m_t              (factor decomposition)
f0_t _|_ m_t                         (identification)
```

The penalized objective balances return fit against macro alignment:

```
min_{Gamma, Delta}  (1/T) sum_t min_{f_t} [ ||r_t - Z_t Gamma f_t||^2 + alpha ||f_t - Delta' m_t||^2 ]
```

Where:
- **Gamma** in R^{L x K} — maps L firm characteristics to K factor loadings
- **Delta** in R^{K x R} — maps R macro variables to K factor returns
- **alpha >= 0** — penalty weight controlling macro influence on estimation
  - alpha = 0: pure IPCA (Delta is post-hoc OLS, macro has no effect on Gamma)
  - alpha > 0: soft GIPCA (penalizes latent component, macro influences Gamma through factors)
  - alpha -> inf: hard GIPCA (forces f_t ~ Delta' m_t)

### Predictive Setup

For out-of-sample prediction, we use **lagged macro variables**: align `returns[1:]` with `macro[:-1]` so that Delta natively regresses f_t on m_{t-1}. This avoids look-ahead bias and enables genuine macro-conditioned forecasting.

## Project Structure

```
GIPCA/
├── src/
│   ├── __init__.py
│   ├── utils.py                              # Subspace error & recovery metrics
│   ├── generate/
│   │   ├── gipca.py                          # Synthetic GIPCA data generation
│   │   └── ipca.py                           # Synthetic IPCA data generation
│   ├── models/
│   │   ├── als_gipca.py                      # ALS estimator for GIPCA
│   │   ├── als_ipca.py                       # ALS estimator for IPCA
│   │   ├── grassmanian_gipca.py              # Riemannian manifold GIPCA (pymanopt)
│   │   └── grassmanian_ipca.py               # Riemannian manifold IPCA (pymanopt)
│   └── _deprecated/                          # Legacy implementations
├── notebooks/
│   ├── gipca_illustration.ipynb              # GIPCA methodology walkthrough
│   ├── ipca_illustration.ipynb               # IPCA methodology walkthrough
│   ├── ipca_asset_pricing.ipynb              # Empirical IPCA application
│   ├── ipca_vs_gipca_synthetic.ipynb         # Synthetic data comparison
│   ├── ipca_vs_gipca_ff5_comparison.ipynb    # Fama-French 5 factors experiment
│   └── ipca_vs_gipca_wg_comparison.ipynb     # Welch-Goyal predictors experiment
├── docs/
│   ├── gipca_asset_pricing_experiment.md     # GIPCA experiment details
│   ├── ipca_asset_pricing_experiment.md      # IPCA experiment details
│   ├── gipca_experimental_results.pptx       # Results presentation
│   └── 2020_Instrumented...pdf               # Reference paper
├── data/
│   ├── crsp_monthly_returns.csv              # CRSP stock returns (1965-2018)
│   ├── datashare.csv                         # 94 firm characteristics
│   ├── F-F_Research_Data_5_Factors_2x3_daily.csv  # Fama-French 5 factors (daily)
│   ├── F-F_Research_Data_Factors.csv         # Fama-French factors
│   └── macro/
│       └── PredictorData2021 - Monthly.csv   # Welch-Goyal macro predictors
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy`, `pandas`, `scipy`, `scikit-learn` — numerical computing
- `matplotlib`, `seaborn` — visualization
- `pymanopt` — Riemannian optimization on Grassmann manifolds
- `autograd` — automatic differentiation for manifold costs

## Usage

### IPCA (Standard)

```python
from src.generate.ipca import generate_ipca_data
from src.models.als_ipca import ALSIPCA

# Generate synthetic panel data
data, truth = generate_ipca_data(T=252, N=500, m=20, k=5)
rets, Z = data

# Fit IPCA via ALS
model = ALSIPCA(num_assets=500, num_fact=5, num_charact=20, win_len=252)
Gamma, obj_history = model.fit(data, max_iter=5000, verbose=True)

# Predict using mean factor returns
results = model.get_results()
```

### GIPCA (Generalized)

```python
from src.generate.gipca import generate_gipca_data
from src.models.als_gipca import HardGIPCA

# Generate synthetic panel data with macro variables
data, truth = generate_gipca_data(T=252, N=500, m=20, k=5, num_macro=3)
rets, Z, M = data

# For predictive use: pass lagged macro
rets_lag, Z_lag, M_lag = rets[1:], Z[1:], M[:-1]

# Fit GIPCA via ALS with alpha penalty
model = HardGIPCA(num_assets=500, num_fact=5, num_charact=20,
                  num_macro=3, win_len=251, alpha=0.1)
Gamma, obj_history = model.fit([rets_lag, Z_lag, M_lag],
                                max_iter=500, verbose=True)

# Access estimated parameters
results = model.get_results()
Delta = model.Delta       # K x R macro loading matrix
f0 = model.f0             # K x T residual factors
r2 = model.score([rets_test, Z_test, M_test])
```

### Grassmannian Estimators

```python
from src.models.grassmanian_ipca import GrassmannManifoldIPCAEstimator
from src.models.grassmanian_gipca import GrassmannManifoldGIPCAEstimator

# IPCA on the Grassmann manifold
ipca = GrassmannManifoldIPCAEstimator(
    num_assets=500, num_fact=5, num_charact=20, win_len=252
)
Wopt, f_hat, history = ipca.fit(data, optimizer="ConjugateGradient", max_iterations=200)

# GIPCA on the Grassmann manifold (profiled IPCA loss + post-hoc Delta)
gipca = GrassmannManifoldGIPCAEstimator(
    num_assets=500, num_fact=5, num_charact=20, num_macro=3, win_len=251
)
Wopt, Delta_hat, f0_hat, f_hat, history = gipca.fit(
    [rets_lag, Z_lag, M_lag], optimizer="ConjugateGradient", max_iterations=200
)
```

## Estimation Methods

### Alternating Least Squares (ALS)

Three-step iteration:

1. **f-step** — update factors given Gamma and Delta:
   - alpha = 0: `f_t = (Lambda_t' Lambda_t)^{-1} Lambda_t' r_t` (OLS)
   - alpha > 0: `f_t = (Lambda_t' Lambda_t + alpha I)^{-1} (Lambda_t' r_t + alpha Delta' m_t)` (ridge toward macro prediction)
2. **Gamma-step** — update characteristic map via Kronecker normal equations + SVD orthonormalization + Procrustes alignment
3. **Delta-step** — time-series OLS of factors on macro variables: `Delta = (F M) (M' M)^{-1}`

Handles missing data, uses Procrustes alignment to prevent rotation cycling, and guarantees monotone convergence of the penalized objective.

### Grassmannian Manifold Optimization

Optimizes Gamma directly on the Grassmann manifold Gr(L, K) using pymanopt (conjugate gradient, steepest descent, or trust regions). Uses the profiled IPCA loss with automatic differentiation via autograd. Delta and residual factors are computed post-hoc via time-series OLS.

## Empirical Results

Evaluated on CRSP monthly returns (1965-2018) with 94 firm characteristics. Train: 1965-2004, Test: 2005-2018.

### Fama-French 5 Factors (Top 500 stocks, K=5, alpha=0.1)

Macro variables: Mkt-RF, SMB, HML, RMW, CMA (daily compounded to monthly).

|                        | ALS IPCA | Grass IPCA | ALS GIPCA | Grass GIPCA |
|------------------------|----------|------------|-----------|-------------|
| IS Total R2            | 35.51%   | 35.51%     | 35.49%    | 35.50%      |
| IS Predictive R2       | 1.63%    | 1.63%      | 0.51%     | 0.50%       |
| OOS Total R2           | 35.71%   | 35.71%     | 35.70%    | 35.71%      |
| OOS Predictive R2      | 0.80%    | 0.80%      | 0.10%     | 0.12%       |
| L/S Sharpe (OOS)       | 0.38     | 0.37       | **0.41**  | **0.43**    |

GIPCA improves portfolio Sharpe ratios (0.41-0.43 vs 0.37-0.38) despite lower predictive R2, suggesting macro conditioning produces more investable factor signals.

### Welch-Goyal Predictors (Top 3000 stocks, K=4, alpha=0.1)

Macro variables: term spread, default yield spread, stock variance, inflation, market return, long-term return, cross-sectional dispersion (7 variables, lagged).

|                        | ALS IPCA | Grass IPCA | ALS GIPCA | Grass GIPCA |
|------------------------|----------|------------|-----------|-------------|
| IS Total R2            | 26.46%   | 26.46%     | 26.46%    | 26.46%      |
| OOS Total R2           | 27.00%   | 27.01%     | 27.00%    | 27.01%      |
| L/S Sharpe (OOS)       | -0.41    | -0.41      | **0.08**  | **0.08**    |

GIPCA rescues a broken IPCA signal (negative Sharpe) to positive using macro conditioning.

## GIPCA vs Standard IPCA

| Feature         | Standard IPCA              | GIPCA                           |
|-----------------|----------------------------|---------------------------------|
| Factors         | Fully latent               | Partially macro-driven          |
| Equations       | 2 (returns, loadings)      | 3 (+ factor decomposition)      |
| Parameters      | Gamma                      | Gamma, Delta                    |
| Penalty         | None                       | alpha on latent component       |
| Prediction      | Unconditional mean factor  | Macro-conditioned factors       |
| Time-varying    | No                         | Yes (via lagged macro)          |
| Interpretation  | Statistical                | Economic + Statistical          |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `gipca_illustration.ipynb` | Visual walkthrough of GIPCA methodology on synthetic data |
| `ipca_illustration.ipynb` | Visual walkthrough of IPCA methodology on synthetic data |
| `ipca_asset_pricing.ipynb` | Full empirical IPCA application on CRSP data |
| `ipca_vs_gipca_synthetic.ipynb` | IPCA vs GIPCA comparison on synthetic data |
| `ipca_vs_gipca_ff5_comparison.ipynb` | FF5 factors as macro variables (N=500, K=5) |
| `ipca_vs_gipca_wg_comparison.ipynb` | Welch-Goyal predictors as macro variables (N=3000, K=4) |

Launch all notebooks:
```bash
./run.command
```

## References

- Kelly, B. T., Pruitt, S., & Su, Y. (2019). "Characteristics Are Covariances: A Unified Model of Risk and Return." *Journal of Financial Economics*.
- Kelly, B. T., Pruitt, S., & Su, Y. (2020). "Instrumented Principal Component Analysis." *SSRN Working Paper*.
- Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*.
- Welch, I., & Goyal, A. (2008). "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction." *Review of Financial Studies*.
