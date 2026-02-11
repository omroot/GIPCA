# Synthetic Data Generation Methodology

This document describes the methodology for generating synthetic data to validate the GIPCA (Generalized Instrumented Principal Component Analysis) model. The synthetic environment provides a controlled setting where the true data generating process (DGP) is known, enabling rigorous evaluation of parameter recovery and model performance.

## 1 Overview

The synthetic data is generated according to the exact GIPCA model specification:

$$r_{i,t} = z_{i,t}' \Gamma f_t + \epsilon_{i,t}$$
$$f_t = \Lambda' m_t + \nu_t$$

This allows us to:
1. Assess whether GIPCA can recover the true parameters ($\Gamma$, $\Lambda$)
2. Compare GIPCA against IPCA and PCA under controlled conditions
3. Understand model behavior across different noise regimes

## 2 Dimensions

| Symbol | Value | Description |
|--------|-------|-------------|
| $T$ | 200 | Number of time periods |
| $N$ | 100 | Number of assets |
| $L$ | 20 | Number of characteristics |
| $K$ | 4 | Number of latent factors |
| $R$ | 8 | Number of macro variables |

These dimensions are chosen to be:
- Large enough to provide meaningful statistical estimates
- Small enough for fast computation during sensitivity analysis
- Representative of typical panel data structures in finance

## 3 Parameter Generation

### 3.1 Characteristic-Loading Matrix ($\Gamma$)

The true $\Gamma \in \mathbb{R}^{L \times K}$ maps characteristics to factor loadings.

**Generation procedure:**
```python
# Initialize with random Gaussian entries
true_Gamma = np.random.randn(L, K)

# Orthonormalize via QR decomposition
true_Gamma, _ = np.linalg.qr(true_Gamma)
true_Gamma = true_Gamma[:, :K]
```

**Rationale:** Orthonormalization ensures:
- Factors are identified (no rotational indeterminacy within the column space)
- Columns have unit norm, providing a natural scale
- The Grassmann distance metric is well-defined for comparing estimated vs. true $\Gamma$

### 3.2 Macro-Factor Matrix ($\Lambda$)

The true $\Lambda \in \mathbb{R}^{R \times K}$ maps macro variables to factor realizations.

**Generation procedure:**
```python
true_Lambda = np.random.randn(R, K) * 0.5
```

**Rationale:**
- Random Gaussian entries provide diverse macro-factor relationships
- Scale factor of 0.5 ensures factors are not dominated by macro (some residual variation $\nu_t$ remains)

## 4 Time Series Generation

### 4.1 Macro Variables ($m_t$)

Macro variables follow a stationary AR(1) process with high persistence:

$$m_t = \rho \cdot m_{t-1} + \sqrt{1 - \rho^2} \cdot \xi_t, \quad \xi_t \sim N(0, I_R)$$

**Parameters:**
- Persistence: $\rho = 0.9$
- Initial condition: $m_0 \sim N(0, I_R)$

**Post-processing:** Standardize to zero mean and unit variance:
```python
M = (M - M.mean(axis=0)) / M.std(axis=0)
```

**Rationale:**
- High persistence ($\rho = 0.9$) mimics real macro variables (GDP growth, inflation, etc.)
- The scaling $\sqrt{1 - \rho^2}$ ensures the unconditional variance is 1
- Standardization matches the preprocessing applied to real Welch-Goyal data

### 4.2 Latent Factors ($f_t$)

Factors are generated as macro-driven plus noise:

$$f_t = \Lambda' m_t + \nu_t, \quad \nu_t \sim N(0, \sigma_\nu^2 I_K)$$

**Parameters:**
- Factor noise: $\sigma_\nu = 0.3$

**Implementation:**
```python
true_factors = M @ true_Lambda + np.random.randn(T, K) * sigma_nu
```

**Rationale:**
- The deterministic component $\Lambda' m_t$ makes factors predictable from macro
- The noise $\nu_t$ represents factor variation not explained by macro
- With $\sigma_\nu = 0.3$, approximately 70-90% of factor variation is macro-driven (depending on $\Lambda$)

### 4.3 Characteristics ($z_{i,t}$)

Characteristics follow a panel AR(1) process:

$$z_{i,t} = 0.8 \cdot z_{i,t-1} + 0.2 \cdot \eta_{i,t}, \quad \eta_{i,t} \sim N(0, 0.25 \cdot I_L)$$

**Special case:** The first characteristic is set to 1.0 (intercept):
```python
Z[:, :, 0] = 1.0
```

**Implementation:**
```python
Z = np.zeros((T, N, L))
Z[0] = np.random.randn(N, L) * 0.5

for t in range(1, T):
    Z[t] = 0.8 * Z[t-1] + 0.2 * np.random.randn(N, L) * 0.5

Z[:, :, 0] = 1.0  # Intercept
```

**Rationale:**
- Persistence (0.8) mimics real firm characteristics that evolve slowly (size, book-to-market, etc.)
- Cross-sectional variation provides identifying information for $\Gamma$
- The intercept allows for a market factor (constant loading across all assets)

### 4.4 Returns ($r_{i,t}$)

Returns are generated according to the factor model:

$$r_{i,t} = z_{i,t}' \Gamma f_t + \epsilon_{i,t}, \quad \epsilon_{i,t} \sim N(0, \sigma_\epsilon^2)$$

**Parameters:**
- Idiosyncratic noise: $\sigma_\epsilon = 0.05$

**Implementation:**
```python
returns = np.zeros((T, N))
for t in range(T):
    loadings_t = Z[t] @ true_Gamma  # N x K
    returns[t] = loadings_t @ true_factors[t] + np.random.randn(N) * sigma_eps
```

**Rationale:**
- Low idiosyncratic noise ($\sigma_\epsilon = 0.05$) provides high signal-to-noise ratio
- This is intentionally optimistic; sensitivity analysis varies $\sigma_\epsilon$ to test robustness

## 5 Signal-to-Noise Analysis

### 5.1 Variance Decomposition

The total return variance can be decomposed as:

$$\text{Var}(r_{i,t}) = \underbrace{\text{Var}(z_{i,t}' \Gamma f_t)}_{\text{Systematic}} + \underbrace{\sigma_\epsilon^2}_{\text{Idiosyncratic}}$$

With our parameters:
- Systematic variance: ~0.26 (from returns std ≈ 0.52)
- Idiosyncratic variance: 0.0025 ($\sigma_\epsilon^2 = 0.05^2$)
- **Signal-to-noise ratio**: ~100:1

This high SNR allows near-perfect recovery in the baseline case.

### 5.2 Factor Predictability

The factor $R^2$ from macro regression:

$$R^2_{\text{macro}} = 1 - \frac{\sigma_\nu^2}{\text{Var}(f_t)}$$

With $\sigma_\nu = 0.3$ and $\text{Var}(\Lambda' m_t) \approx 0.25$:
- Factor variance: ~0.34
- Noise variance: 0.09
- **Macro $R^2$**: ~70-90% per factor

## 6 Train/Test Split

| Period | Time Range | Observations |
|--------|------------|--------------|
| Training | $t = 1, \ldots, 140$ | 70% of sample |
| Testing | $t = 141, \ldots, 200$ | 30% of sample |

**Implementation:**
```python
split_idx = int(0.7 * T)  # = 140

train_returns = returns[:split_idx]
test_returns = returns[split_idx:]
# Similarly for Z and M
```

**Rationale:**
- 70/30 split is standard in machine learning
- 140 training periods provide sufficient data for stable estimation
- 60 test periods allow meaningful out-of-sample evaluation

## 7 Complete Generation Algorithm

```python
import numpy as np

def generate_synthetic_data(T=200, N=100, L=20, K=4, R=8,
                            sigma_eps=0.05, sigma_nu=0.3,
                            rho=0.9, seed=42):
    """
    Generate synthetic data for GIPCA validation.

    Returns
    -------
    returns : ndarray (T, N)
        Asset returns
    Z : ndarray (T, N, L)
        Firm characteristics
    M : ndarray (T, R)
        Macro variables
    true_Gamma : ndarray (L, K)
        True characteristic-loading map
    true_Lambda : ndarray (R, K)
        True macro-factor map
    true_factors : ndarray (T, K)
        True latent factors
    """
    np.random.seed(seed)

    # 1. Generate true parameters
    true_Gamma = np.random.randn(L, K)
    true_Gamma, _ = np.linalg.qr(true_Gamma)
    true_Gamma = true_Gamma[:, :K]

    true_Lambda = np.random.randn(R, K) * 0.5

    # 2. Generate macro variables (AR(1))
    M = np.zeros((T, R))
    M[0] = np.random.randn(R)
    for t in range(1, T):
        M[t] = rho * M[t-1] + np.sqrt(1 - rho**2) * np.random.randn(R)
    M = (M - M.mean(axis=0)) / M.std(axis=0)

    # 3. Generate factors
    true_factors = M @ true_Lambda + np.random.randn(T, K) * sigma_nu

    # 4. Generate characteristics (AR(1) panel)
    Z = np.zeros((T, N, L))
    Z[0] = np.random.randn(N, L) * 0.5
    for t in range(1, T):
        Z[t] = 0.8 * Z[t-1] + 0.2 * np.random.randn(N, L) * 0.5
    Z[:, :, 0] = 1.0  # Intercept

    # 5. Generate returns
    returns = np.zeros((T, N))
    for t in range(T):
        loadings_t = Z[t] @ true_Gamma
        returns[t] = loadings_t @ true_factors[t] + np.random.randn(N) * sigma_eps

    return returns, Z, M, true_Gamma, true_Lambda, true_factors
```

## 8 Sensitivity Analysis Parameters

To test model robustness, we vary key parameters:

### 8.1 Idiosyncratic Noise ($\sigma_\epsilon$)

| $\sigma_\epsilon$ | SNR | Expected Performance |
|-------------------|-----|---------------------|
| 0.01 | ~2500:1 | Near-perfect recovery |
| 0.05 | ~100:1 | Excellent recovery |
| 0.10 | ~25:1 | Good recovery |
| 0.20 | ~6:1 | Moderate recovery |
| 0.50 | ~1:1 | Challenging |

### 8.2 Regularization Weight ($\alpha$)

| $\alpha$ | Effect |
|----------|--------|
| 0.001 | Essentially IPCA (minimal macro constraint) |
| 0.1 | Mild regularization |
| 1.0 | Balanced (default) |
| 5.0 | Strong macro constraint |

## 9 Evaluation Metrics

### 9.1 R-squared

$$R^2 = 1 - \frac{\sum_{i,t}(r_{i,t} - \hat{r}_{i,t})^2}{\sum_{i,t}(r_{i,t} - \bar{r})^2}$$

### 9.2 Grassmann Distance

For comparing subspaces spanned by $\Gamma$ and $\hat{\Gamma}$:

$$d_G(\Gamma, \hat{\Gamma}) = \sqrt{\sum_{j=1}^K \theta_j^2}$$

where $\theta_j = \arccos(\sigma_j)$ are principal angles and $\sigma_j$ are singular values of $Q_\Gamma' Q_{\hat{\Gamma}}$.

**Interpretation:**
- $d_G = 0$: Perfect recovery (identical subspaces)
- $d_G < 0.1$: Excellent recovery
- $d_G < 0.3$: Good recovery
- $d_G > 0.5$: Poor recovery

## 10 Reproducibility

All experiments use:
- **Random seed**: 42
- **NumPy version**: Compatible with 1.20+
- **Implementation**: See `notebooks/synthetic_comparison.ipynb`

To reproduce:
```python
np.random.seed(42)
returns, Z, M, true_Gamma, true_Lambda, true_factors = generate_synthetic_data()
```

## References

- Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics*, 134(3), 501-524.

- Kelly, B., Pruitt, S., & Su, Y. (2020). Instrumented principal component analysis. *Working Paper*.
