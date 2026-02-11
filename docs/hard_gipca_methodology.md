# Hard GIPCA (Method A) — Methodology

## Overview

Hard GIPCA is a **constrained factor model** where factors are fully determined by the macro state. Unlike the "soft" GIPCA (Method B) which allows for latent factor residuals, the hard model enforces that all factor variation comes from observable macro variables.

**Key assumption:** $f_t \equiv \Lambda^\top m_t$ (no free/latent factors)

---

## Model Specification

### Returns Equation

$$
r_{it} = \lambda_{it}^\top f_t + \varepsilon_{it}
$$

where:
- $r_{it}$ — return of asset $i$ at time $t$
- $\lambda_{it} = Z_{it}^\top \Gamma$ — time-varying factor loadings
- $Z_{it} \in \mathbb{R}^L$ — characteristics of asset $i$ at time $t$
- $\Gamma \in \mathbb{R}^{L \times K}$ — characteristic-to-loading map (Grassmannian)
- $f_t \in \mathbb{R}^K$ — latent factors

### Hard Constraint

$$
f_t = \Lambda^\top m_t
$$

where:
- $m_t \in \mathbb{R}^R$ — observable macro variables (may include intercept)
- $\Lambda \in \mathbb{R}^{R \times K}$ — macro-to-factor loadings

### Combined Model

Substituting the constraint:

$$
r_{it} = Z_{it}^\top \Gamma \Lambda^\top m_t + \varepsilon_{it}
$$

In matrix form for all assets at time $t$:

$$
r_t = Z_t \Gamma \Lambda^\top m_t + \varepsilon_t
$$

---

## Objective Function

Minimize the sum of squared residuals subject to orthonormality:

$$
\min_{\Gamma, \Lambda} \quad \frac{1}{N} \sum_{t=1}^{T} \left\| r_t - Z_t \Gamma (\Lambda^\top m_t) \right\|^2 \quad \text{s.t.} \quad \Gamma^\top \Gamma = I_K
$$

The orthonormality constraint $\Gamma^\top \Gamma = I_K$ ensures identifiability (the columns of $\Gamma$ span a point on the Grassmannian $\text{Gr}(L, K)$).

---

## ALS Algorithm

The Alternating Least Squares (ALS) algorithm iterates between two closed-form updates until convergence.

### Notation

Define managed portfolios (pre-computed once):

$$
X_t = \frac{1}{N} Z_t^\top r_t \in \mathbb{R}^L \qquad W_t = \frac{1}{N} Z_t^\top Z_t \in \mathbb{R}^{L \times L}
$$

---

### Γ-Step (Update $\Gamma$ given $\Lambda$)

**Goal:** Solve for $\Gamma$ with $\Lambda$ fixed.

1. **Compute macro-implied factors:**
   $$g_t = \Lambda^\top m_t \in \mathbb{R}^K$$

2. **Form Kronecker normal equations:**

   Using row-major vectorization $\text{vec}_C(\Gamma) \in \mathbb{R}^{LK}$:

   $$
   \left[ \sum_{t=1}^{T} n_t \cdot \text{kron}(W_t, g_t g_t^\top) \right] \text{vec}_C(\Gamma) = \sum_{t=1}^{T} n_t \cdot \text{kron}(X_t, g_t)
   $$

   where $n_t$ is the number of valid observations at time $t$.

3. **Solve the linear system:**
   $$
   \gamma = (D + \rho I)^{-1} b
   $$
   where $D$ is the $LK \times LK$ denominator matrix, $b$ is the $LK$ numerator vector, and $\rho$ is a ridge parameter.

4. **Reshape and orthonormalize via SVD:**
   $$
   \Gamma_{\text{unc}} = \text{reshape}(\gamma, L, K)
   $$
   $$
   \Gamma_{\text{unc}} = U \Sigma V^\top \quad \Rightarrow \quad \Gamma = U V^\top
   $$

   This projection onto the Stiefel manifold ensures $\Gamma^\top \Gamma = I_K$.

---

### Λ-Step (Update $\Lambda$ given $\Gamma$)

**Goal:** Solve for $\Lambda$ with $\Gamma$ fixed.

1. **Project to K-dimensional space:**
   $$
   P_t = \Gamma^\top W_t \Gamma \in \mathbb{R}^{K \times K}
   $$
   $$
   q_t = \Gamma^\top X_t \in \mathbb{R}^K
   $$

2. **Form Kronecker normal equations:**

   Using row-major vectorization $\theta = \text{vec}_C(\Lambda^\top) \in \mathbb{R}^{KR}$:

   $$
   \left[ \sum_{t=1}^{T} n_t \cdot \text{kron}(P_t, m_t m_t^\top) + \rho I \right] \theta = \sum_{t=1}^{T} n_t \cdot \text{kron}(q_t, m_t)
   $$

3. **Solve and reshape:**
   $$
   \theta = (D + \rho I)^{-1} b
   $$
   $$
   \Lambda = \text{reshape}(\theta, K, R)^\top \in \mathbb{R}^{R \times K}
   $$

---

### Convergence Criterion

The algorithm terminates when:

$$
\max\left( \|\Gamma^{(k+1)} - \Gamma^{(k)}\|_\infty, \|\Lambda^{(k+1)} - \Lambda^{(k)}\|_\infty \right) < \tau
$$

where $\tau$ is the tolerance (default: $10^{-6}$).

---

## Initialization

The initialization uses a **best-of-population search on the Grassmannian** to find a good starting point.

### Algorithm

```
Input: pop_size = min(5 × L × K, 500)
Output: (Γ₀, Λ₀) with lowest loss

best_obj ← ∞
for i = 1 to pop_size do:

    1. Sample random Γ candidate:
       A ~ Uniform(-1, 1)^{L×K}
       Γ_cand ← QR(A)[:, :K]    # orthonormalize

    2. Estimate pseudo-factors via cross-sectional OLS:
       for t = 1 to T do:
           Ã_t ← Z_t Γ_cand     # N × K
           f̂_t ← (Ã_t'Ã_t)⁻¹ Ã_t' r_t   # K × 1
       end

    3. Estimate Λ via time-series OLS:
       F̂ ← [f̂_1, ..., f̂_T]'    # T × K
       Λ_cand ← (M'M)⁻¹ M' F̂    # R × K

    4. Evaluate hard loss:
       obj ← (1/N) Σ_t ‖r_t - Z_t Γ_cand Λ_cand' m_t‖²

    5. Update best:
       if obj < best_obj then:
           best_obj ← obj
           Γ₀ ← Γ_cand
           Λ₀ ← Λ_cand

return (Γ₀, Λ₀)
```

### Intuition

1. **Random Grassmannian sampling:** Generate diverse starting points by sampling random orthonormal matrices.

2. **Two-stage regression:** For each $\Gamma$ candidate:
   - First, extract what the factors "would be" if this $\Gamma$ were correct (cross-sectional OLS)
   - Then, find the best $\Lambda$ to explain those factors using macro variables (time-series OLS)

3. **Selection:** Keep the pair that minimizes the hard objective — this ensures the ALS algorithm starts from a reasonable basin of attraction.

---

## Complexity Analysis

| Component | Complexity |
|-----------|------------|
| Managed portfolios (pre-compute) | $O(T \cdot N \cdot L^2)$ |
| Γ-step (per iteration) | $O(T \cdot L^2 K^2 + (LK)^3)$ |
| Λ-step (per iteration) | $O(T \cdot K^2 R^2 + (KR)^3)$ |
| Initialization | $O(\text{pop\_size} \cdot T \cdot N \cdot K^2)$ |

Typically $K, R \ll L \ll N$, so the algorithm is efficient for large cross-sections.

---

## Relation to Soft GIPCA (Method B)

Hard GIPCA is the **$\alpha \to \infty$ limit** of the soft objective:

$$
\mathcal{L}_\alpha(\Gamma, \Lambda, f^0) = \frac{1}{N} \sum_t \|r_t - Z_t \Gamma f_t\|^2 + \alpha \sum_t \|f_t - \Lambda^\top m_t\|^2
$$

where $f_t = f_t^0 + \Lambda^\top m_t$ in the soft model.

As $\alpha \to \infty$, the penalty forces $f_t^0 \to 0$, recovering the hard constraint $f_t = \Lambda^\top m_t$.

---

## Properties

1. **Monotonic convergence:** The objective is guaranteed to decrease (or stay constant) at each ALS iteration.

2. **Macro R² = 1:** By construction, 100% of factor variation is explained by macro variables.

3. **Identifiability:** The Grassmannian constraint $\Gamma^\top \Gamma = I_K$ resolves rotational ambiguity up to sign/permutation.

4. **Interpretability:** $\Lambda$ directly maps macro variables to factors, enabling economic interpretation.

---

## References

- Missaoui & Lesniewski (2026), "Generalized Instrumented PCA"
- Kelly, Pruitt & Su (2019), "Characteristics are covariances: A unified model of risk and return"
- Connor & Korajczyk (1986), "Performance measurement with the arbitrage pricing theory"
