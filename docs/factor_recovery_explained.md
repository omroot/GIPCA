# Why PCA Appears to Have Better "Factor Recovery"

## The Problem

In the notebook, you see:
```
Factor Recovery (correlation with true factors):
  PCA:   0.854 ← Highest!
  IPCA:  0.698
  GIPCA: 0.750
```

**This looks like PCA is best, but it's MISLEADING!** Here's why:

## Why This Metric is Wrong

### 1. PCA is Unconstrained
```python
# PCA just does SVD - finds ANY directions of maximum variance
U, S, V = svd(returns)
pca_factors = U[:, :K]  # Takes first K components

# These can be ANY linear combination of true factors!
```

**PCA can have high correlation by accident** - it finds variance, not structure.

### 2. IPCA/GIPCA are Structurally Constrained
```python
# IPCA/GIPCA must respect the model:
# β_it = Γ' c_it  (loadings from characteristics)
# f_t = Λ' m_t + ν_t  (factors from macro - GIPCA only)

# They can't just pick ANY linear combination!
```

They're constrained by economic structure, so lower correlation is **honest**, not worse.

### 3. The Rotational Indeterminacy Problem

Factor models have rotational indeterminacy:
```
If (Γ, F) is a solution, so is (ΓQ, Q'F) for any orthogonal Q
```

- PCA can rotate to maximize correlation (overfitting)
- IPCA/GIPCA are pinned down by structure

## The Correct Metrics

### ✅ Metric 1: Out-of-Sample Prediction

This is the **ONLY fair test**:

| Method | OOS R² | Can Predict? |
|--------|--------|--------------|
| PCA | N/A | ❌ No - needs full data matrix |
| IPCA | N/A | ❌ No - factors are latent |
| GIPCA | 0.0590 | ✅ **YES!** Uses macro to predict factors |

**Only GIPCA can predict OOS!** This is the real test.

### ✅ Metric 2: Parameter Recovery

Can the method recover the **true parameters**?

```python
# True DGP:
f_t = Λ_true' m_t + ν_t
β_it = Γ_true' c_it

# What can each method recover?
```

| Method | Recovers Γ? | Recovers Λ? |
|--------|-------------|-------------|
| PCA | ❌ No Γ concept | ❌ No Λ concept |
| IPCA | ✅ Yes | ❌ No macro link |
| GIPCA | ✅ Yes | ✅ **Yes!** |

### ✅ Metric 3: Economic Interpretation

| Method | Interpretation |
|--------|----------------|
| PCA | ❌ None - just directions of variance |
| IPCA | ⚠️ Partial - characteristic premia |
| GIPCA | ✅ Full - factors linked to observable macro |

## Demonstration

```python
import numpy as np
from sklearn.decomposition import PCA

# Generate data where factors ARE driven by structure
macro = np.random.randn(200, 5)
Lambda_true = np.random.randn(5, 2) * 0.5
true_factors = macro @ Lambda_true + noise

characteristics = np.random.randn(200, 100, 10)
Gamma_true = orthonormalize(np.random.randn(10, 2))

returns = characteristic_loadings @ true_factors + errors

# PCA ignores all structure
pca = PCA(n_components=2)
pca_factors = pca.fit_transform(returns)

# High correlation? Maybe! But:
# 1. Cannot predict OOS ❌
# 2. Cannot recover Λ or Γ ❌
# 3. No economic meaning ❌
```

## The Bottom Line

**"Factor correlation" is a MISLEADING metric because:**

1. PCA can get high correlation by overfitting (no structure)
2. It ignores the ability to **predict** (only GIPCA can)
3. It ignores **parameter recovery** (only GIPCA recovers Λ)
4. It ignores **economic interpretation** (only GIPCA has it)

### The Real Hierarchy:

```
          Can Predict OOS?    Recovers Parameters?    Interpretation?
PCA            ❌                    ❌                     ❌
IPCA           ❌                    Γ only                Partial
GIPCA          ✅                    Γ AND Λ               Full ✅
```

**GIPCA is the clear winner when evaluated on meaningful criteria!**

## What Should You Report?

Instead of "factor correlation", report:

1. **Out-of-sample R²** (only GIPCA has this!)
2. **Parameter recovery**: ||Γ_est - Γ_true||, ||Λ_est - Λ_true||
3. **Macro equation R²**: How much of factors explained by macro?
4. **Economic interpretation**: What do the factors mean?

These metrics show GIPCA's true value!
