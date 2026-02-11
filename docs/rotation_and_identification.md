# Rotational Invariance and Identification in GIPCA

## Executive Summary

**Question:** Is GIPCA robust to rotation?

**Answer:** ❌ **No, GIPCA is NOT fully robust to rotation.** While the model predictions are invariant to orthogonal rotations, the individual parameter estimates (Γ and Λ) suffer from rotational indeterminacy.

---

## The Identification Problem

### Mathematical Formulation

GIPCA estimates three sets of parameters from the system:

```
x_{i,t} = β_{i,t}' f_t + μ_{i,t}        (1) Asset returns
β_{i,t} = c_{i,t}' Γ + η_{i,t}          (2) Characteristic loadings
f_t = m_t' Λ + ν_t                       (3) Macro-driven factors
```

### Rotational Equivalence

For **any invertible K×K matrix Q**, we can define rotated parameters:

| Original | Rotated | Relationship |
|----------|---------|--------------|
| f_t | f*_t = Q⁻¹ f_t | Rotate factors |
| β_{i,t} | β*_{i,t} = Q' β_{i,t} | Rotate loadings |
| Γ | Γ* = Γ Q | Rotate characteristic map |
| Λ | Λ* = Λ (Q')⁻¹ | Rotate macro map |

**Critical insight:** The rotated system gives **identical predictions**:

```
x_{i,t} = β_{i,t}' f_t = β*_{i,t}' f*_t = c_{i,t}' Γ Q Q⁻¹ f_t
```

Therefore, **we cannot uniquely identify Γ, Λ, or f_t without additional restrictions.**

---

## What IS Identified vs What IS NOT

### ✅ Identified (Invariant to Rotation)

1. **Model predictions**: x̂_{i,t} = c_{i,t}' Γ f_t
2. **In-sample R²**: Model fit quality
3. **Out-of-sample forecasts**: Prediction accuracy
4. **Column space of Γ**: The subspace spanned by factor loadings
5. **Goodness of fit statistics**: MSE, likelihood, etc.

### ❌ NOT Identified (Rotation Indeterminacy)

1. **Individual elements of Γ**: Specific characteristic-factor relationships
2. **Individual elements of Λ**: Specific macro-factor relationships
3. **Factor realizations f_t**: Time series of factors can be rotated
4. **Factor ordering**: Which factor is "first", "second", etc.
5. **Factor signs**: Positive vs negative orientation
6. **Economic interpretation**: Without restrictions, factors lack clear meaning

---

## Why Does the Macro Equation Not Resolve This?

One might hope that equation (3), f_t = m_t' Λ + ν_t, would pin down the rotation. However:

### Theoretical Argument

If we rotate factors by Q:
```
f*_t = Q⁻¹ f_t = Q⁻¹ (m_t' Λ + ν_t)
     = m_t' (Λ (Q')⁻¹) + Q⁻¹ ν_t
     = m_t' Λ* + ν*_t
```

The macro equation is **satisfied by the rotated parameters Λ* = Λ (Q')⁻¹** with rotated residuals ν*_t = Q⁻¹ ν_t.

### Empirical Evidence

The macro equation provides **partial identification**:
- It constrains the solution space (fewer admissible rotations)
- Factors with strong macro links are more stable
- But it doesn't eliminate rotational freedom entirely

**Why not?**
- Macro variables rarely span the full factor space (R ≥ K but macro R² < 1)
- Residual component ν_t can absorb rotations
- Multiple factors may have similar macro explanations

---

## Current Implementation: QR Normalization

### What the Code Does

In `gipca/generalized_ipca.py`, line 211:

```python
Gamma, _ = linalg.qr(Gamma, mode='economic')
```

This applies **QR decomposition** to make Γ orthonormal: Γ' Γ = I_K

### Benefits of QR Normalization

1. ✅ Ensures Γ has orthogonal columns (uncorrelated factors)
2. ✅ Fixes the scale of factor loadings
3. ✅ Makes numerical optimization more stable
4. ✅ Provides a canonical representation

### Limitations

1. ❌ Still allows orthogonal rotations (Q such that Q' Q = I)
2. ❌ Doesn't fix factor ordering
3. ❌ Doesn't fix factor signs
4. ❌ Choice of orthonormal basis is arbitrary

**Bottom line:** QR normalization provides **partial identification** but leaves rotational freedom within the orthogonal group O(K).

---

## Practical Implications

### 1. Comparing Estimates Across Runs

If you fit GIPCA multiple times (different initializations or subsamples), you'll get:

- ✅ **Identical predictions** (up to numerical precision)
- ✅ **Same R²** and fit statistics
- ❌ **Different Γ and Λ elements** (related by rotation)
- ❌ **Different factor time series** (rotated versions)

**Example:**
```python
# Two runs may give
Gamma_1 = [0.8, 0.6, 0.0]  # Run 1, Factor 1
Gamma_2 = [0.6, -0.8, 0.0] # Run 2, Factor 1 (rotated!)
```

### 2. Interpreting Parameters

⚠️ **Be cautious when interpreting individual elements of Γ or Λ**

- Don't over-interpret specific values
- Focus on patterns and relative magnitudes within factors
- Compare subspaces, not individual loadings
- Use economic theory to guide interpretation

### 3. Reporting Results

When reporting GIPCA results:

1. **DO report:**
   - Overall model fit (R²)
   - Subspace recovery metrics
   - Prediction accuracy
   - Explained variance by factors

2. **AVOID claiming:**
   - "Factor 1 loads 0.75 on size" (without identification restrictions)
   - Precise quantitative interpretations of Γ or Λ
   - Comparisons of factor magnitudes across different model fits

---

## Proposed Solutions

### Strategy 1: Sign and Ordering Restrictions ⭐ Recommended

**Idea:** Fix signs and ordering based on interpretable criteria

```python
def identify_factors(model, macro_vars, characteristics):
    """
    Apply identification restrictions to GIPCA estimates

    1. Order factors by variance explained
    2. Normalize first element of each factor loading to be positive
    3. Order by macro R² (how much macro explains each factor)
    """
    K = model.n_factors

    # Compute variance explained by each factor
    factor_vars = np.var(model.factors_, axis=0)

    # Compute macro R² for each factor
    macro_r2 = np.zeros(K)
    for k in range(K):
        f_k = model.factors_[:, k]
        f_k_pred = macro_vars @ model.Lambda_[:, k]
        macro_r2[k] = 1 - np.var(f_k - f_k_pred) / np.var(f_k)

    # Order factors by macro R² (descending)
    order = np.argsort(-macro_r2)

    # Reorder
    model.Gamma_ = model.Gamma_[:, order]
    model.Lambda_ = model.Lambda_[:, order]
    model.factors_ = model.factors_[:, order]

    # Fix signs: make first non-zero Gamma element positive for each factor
    for k in range(K):
        first_nonzero_idx = np.argmax(np.abs(model.Gamma_[:, k]))
        if model.Gamma_[first_nonzero_idx, k] < 0:
            model.Gamma_[:, k] *= -1
            model.Lambda_[:, k] *= -1
            model.factors_[:, k] *= -1

    return model
```

**Advantages:**
- Simple to implement
- Provides consistent ordering across runs
- Based on economically meaningful criteria

**Limitations:**
- Ordering by macro R² is somewhat arbitrary
- Sign normalization is a convention, not economic restriction

---

### Strategy 2: Varimax Rotation 🔄

**Idea:** Rotate factors to maximize interpretability (sparse loadings)

```python
from sklearn.decomposition import PCA

def varimax_rotation(model, gamma_penalty=1.0):
    """
    Apply Varimax rotation to make Gamma more interpretable

    Rotates factors to maximize variance of squared loadings,
    creating 'simple structure' with clear characteristic-factor relationships
    """
    # Standard Varimax on Gamma
    from scipy.stats import ortho_group

    # Use numerical optimization to find Q that maximizes simplicity
    # Objective: maximize sum of variances of Gamma²

    def varimax_objective(Q_flat):
        Q = Q_flat.reshape(model.n_factors, model.n_factors)
        Gamma_rot = model.Gamma_ @ Q
        # Variance of squared loadings (encourages sparsity)
        return -np.sum(np.var(Gamma_rot**2, axis=0))

    # Optimize over orthogonal matrices
    # (Implementation requires constrained optimization)

    return model
```

**Advantages:**
- Statistical criterion (interpretability)
- Widely used in factor analysis
- Creates "simple structure"

**Limitations:**
- No guarantee of economic interpretation
- May not align with macro equation
- Computationally more expensive

---

### Strategy 3: Target Rotation Using Economic Theory 🎯

**Idea:** Rotate factors to align with economically meaningful targets

```python
def targeted_rotation(model, target_loadings):
    """
    Rotate factors to align with pre-specified target loadings

    Parameters:
    -----------
    target_loadings : array (L x K)
        Desired pattern for Gamma (e.g., market, size, value factors)

    Finds rotation Q that minimizes ||Gamma @ Q - target_loadings||²
    """
    # Solve Procrustes problem
    U, _, Vt = linalg.svd(model.Gamma_.T @ target_loadings)
    Q = U @ Vt

    # Apply rotation
    model.Gamma_ = model.Gamma_ @ Q
    model.Lambda_ = model.Lambda_ @ linalg.inv(Q.T)
    model.factors_ = model.factors_ @ Q.T

    return model
```

**Advantages:**
- Incorporates economic theory
- Aligns with known factor structures
- Facilitates interpretation

**Limitations:**
- Requires pre-specified targets
- May not fit data well if targets are misspecified

---

### Strategy 4: Maximize Macro Explanation 📊 ⭐ Best for GIPCA

**Idea:** Rotate factors to maximize total R² from macro equation

```python
def maximize_macro_explanation(model, macro_vars):
    """
    Find rotation that maximizes sum of macro R² across factors

    This uses the unique structure of GIPCA - rotate to make factors
    as explainable by macro variables as possible
    """
    from scipy.optimize import minimize
    from scipy.linalg import expm

    def total_macro_r2(Q_params):
        # Parameterize orthogonal matrix via exponential map
        K = model.n_factors
        # Skew-symmetric matrix
        A = np.zeros((K, K))
        idx = 0
        for i in range(K):
            for j in range(i+1, K):
                A[i, j] = Q_params[idx]
                A[j, i] = -Q_params[idx]
                idx += 1
        Q = expm(A)  # Orthogonal matrix

        # Rotate factors and Lambda
        factors_rot = model.factors_ @ Q.T
        Lambda_rot = model.Lambda_ @ linalg.inv(Q.T)

        # Compute total macro R²
        total_r2 = 0
        for k in range(K):
            f_k = factors_rot[:, k]
            f_pred = macro_vars @ Lambda_rot[:, k]
            r2_k = 1 - np.var(f_k - f_pred) / np.var(f_k)
            total_r2 += r2_k

        return -total_r2  # Minimize negative = maximize

    # Optimize
    n_params = model.n_factors * (model.n_factors - 1) // 2
    result = minimize(total_macro_r2, np.zeros(n_params), method='BFGS')

    # Apply optimal rotation
    # (extract Q from result.x and apply)

    return model
```

**Advantages:**
- ✅ Uses GIPCA's unique structure
- ✅ Maximizes macro interpretability
- ✅ Economically motivated
- ✅ Automatic, no manual specification needed

**Limitations:**
- Computationally intensive
- May overfit to macro variables
- Requires optimization over Stiefel manifold

---

## Recommendations

### For Research Applications

1. **Report model fit metrics** (R², MSE) - these are identified ✅
2. **Use Strategy 1** (sign + ordering restrictions) for consistent reporting
3. **Focus on subspace recovery** rather than individual parameter values
4. **When comparing runs**, align factors using rotation before comparison

### For Applied/Interpretation Work

1. **Use Strategy 4** (maximize macro explanation) - best leverages GIPCA structure
2. **Combine with Strategy 1** for final presentation
3. **Report macro R²** for each factor to show macro interpretability
4. **Use economic labels** based on which characteristics/macros load strongly

### For Production Systems

1. **Standardize identification scheme** across all model fits
2. **Store rotation matrices** when updating models over time
3. **Monitor subspace stability** not individual parameters
4. **Use out-of-sample validation** (predictions, not parameters)

---

## Implementation Roadmap

### Short-term (Easy to add)

1. Add `identify_factors()` function to GIPCA class (Strategy 1)
2. Add option to order by macro R² or variance
3. Document identification assumptions in docstrings

### Medium-term (Moderate effort)

1. Implement Varimax rotation option (Strategy 2)
2. Add Procrustes alignment method (Strategy 3)
3. Create visualization tools for comparing subspaces

### Long-term (Research needed)

1. Implement optimal macro-explanation rotation (Strategy 4)
2. Develop asymptotic theory for identified parameters
3. Bootstrap inference accounting for rotation uncertainty

---

## Conclusion

**Key Takeaway:** GIPCA, like all factor models, suffers from rotational indeterminacy. The macro equation helps by providing structure but does not fully resolve identification.

**Practical advice:**
- ✅ Trust the model's **predictions and fit metrics**
- ⚠️  Be careful with **parameter interpretation**
- 🔧 Apply **identification restrictions** for consistent reporting
- 📊 Use **macro R²** to assess economic interpretability

**The good news:** This doesn't undermine GIPCA's value for:
- Prediction
- Dimensionality reduction
- Out-of-sample forecasting
- Risk decomposition

**The caveat:** Structural interpretation of Γ and Λ requires additional assumptions or restrictions.

---

## References

1. **Kelly, Pruitt, & Su (2019)**: "Instrumented Principal Component Analysis" - original IPCA paper
2. **Bai & Ng (2013)**: "Principal Components Estimation and Identification of Static Factors" - discusses factor identification
3. **Stock & Watson (2002)**: "Forecasting Using Principal Components from a Large Number of Predictors" - rotation in factor models
4. **Lawley & Maxwell (1971)**: "Factor Analysis as a Statistical Method" - classical treatment of rotation problem

---

## See Also

- `/notebooks/rotation_sensitivity_test.ipynb` - Empirical demonstration
- `/docs/factor_recovery_explained.md` - Parameter recovery analysis
- GIPCA paper Section 2.3 - Identification discussion
