# Generalized Instrumented Principal Component Analysis (GIPCA)

## Overview

This repository contains a Python implementation of **Generalized Instrumented Principal Component Analysis (GIPCA)** - an extension of the standard IPCA model that incorporates macroeconomic factors to partially explain latent factors in asset pricing models.

Based on the paper: "Instrumented PCA with Macroeconomic Factors"

## Key Features

### Model Innovation
The Generalized IPCA extends the standard IPCA framework by adding a third equation:
- **Standard IPCA**: Links asset returns to factors via characteristic-dependent loadings
- **GIPCA Addition**: Links factors to macroeconomic variables

Model equations:
```
x_{i,t} = β_{i,t}' f_t + μ_{i,t}           (Asset returns)
β_{i,t} = c_{i,t}' Γ + η_{i,t}             (Characteristic loadings)
f_t = m_t' Λ + ν_t                          (NEW: Macro-driven factors)
```

### Implementation Features
- Alternating Least Squares (ALS) optimization
- Handles missing data in panel datasets
- Bootstrap inference for parameter uncertainty
- Comprehensive evaluation metrics
- Visualization tools for interpretation
- Factor mimicking portfolio construction

## Project Structure

```
GIPCA/
├── gipca/                          # Core package
│   ├── __init__.py
│   ├── generalized_ipca.py         # Main GIPCA model
│   ├── utils.py                    # Utility functions
│   └── simulation.py               # Simulation framework
├── notebooks/                      # Jupyter notebooks
│   ├── power_of_gipca_vs_ipca_pca.ipynb   # Parameter recovery comparison
│   └── comparison_gipca_ipca_pca.ipynb    # Method comparison
├── examples/                       # Example scripts
│   └── gipca_example.py           # Comprehensive usage example
├── docs/                           # Documentation
│   └── factor_recovery_explained.md
├── requirements.txt                # Package dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Installation

### Option 1: Install as Package (Recommended)

Install in development mode from the project root:
```bash
cd /path/to/GIPCA
pip install -e .
```

This will install the package and all dependencies, allowing you to import from anywhere:
```python
from gipca import GeneralizedIPCA
```

### Option 2: Install Dependencies Only

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn ipca jupyter
```

Then add the project directory to your Python path when using it.

## Quick Start

```python
from gipca import GeneralizedIPCA
import numpy as np

# Initialize model
model = GeneralizedIPCA(
    n_factors=3,      # Number of latent factors
    alpha=1.0,        # Weight for macro equation
    max_iter=500,     # Maximum iterations
    verbose=True      # Print convergence info
)

# Fit model
# X: (T, N) panel of returns
# characteristics: (T, N, L) time-varying characteristics
# macro_vars: (T, R) macroeconomic variables
factors = model.fit_transform(X, characteristics, macro_vars)

# Make predictions
predicted_returns = model.predict(characteristics, macro_vars=macro_vars)

# Evaluate
r2 = model.score(X, characteristics, macro_vars)
print(f"Model R-squared: {r2:.4f}")
```

## Model Components

### 1. Characteristic Map (Γ)
- Maps L characteristics to K factor loadings
- Dimension: (L × K)
- Interpretation: How firm characteristics determine factor exposures

### 2. Macro Map (Λ)
- Maps R macro variables to K factors
- Dimension: (R × K)
- Interpretation: How macroeconomic conditions drive factor realizations

### 3. Factors (f_t)
- K latent factors at each time t
- Partially explained by macro variables
- Residual component captures non-macro variation

## Estimation Algorithm

The model uses Alternating Least Squares (ALS) with three steps:

1. **Update factors** given Γ and Λ:
   ```
   f_t = (Γ' Z_t' Z_t Γ + I_K)^{-1} (Γ' Z_t' x_t + α Λ' m_t)
   ```

2. **Update Γ** given factors and Λ:
   - Panel regression of returns on characteristic-factor interactions

3. **Update Λ** given factors:
   - Time-series regression of factors on macro variables

## Example Usage

### Full Example with Financial Data

```python
import pandas as pd
from gipca import GeneralizedIPCA
from gipca.utils import prepare_panel_data, prepare_macro_data

# Load your data
returns_df = pd.read_csv('returns.csv', index_col=0, parse_dates=True)
macro_df = pd.read_csv('macro_variables.csv', index_col=0, parse_dates=True)

# Load characteristics (e.g., size, value, momentum)
characteristics_dict = {
    'size': pd.read_csv('size.csv', index_col=0, parse_dates=True),
    'value': pd.read_csv('value.csv', index_col=0, parse_dates=True),
    'momentum': pd.read_csv('momentum.csv', index_col=0, parse_dates=True)
}

# Prepare data
returns, characteristics, dates, assets, char_names = prepare_panel_data(
    returns_df, characteristics_dict
)
macro_vars, macro_names = prepare_macro_data(macro_df, dates)

# Fit GIPCA model
model = GeneralizedIPCA(n_factors=5, alpha=1.0)
factors = model.fit_transform(returns, characteristics, macro_vars)

# Analyze results
print(f"In-sample R²: {model.score(returns, characteristics, macro_vars):.4f}")

# Get factor loadings for specific characteristics
loadings = model.get_factor_loadings(characteristics)

# Construct factor mimicking portfolios
from gipca.utils import factor_mimicking_portfolios
portfolios, weights = factor_mimicking_portfolios(model, returns, characteristics)
```

## Advantages over Standard IPCA

1. **Economic Interpretation**: Factors are partially explained by observable macro variables
2. **Better Out-of-Sample Performance**: Macro information helps predict future factor realizations
3. **Risk Decomposition**: Separate macro-driven vs idiosyncratic factor components
4. **Factor Timing**: Use macro forecasts to predict factor returns
5. **Unified Framework**: Bridges statistical and fundamental factor models

## Model Diagnostics

### Evaluation Metrics
- Overall R²: Total variation explained
- Cross-sectional R²: Average R² across time periods
- Time-series R²: Average R² across assets
- Factor-macro R²: Variation in factors explained by macro variables

### Visualization Tools
```python
from gipca.utils import (
    plot_factor_loadings,    # Visualize Γ matrix
    plot_macro_loadings,     # Visualize Λ matrix
    plot_factors,            # Time series of factors
)

# Create visualizations
plot_factor_loadings(model, char_names)
plot_macro_loadings(model, macro_names)
plot_factors(factors, dates)
```

## Applications

### Asset Pricing
- Test whether anomalies are explained by macro risks
- Decompose expected returns into macro and non-macro components
- Build macro-aware factor models

### Risk Management
- Measure portfolio exposure to macro factors
- Stress testing with macro scenarios
- Dynamic hedging based on macro conditions

### Portfolio Construction
- Tilt portfolios based on macro views
- Construct macro-neutral factor portfolios
- Time factor exposures using macro signals

### Macroeconomic Analysis
- Extract latent economic factors from financial data
- Study transmission of macro shocks to asset prices
- Forecast economic variables using asset price information

## Parameter Tuning

### Key Parameters
- `n_factors`: Number of latent factors (typically 3-10)
- `alpha`: Weight for macro equation (0 = ignore macro, >1 = emphasize macro fit)
- `tol`: Convergence tolerance (1e-6 is usually sufficient)
- `standardize_factors`: Whether to normalize factors to unit variance

### Selection Guidelines
1. **Number of factors**: Use information criteria or cross-validation
2. **Alpha parameter**: Grid search to optimize out-of-sample performance
3. **Convergence**: Monitor objective function decrease

## Comparison with Original IPCA

| Feature | Original IPCA | Generalized IPCA |
|---------|---------------|------------------|
| Factors | Fully latent | Partially macro-driven |
| Equations | 2 (returns, loadings) | 3 (+ macro equation) |
| Parameters | Γ | Γ, Λ |
| Interpretation | Statistical | Economic + Statistical |
| Prediction | Requires full panel | Can use macro forecasts |

## References

- Kelly, B. T., Pruitt, S., & Su, Y. (2019). "Instrumented Principal Component Analysis." *SSRN Working Paper*
- Bai, J., & Ng, S. (2006). "Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions." *Econometrica*
- Stock, J. H., & Watson, M. W. (2002). "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics*

## License

MIT License - See LICENSE file for details

## Citation

If you use this implementation in your research, please cite:
```bibtex
@article{gipca2024,
  title={Generalized Instrumented Principal Component Analysis with Macroeconomic Factors},
  author={[Your Name]},
  year={2024}
}
```

## Contact

For questions or suggestions, please open an issue on GitHub.
