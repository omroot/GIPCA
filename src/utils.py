"""
Utility functions for Generalized IPCA
Helper functions for data preparation, evaluation, and visualization
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error, r2_score


def subspace_error(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the subspace error (Grassmann distance) between column spaces of A and B.

    Parameters
    ----------
    A : np.ndarray
        First matrix
    B : np.ndarray
        Second matrix (same number of columns as A)

    Returns
    -------
    float
        Grassmann distance between the column spaces
    """
    # Orthonormalize both matrices
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)

    # Compute singular values of Qa' @ Qb
    s = svd(Qa.T @ Qb, compute_uv=False)

    # Clamp for numerical stability
    s = np.clip(s, -1, 1)

    # Grassmann distance
    return np.sqrt(np.sum(np.arccos(s)**2))


def prepare_panel_data(
    returns_df: pd.DataFrame,
    characteristics_dict: Dict[str, pd.DataFrame],
    align: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Index, List[str]]:
    """
    Prepare panel data for GIPCA from DataFrames

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with dates as index and assets as columns
    characteristics_dict : dict of pd.DataFrame
        Dictionary where keys are characteristic names and values are DataFrames
        with same index/columns as returns_df
    align : bool, default=True
        Whether to align all data to common dates/assets

    Returns
    -------
    returns : np.ndarray of shape (n_times, n_assets)
    characteristics : np.ndarray of shape (n_times, n_assets, n_characteristics)
    dates : pd.DatetimeIndex
    assets : pd.Index
    char_names : list
    """
    if align:
        # Find common dates and assets
        dates = returns_df.index
        assets = returns_df.columns
        
        for char_name, char_df in characteristics_dict.items():
            dates = dates.intersection(char_df.index)
            assets = assets.intersection(char_df.columns)
        
        # Align all data
        returns_df = returns_df.loc[dates, assets]
        for char_name in characteristics_dict:
            characteristics_dict[char_name] = characteristics_dict[char_name].loc[dates, assets]
    else:
        dates = returns_df.index
        assets = returns_df.columns
    
    # Convert to numpy arrays
    returns = returns_df.values
    
    # Stack characteristics
    char_names = list(characteristics_dict.keys())
    n_times = len(dates)
    n_assets = len(assets)
    n_chars = len(char_names)
    
    characteristics = np.zeros((n_times, n_assets, n_chars))
    for i, char_name in enumerate(char_names):
        characteristics[:, :, i] = characteristics_dict[char_name].values
    
    return returns, characteristics, dates, assets, char_names


def prepare_macro_data(
    macro_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    standardize: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare macroeconomic variables for GIPCA

    Parameters
    ----------
    macro_df : pd.DataFrame
        DataFrame with dates as index and macro variables as columns
    dates : pd.DatetimeIndex
        Dates to align with panel data
    standardize : bool, default=True
        Whether to standardize macro variables

    Returns
    -------
    macro_vars : np.ndarray of shape (n_times, n_macro_vars)
    macro_names : list
    """
    # Align with panel dates
    macro_df = macro_df.reindex(dates)
    
    # Forward fill missing values (common for macro data)
    macro_df = macro_df.fillna(method='ffill')
    
    # Convert to numpy
    macro_vars = macro_df.values
    macro_names = list(macro_df.columns)
    
    # Standardize if requested
    if standardize:
        mean = np.nanmean(macro_vars, axis=0, keepdims=True)
        std = np.nanstd(macro_vars, axis=0, keepdims=True)
        std[std == 0] = 1
        macro_vars = (macro_vars - mean) / std
    
    return macro_vars, macro_names


def split_time_series(
    n_samples: int,
    test_size: Union[float, int] = 0.2,
    gap_size: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split for time series data

    Parameters
    ----------
    n_samples : int
        Total number of time periods
    test_size : float or int
        If float, proportion of data for testing
        If int, number of periods for testing
    gap_size : int
        Number of periods to skip between train and test

    Returns
    -------
    train_idx : np.ndarray
    test_idx : np.ndarray
    """
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size
    
    n_train = n_samples - n_test - gap_size
    
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train + gap_size, n_samples)
    
    return train_idx, test_idx


def evaluate_factor_model(
    true_returns: np.ndarray,
    predicted_returns: np.ndarray,
    factors: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of factor model performance

    Parameters
    ----------
    true_returns : np.ndarray
        True asset returns
    predicted_returns : np.ndarray
        Predicted asset returns
    factors : np.ndarray, optional
        Estimated factors for additional diagnostics

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Handle missing values
    mask = ~np.isnan(true_returns) & ~np.isnan(predicted_returns)
    true_flat = true_returns[mask]
    pred_flat = predicted_returns[mask]
    
    metrics = {}
    
    # Overall metrics
    metrics['r2'] = r2_score(true_flat, pred_flat)
    metrics['rmse'] = np.sqrt(mean_squared_error(true_flat, pred_flat))
    metrics['mae'] = np.mean(np.abs(true_flat - pred_flat))
    
    # Cross-sectional R2 (average across time)
    r2_cs = []
    for t in range(true_returns.shape[0]):
        mask_t = ~np.isnan(true_returns[t, :]) & ~np.isnan(predicted_returns[t, :])
        if np.sum(mask_t) > 2:
            r2_t = r2_score(true_returns[t, mask_t], predicted_returns[t, mask_t])
            r2_cs.append(r2_t)
    metrics['r2_cross_sectional'] = np.mean(r2_cs) if r2_cs else 0
    
    # Time-series R2 (average across assets)
    r2_ts = []
    for i in range(true_returns.shape[1]):
        mask_i = ~np.isnan(true_returns[:, i]) & ~np.isnan(predicted_returns[:, i])
        if np.sum(mask_i) > 2:
            r2_i = r2_score(true_returns[mask_i, i], predicted_returns[mask_i, i])
            r2_ts.append(r2_i)
    metrics['r2_time_series'] = np.mean(r2_ts) if r2_ts else 0
    
    # Factor statistics if provided
    if factors is not None:
        metrics['factor_mean'] = np.mean(factors, axis=0)
        metrics['factor_std'] = np.std(factors, axis=0)
        metrics['factor_sharpe'] = metrics['factor_mean'] / (metrics['factor_std'] + 1e-8)
        
        # Factor correlations
        factor_corr = np.corrcoef(factors.T)
        metrics['factor_correlation'] = factor_corr
    
    return metrics


def plot_factor_loadings(
    model: Any,
    char_names: List[str],
    asset_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    Visualize the characteristic-to-loading map (Gamma matrix)

    Parameters
    ----------
    model : GeneralizedIPCA
        Fitted GIPCA model
    char_names : list
        Names of characteristics
    asset_names : list, optional
        Names of assets for labeling
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    n_factors = model.n_factors
    n_chars = len(char_names)
    
    fig, axes = plt.subplots(1, n_factors, figsize=figsize)
    if n_factors == 1:
        axes = [axes]
    
    for k in range(n_factors):
        ax = axes[k]
        loadings = model.Gamma_[:, k]
        
        # Create bar plot
        x_pos = np.arange(n_chars)
        colors = ['green' if l > 0 else 'red' for l in loadings]
        ax.bar(x_pos, loadings, color=colors, alpha=0.7)
        
        ax.set_xlabel('Characteristic')
        ax.set_ylabel('Loading')
        ax.set_title(f'Factor {k+1} Loadings')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(char_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_macro_loadings(
    model: Any,
    macro_names: List[str],
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the macro-to-factor map (Lambda matrix)

    Parameters
    ----------
    model : GeneralizedIPCA
        Fitted GIPCA model
    macro_names : list
        Names of macro variables
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap of Lambda matrix
    im = ax.imshow(model.Lambda_, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(model.n_factors))
    ax.set_yticks(np.arange(len(macro_names)))
    ax.set_xticklabels([f'Factor {i+1}' for i in range(model.n_factors)])
    ax.set_yticklabels(macro_names)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(len(macro_names)):
        for j in range(model.n_factors):
            text = ax.text(j, i, f'{model.Lambda_[i, j]:.2f}',
                         ha='center', va='center', color='black' if abs(model.Lambda_[i, j]) < 0.5 else 'white')
    
    ax.set_title('Macro Variables to Factors Mapping (Lambda)')
    ax.set_xlabel('Factors')
    ax.set_ylabel('Macro Variables')
    
    plt.tight_layout()
    return fig, ax


def plot_factors(
    factors: np.ndarray,
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
    factor_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    Plot time series of estimated factors

    Parameters
    ----------
    factors : np.ndarray of shape (n_times, n_factors)
        Factor time series
    dates : pd.DatetimeIndex, optional
        Dates for x-axis
    factor_names : list, optional
        Names for factors
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    n_factors = factors.shape[1]
    n_cols = min(3, n_factors)
    n_rows = (n_factors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_factors == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    if dates is None:
        dates = np.arange(factors.shape[0])
    
    if factor_names is None:
        factor_names = [f'Factor {i+1}' for i in range(n_factors)]
    
    for k in range(n_factors):
        ax = axes[k]
        ax.plot(dates, factors[:, k], linewidth=1.5)
        ax.set_title(factor_names[k])
        ax.set_xlabel('Date')
        ax.set_ylabel('Factor Value')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add rolling mean
        window = min(20, factors.shape[0] // 10)
        if window > 1:
            rolling_mean = pd.Series(factors[:, k]).rolling(window=window, center=True).mean()
            ax.plot(dates, rolling_mean, color='red', alpha=0.5, linewidth=2, label=f'{window}-period MA')
            ax.legend()
    
    # Hide extra subplots
    for k in range(n_factors, len(axes)):
        axes[k].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def factor_mimicking_portfolios(
    model: Any,
    returns: np.ndarray,
    characteristics: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct factor mimicking portfolios

    Parameters
    ----------
    model : GeneralizedIPCA
        Fitted GIPCA model
    returns : np.ndarray
        Asset returns
    characteristics : np.ndarray
        Asset characteristics

    Returns
    -------
    portfolios : np.ndarray of shape (n_times, n_factors)
        Returns of factor mimicking portfolios
    weights : np.ndarray of shape (n_times, n_assets, n_factors)
        Portfolio weights
    """
    n_times, n_assets = returns.shape
    n_factors = model.n_factors
    
    portfolios = np.zeros((n_times, n_factors))
    weights = np.zeros((n_times, n_assets, n_factors))
    
    for t in range(n_times):
        # Get loadings for this period
        loadings = characteristics[t, :, :] @ model.Gamma_  # (n_assets, n_factors)
        
        # Compute weights (normalized loadings)
        for k in range(n_factors):
            loading_k = loadings[:, k]
            # Long-short portfolio
            weight_k = loading_k / (np.sum(np.abs(loading_k)) + 1e-8)
            weights[t, :, k] = weight_k
            
            # Portfolio return
            valid_idx = ~np.isnan(returns[t, :])
            if np.any(valid_idx):
                portfolios[t, k] = np.sum(weight_k[valid_idx] * returns[t, valid_idx])
    
    return portfolios, weights


def bootstrap_inference(
    model: Any,
    returns: np.ndarray,
    characteristics: np.ndarray,
    macro_vars: np.ndarray,
    n_bootstrap: int = 100,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Bootstrap inference for model parameters

    Parameters
    ----------
    model : GeneralizedIPCA
        Fitted GIPCA model
    returns : np.ndarray
        Asset returns
    characteristics : np.ndarray
        Asset characteristics
    macro_vars : np.ndarray
        Macro variables
    n_bootstrap : int
        Number of bootstrap samples
    alpha : float
        Significance level for confidence intervals

    Returns
    -------
    results : dict
        Bootstrap statistics and confidence intervals
    """
    n_times = returns.shape[0]
    
    # Store bootstrap estimates
    gamma_samples = []
    lambda_samples = []
    
    for b in range(n_bootstrap):
        # Bootstrap sample (block bootstrap for time series)
        block_size = max(5, n_times // 20)
        n_blocks = (n_times + block_size - 1) // block_size
        
        idx = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n_times - block_size + 1)
            idx.extend(range(start, min(start + block_size, n_times)))
        idx = idx[:n_times]
        
        # Resample data
        returns_b = returns[idx, :]
        characteristics_b = characteristics[idx, :, :]
        macro_vars_b = macro_vars[idx, :]
        
        # Fit model on bootstrap sample
        model_b = model.__class__(n_factors=model.n_factors, 
                                  max_iter=model.max_iter,
                                  tol=model.tol,
                                  alpha=model.alpha)
        try:
            model_b.fit(returns_b, characteristics_b, macro_vars_b)
            gamma_samples.append(model_b.Gamma_)
            lambda_samples.append(model_b.Lambda_)
        except:
            continue
    
    # Compute statistics
    gamma_samples = np.array(gamma_samples)
    lambda_samples = np.array(lambda_samples)
    
    results = {
        'gamma_mean': np.mean(gamma_samples, axis=0),
        'gamma_std': np.std(gamma_samples, axis=0),
        'gamma_ci_lower': np.percentile(gamma_samples, alpha/2 * 100, axis=0),
        'gamma_ci_upper': np.percentile(gamma_samples, (1-alpha/2) * 100, axis=0),
        'lambda_mean': np.mean(lambda_samples, axis=0),
        'lambda_std': np.std(lambda_samples, axis=0),
        'lambda_ci_lower': np.percentile(lambda_samples, alpha/2 * 100, axis=0),
        'lambda_ci_upper': np.percentile(lambda_samples, (1-alpha/2) * 100, axis=0),
        'n_successful': len(gamma_samples)
    }
    
    return results


def align_factors(
    estimated_factors: np.ndarray,
    true_factors: np.ndarray,
    estimated_Gamma: Optional[np.ndarray] = None,
    estimated_Lambda: Optional[np.ndarray] = None,
    true_Gamma: Optional[np.ndarray] = None,
    true_Lambda: Optional[np.ndarray] = None,
    normalize_signs: bool = True
) -> Dict[str, np.ndarray]:
    """
    Align estimated factors to true factors using Procrustes rotation.

    This implements the standard alignment procedure used in Kelly, Pruitt & Su (2019)
    and other factor model literature. It finds the optimal orthogonal rotation matrix Q
    that minimizes ||true_factors - estimated_factors @ Q||²_F.

    **CRITICAL for proper evaluation**: Without this alignment, sign flips and rotation
    indeterminacy will make results appear poor when they're actually excellent!

    Method: Procrustes Rotation via SVD
    ------------------------------------
    1. Compute M = estimated_factors' @ true_factors
    2. SVD: M = U @ Sigma @ V'
    3. Optimal rotation: Q = U @ V'
    4. Apply: aligned_factors = estimated_factors @ Q

    Parameters
    ----------
    estimated_factors : np.ndarray of shape (T, K)
        Estimated factor time series from fitted model
    true_factors : np.ndarray of shape (T, K)
        True factor time series from data generating process
    estimated_Gamma : np.ndarray of shape (L, K), optional
        Estimated characteristic map (will be rotated consistently)
    estimated_Lambda : np.ndarray of shape (R, K), optional
        Estimated macro map (will be rotated consistently)
    true_Gamma : np.ndarray of shape (L, K), optional
        True characteristic map (for computing recovery metrics)
    true_Lambda : np.ndarray of shape (R, K), optional
        True macro map (for computing recovery metrics)
    normalize_signs : bool, default=True
        If True, flip signs after Procrustes to ensure positive correlations
        (makes results more interpretable)

    Returns
    -------
    aligned : dict
        Dictionary containing:
        - 'factors': aligned factor time series (T, K)
        - 'Gamma': aligned Gamma if provided (L, K)
        - 'Lambda': aligned Lambda if provided (R, K)
        - 'rotation_matrix': orthogonal rotation matrix Q (K, K)
        - 'correlations': per-factor correlations with true factors (K,)
        - 'mean_correlation': average absolute correlation
        - 'mean_abs_correlation': same as mean_correlation
        - 'sign_flips': which factors had signs flipped (K,) if normalize_signs=True
        - 'Gamma_mae': mean absolute error in Gamma (if true_Gamma provided)
        - 'Lambda_correlations': per-factor Lambda correlations (if true_Lambda provided)

    Examples
    --------
    >>> # Basic usage
    >>> aligned = align_factors(model.factors_, true_factors)
    >>> print(f"Mean correlation: {aligned['mean_correlation']:.4f}")
    >>> aligned_factors = aligned['factors']

    >>> # With parameters
    >>> aligned = align_factors(
    ...     model.factors_, true_factors,
    ...     estimated_Gamma=model.Gamma_,
    ...     estimated_Lambda=model.Lambda_,
    ...     true_Gamma=true_Gamma,
    ...     true_Lambda=true_Lambda
    ... )
    >>> print(f"Gamma recovery MAE: {aligned['Gamma_mae']:.4f}")

    References
    ----------
    Kelly, B. T., Pruitt, S., & Su, Y. (2019). "Instrumented Principal Component Analysis."
    Section on factor alignment and identification.

    Notes
    -----
    This function solves the rotational indeterminacy problem in factor models.
    Without alignment, estimated factors can be rotated versions of true factors,
    leading to misleading evaluation metrics (e.g., negative correlations).
    """
    from scipy.linalg import svd, inv

    T, K = estimated_factors.shape

    # Step 1: Compute cross-product matrix M = F_est' @ F_true
    M = estimated_factors.T @ true_factors  # (K, K)

    # Step 2: SVD of M
    U, singular_values, Vt = svd(M, full_matrices=False)

    # Step 3: Optimal orthogonal rotation matrix (Procrustes solution)
    Q = U @ Vt  # (K, K)

    # Verify Q is orthogonal (sanity check)
    orthogonality_error = np.max(np.abs(Q.T @ Q - np.eye(K)))
    if orthogonality_error > 1e-6:
        import warnings
        warnings.warn(f"Rotation matrix Q is not orthogonal (error={orthogonality_error:.2e})")

    # Step 4: Apply rotation to factors
    aligned_factors = estimated_factors @ Q  # (T, K)

    # Step 5: Compute correlations
    correlations = np.zeros(K)
    for k in range(K):
        correlations[k] = np.corrcoef(true_factors[:, k], aligned_factors[:, k])[0, 1]

    # Step 6: Optional sign normalization for interpretability
    sign_flips = np.ones(K)
    if normalize_signs:
        for k in range(K):
            if correlations[k] < 0:
                aligned_factors[:, k] *= -1
                correlations[k] *= -1
                sign_flips[k] = -1.0

    # Prepare result dictionary
    result = {
        'factors': aligned_factors,
        'rotation_matrix': Q,
        'correlations': correlations,
        'mean_correlation': np.mean(correlations),
        'mean_abs_correlation': np.mean(np.abs(correlations)),
        'sign_flips': sign_flips
    }

    # Step 7: Apply same rotation to Gamma if provided
    if estimated_Gamma is not None:
        aligned_Gamma = estimated_Gamma @ Q

        # Apply sign flips if we normalized
        if normalize_signs:
            aligned_Gamma = aligned_Gamma * sign_flips[np.newaxis, :]

        result['Gamma'] = aligned_Gamma

        # Compute Gamma recovery metrics if true Gamma provided
        if true_Gamma is not None:
            diff = aligned_Gamma - true_Gamma
            result['Gamma_mae'] = np.mean(np.abs(diff))
            result['Gamma_rmse'] = np.sqrt(np.mean(diff ** 2))
            result['Gamma_max_error'] = np.max(np.abs(diff))

            # Subspace similarity
            from scipy.linalg import qr
            Q_true, _ = qr(true_Gamma, mode='economic')
            Q_est, _ = qr(aligned_Gamma, mode='economic')
            singular_vals = svd(Q_true.T @ Q_est, compute_uv=False)
            result['Gamma_subspace_similarity'] = np.mean(singular_vals ** 2)

    # Step 8: Apply rotation to Lambda if provided
    if estimated_Lambda is not None:
        # Lambda rotates as: Lambda_aligned = Lambda_est @ inv(Q')
        # Which is equivalent to: Lambda_est @ Q when Q is orthogonal (since inv(Q') = Q)
        aligned_Lambda = estimated_Lambda @ Q

        # Apply sign flips if we normalized
        if normalize_signs:
            aligned_Lambda = aligned_Lambda * sign_flips[np.newaxis, :]

        result['Lambda'] = aligned_Lambda

        # Compute Lambda recovery metrics if true Lambda provided
        if true_Lambda is not None:
            # Per-factor correlations
            lambda_correlations = np.zeros(K)
            for k in range(K):
                lambda_correlations[k] = np.corrcoef(
                    true_Lambda[:, k],
                    aligned_Lambda[:, k]
                )[0, 1]

            result['Lambda_correlations'] = lambda_correlations
            result['Lambda_mean_correlation'] = np.mean(np.abs(lambda_correlations))

            # Element-wise error
            diff = aligned_Lambda - true_Lambda
            result['Lambda_mae'] = np.mean(np.abs(diff))
            result['Lambda_rmse'] = np.sqrt(np.mean(diff ** 2))

    return result


def print_alignment_summary(alignment_result: Dict[str, np.ndarray]) -> None:
    """
    Print a formatted summary of factor alignment results.

    Parameters
    ----------
    alignment_result : dict
        Output dictionary from align_factors()

    Examples
    --------
    >>> aligned = align_factors(estimated_factors, true_factors)
    >>> print_alignment_summary(aligned)
    """
    K = len(alignment_result['correlations'])

    print("\n" + "=" * 70)
    print("FACTOR ALIGNMENT SUMMARY (Procrustes Rotation)")
    print("=" * 70)

    print("\n✅ Factor Recovery (After Alignment):")
    print("-" * 70)
    for k in range(K):
        corr = alignment_result['correlations'][k]
        sign_indicator = ""
        if 'sign_flips' in alignment_result and alignment_result['sign_flips'][k] < 0:
            sign_indicator = " (sign flipped)"
        print(f"  Factor {k+1}: correlation = {corr:7.4f}{sign_indicator}")

    print(f"\n  Mean absolute correlation: {alignment_result['mean_abs_correlation']:.4f}")

    # Gamma recovery
    if 'Gamma' in alignment_result:
        print("\n📊 Γ (Characteristic Map) Recovery:")
        print("-" * 70)
        if 'Gamma_mae' in alignment_result:
            print(f"  Mean Absolute Error:     {alignment_result['Gamma_mae']:.4f}")
            print(f"  Root Mean Squared Error: {alignment_result['Gamma_rmse']:.4f}")
            print(f"  Max Absolute Error:      {alignment_result['Gamma_max_error']:.4f}")
        if 'Gamma_subspace_similarity' in alignment_result:
            print(f"  Subspace Similarity:     {alignment_result['Gamma_subspace_similarity']:.4f}")

    # Lambda recovery
    if 'Lambda' in alignment_result:
        print("\n📈 Λ (Macro Map) Recovery:")
        print("-" * 70)
        if 'Lambda_correlations' in alignment_result:
            print("  Per-factor correlations:")
            for k in range(K):
                print(f"    Factor {k+1}: {alignment_result['Lambda_correlations'][k]:7.4f}")
            print(f"  Mean correlation:        {alignment_result['Lambda_mean_correlation']:.4f}")
        if 'Lambda_mae' in alignment_result:
            print(f"  Mean Absolute Error:     {alignment_result['Lambda_mae']:.4f}")
            print(f"  Root Mean Squared Error: {alignment_result['Lambda_rmse']:.4f}")

    print("\n" + "=" * 70)
    print("✓ Alignment complete - factors are now directly comparable!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("GIPCA utilities loaded successfully!")
