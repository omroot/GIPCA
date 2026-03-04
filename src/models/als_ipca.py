"""
ALS IPCA Implementation

A streamlined implementation of Instrumented Principal Component Analysis
using Alternating Least Squares algorithm.

Based on:
Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: 
A unified model of risk and return. Journal of Financial Economics.
"""

import pandas as pd
import numpy as np
import scipy.linalg as sla
from typing import Any, Dict, List, Optional, Tuple

from src.utils import subspace_error, recovery_report


class ALSIPCA:
    """
    Alternating Least Squares implementation of IPCA.

    The IPCA model:
        x_{i,t} = β_{i,t}' f_t + ε_{i,t}
        β_{i,t} = Γ' c_{i,t}

    Where:
        x_{i,t}: asset returns
        β_{i,t}: time-varying factor loadings
        f_t: latent factors
        c_{i,t}: observable characteristics
        Γ: characteristic-to-loading mapping matrix

    Interface matches GrassmannIPCAEstimator from ipca_opt.py for easy comparison.
    """

    def __init__(self, 
                 num_assets: int, 
                 num_fact: int, num_charact: int, 
                 win_len: int):
        """
        Initialize ALS IPCA estimator.

        Parameters
        ----------
        num_assets : int
            Number of assets (N)
        num_fact : int
            Number of latent factors (K)
        num_charact : int
            Number of characteristics (L)
        win_len : int
            Number of time periods (T)
        """
        self.num_assets: int = num_assets  # N
        self.K: int = num_fact  # k
        self.L: int = num_charact  # L
        self.T: int = win_len  # T

        self.Gamma: Optional[pd.DataFrame] = None
        self.factors: Optional[pd.DataFrame] = None
        self.Lambda: Optional[pd.Series] = None
        self._fitted: bool = False
        self.objective_history: List[float] = []

    def loss_fct(self, Gamma: np.ndarray, data: List[np.ndarray]) -> float:
        """
        Compute the IPCA loss function.

        Parameters
        ----------
        Gamma : np.ndarray
            Gamma matrix (L x K), can be flattened
        data : List[np.ndarray]
            [rets, Z] where rets is (T, N) and Z is (T, N, L)

        Returns
        -------
        float
            Objective function value (averaged over time periods)
        """
        Gamma = np.asarray(Gamma)
        if Gamma.ndim == 1:
            Gamma = Gamma.reshape(self.L, self.K)
        if Gamma.ndim == 2:
            assert Gamma.shape == (self.L, self.K)

        rets, Z = data
        assert rets.shape == (self.T, self.num_assets)
        assert Z.shape == (self.T, self.num_assets, self.L)

        obj = 0.0
        for t in range(self.T):
            Z_t = Z[t, :, :]  # (N, L)
            Lambda_t = Z_t @ Gamma  # (N, K)
            f_t, *_ = np.linalg.lstsq(Lambda_t, rets[t, :], rcond=None)  # (K,)
            fit = Lambda_t @ f_t  # (N,)
            resid = rets[t, :] - fit
            obj += resid @ resid

        return obj / self.T

    def fit(self,
             data: List[np.ndarray],
                 max_iter: int = 5000,
               tol: float = 1e-6,
                verbose: bool = True,
                seed: Optional[int] = None,
                W_star: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the ALS IPCA model.

        Parameters
        ----------
        data : List[np.ndarray]
            [rets, Z] where rets is (T, N) and Z is (T, N, L)
        max_iter : int, default 5000
            Maximum number of ALS iterations
        tol : float, default 1e-6
            Convergence tolerance
        verbose : bool, default True
            Whether to print iteration progress
        seed : Optional[int], default None
            Random seed for initialization. If provided, sets np.random.seed(seed)
            before random initialization.
        W_star : Optional[np.ndarray], default None
            True Gamma matrix (L x K) for computing subspace error at each
            iteration. Only used for diagnostics in simulation settings.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Gamma values array (L x K), objective history array)
        """
        rets, Z = data
        assert rets.shape == (self.T, self.num_assets), \
            f"rets shape {rets.shape} doesn't match expected {(self.T, self.num_assets)}"
        assert Z.shape == (self.T, self.num_assets, self.L), \
            f"Z shape {Z.shape} doesn't match expected {(self.T, self.num_assets, self.L)}"

        # Store raw data
        self._returns = rets
        self._characteristics = Z
        self._data = data

        # Compute managed portfolios and second moments from raw data
        self._X, self._W, self._N_valid = self._compute_managed_portfolios(rets, Z)

        self.times = list(range(self.T))
        self.characteristics_names = [f'char_{l}' for l in range(self.L)]

        # Reset objective history
        self.objective_history = []
        self.subspace_error_history: List[float] = []

        # Initialize Gamma randomly on the Grassmannian
        Gamma_old, factors_old = self._initialize_random(seed=seed)

        # Compute and store initial objective
        obj_init = self._compute_objective(Gamma_old)
        self.objective_history.append(obj_init)

        if W_star is not None:
            self.subspace_error_history.append(subspace_error(Gamma_old.values, W_star))

        if verbose:
            print(f"Iteration    0: Objective = {obj_init:.6f}")

        # ALS iterations
        for iteration in range(max_iter):
            # Update factors given Gamma
            factors_new = self._update_factors(Gamma_old, self._X, self._W)

            # Update Gamma given factors
            Gamma_new = self._update_gamma(factors_new, self._X, self._W, self._N_valid)

            # Compute objective function
            obj = self._compute_objective(Gamma_new)
            self.objective_history.append(obj)

            if W_star is not None:
                self.subspace_error_history.append(subspace_error(Gamma_new.values, W_star))

            # Check convergence
            gamma_change = np.max(np.abs(Gamma_new.values - Gamma_old.values))
            factor_change = np.max(np.abs(factors_new.values - factors_old.values))
            max_change = max(gamma_change, factor_change)

            if verbose:
                print(f"Iteration {iteration + 1:4d}: Objective = {obj:.6f}, "
                      f"Gamma change = {gamma_change:.8f}, "
                      f"Factor change = {factor_change:.8f}")

            if max_change < tol:
                if verbose:
                    print(f"\nConverged after {iteration + 1} iterations")
                    print(f"Initial objective: {self.objective_history[0]:.6f}")
                    print(f"Final objective:   {obj:.6f}")
                    print(f"Reduction:         {(1 - obj/self.objective_history[0])*100:.2f}%")
                break

            Gamma_old, factors_old = Gamma_new, factors_new

        # Store results
        self.Gamma = Gamma_new
        self.factors = factors_new
        self.Lambda = self.factors.mean(axis=1)
        self._fitted = True
        self.n_iterations = iteration + 1

        # Return Gamma and history
        return self.Gamma.values, np.array(self.objective_history)
    
    def predict(self, Z: pd.DataFrame) -> pd.Series:
        """
        Predict returns using fitted model with mean factor returns (Lambda).

        Parameters
        ----------
        Z : pd.DataFrame (N x L)
            Characteristics matrix for prediction

        Returns
        -------
        pd.Series
            Predicted returns
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before making predictions")

        loadings = Z @ self.Gamma  # N x K
        predicted_returns = loadings @ self.Lambda  # N x 1
        return predicted_returns
    
    def _compute_objective(self, Gamma: pd.DataFrame) -> float:
        """
        Compute the IPCA objective function (average squared residuals).

        Simply calls loss_fct with the Gamma values to ensure consistency.

        Parameters
        ----------
        Gamma : pd.DataFrame (L x K)
            Current Gamma estimate

        Returns
        -------
        float
            Objective function value (averaged over time periods)
        """
        return self.loss_fct(Gamma.values, self._data)

    def _compute_managed_portfolios(self, 
                                    returns: np.ndarray,
                                    characteristics: np.ndarray) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], pd.Series]:
        """
        Compute managed portfolios and second moments from raw data.

        Parameters
        ----------
        returns : np.ndarray (T x N)
            Asset returns
        characteristics : np.ndarray (T x N x L)
            Characteristics tensor

        Returns
        -------
        X : pd.DataFrame (L x T)
            Managed portfolio returns: X[t] = Z[t].T @ r[t] / N
        W : Dict[int, pd.DataFrame]
            Characteristic second moments: W[t] = Z[t].T @ Z[t] / N
        N_valid : pd.Series
            Number of valid observations per time period
        """
        T, N, L = characteristics.shape
        char_names = [f'char_{l}' for l in range(L)]
        times = list(range(T))

        X = pd.DataFrame(index=char_names, columns=times, dtype=float)
        W = {}
        N_valid = pd.Series(index=times, dtype=float)

        for t in range(T):
            Z_t = characteristics[t, :, :]  # (N, L)
            r_t = returns[t, :]  # (N,)

            # Managed portfolio: X_t = Z_t' @ r_t / N
            X[t] = (Z_t.T @ r_t) / N

            # Second moment: W_t = Z_t' @ Z_t / N
            W[t] = pd.DataFrame(
                (Z_t.T @ Z_t) / N,
                index=char_names,
                columns=char_names
            )

            N_valid[t] = N

        return X, W, N_valid

    def _initialize_random(self, use_best_of_population: bool = True,
                           seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Initialize Gamma randomly on the Grassmannian (same as ipca_opt).

        This generates a random matrix and orthonormalizes it via QR decomposition,
        which gives a uniform random point on the Grassmannian manifold.

        Parameters
        ----------
        use_best_of_population : bool, default True
            If True, generates a population of candidates (like ipca_opt's DE)
            and selects the one with the best objective. This matches ipca_opt's
            behavior at gen=0.
        seed : Optional[int], default None
            Random seed. If provided, sets np.random.seed(seed) before generating
            the population. Use the same seed as ipca_opt to get identical initial
            objectives.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (Gamma_init (L x K), factors_init (K x T))
        """
        char_names = [f'char_{l}' for l in range(self.L)]
        factor_names = [f'Factor_{k+1}' for k in range(self.K)]

        if use_best_of_population:
            # Match ipca_opt: generate population and pick best
            dim = self.L * self.K
            pop_size = 5 * dim  # Same as ipca_opt

            # Reset random seed to match ipca_opt's initialization state
            if seed is not None:
                np.random.seed(seed)

            # Generate population (same as ipca_opt's initialize_population)
            population = np.random.uniform(-1.0, 1.0, size=(pop_size, dim))

            # Project each onto Grassmannian (same as ipca_opt's project)
            best_obj = np.inf
            best_Gamma = None

            for i in range(pop_size):
                # Reshape and project via QR
                A = population[i].reshape(self.L, self.K)
                Q, _ = np.linalg.qr(A)
                Gamma_arr = Q[:, :self.K]

                # Compute objective
                obj = self.loss_fct(Gamma_arr, self._data)

                if obj < best_obj:
                    best_obj = obj
                    best_Gamma = Gamma_arr

            Gamma_arr = best_Gamma
        else:
            # Simple random initialization
            A = np.random.uniform(-1.0, 1.0, size=(self.L, self.K))
            Q, _ = np.linalg.qr(A)
            Gamma_arr = Q[:, :self.K]

        Gamma_init = pd.DataFrame(
            Gamma_arr,
            index=char_names,
            columns=factor_names
        )

        # Initialize factors by solving least squares for each time period
        factors_init = pd.DataFrame(
            index=factor_names,
            columns=list(range(self.T)),
            dtype=float
        )

        for t in range(self.T):
            Z_t = self._characteristics[t, :, :]  # (N, L)
            r_t = self._returns[t, :]  # (N,)
            Lambda_t = Z_t @ Gamma_arr  # (N, K)
            f_t, *_ = np.linalg.lstsq(Lambda_t, r_t, rcond=None)
            factors_init[t] = f_t

        return Gamma_init, factors_init

    def _update_factors(self,
                        Gamma: pd.DataFrame,
                        X: pd.DataFrame,
                        W: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Update factor estimates given Gamma."""
        factors_new = pd.DataFrame(
            index=Gamma.columns,
            columns=self.times,
            dtype=float
        )

        for t in self.times:
            # F_t = (Γ' W_t Γ)^{-1} Γ' X_t
            GWG = Gamma.T @ W[t] @ Gamma
            GX = Gamma.T @ X[t]

            # Use lstsq for numerical stability
            factors_new[t], *_ = np.linalg.lstsq(GWG.values, GX.values, rcond=None)

        return factors_new
    
    def _update_gamma(self, factors: pd.DataFrame, X: pd.DataFrame,
                     W: Dict[int, pd.DataFrame], N_valid: pd.Series) -> pd.DataFrame:
        """Update Gamma estimates given factors."""
        # Vectorized update using pooled regression
        vec_length = self.L * self.K
        numerator = np.zeros(vec_length)
        denominator = np.zeros((vec_length, vec_length))

        for t in self.times:
            F_t = factors[t].values  # K x 1
            X_t = X[t].values  # L x 1
            W_t = W[t].values  # L x L
            n_t = N_valid[t]

            # Kronecker product terms
            FF = np.outer(F_t, F_t)  # K x K
            numerator += np.kron(X_t, F_t) * n_t
            denominator += np.kron(W_t, FF) * n_t

        # Solve for vectorized Gamma - use pinv for numerical stability
        # Add small regularization for numerical stability
        denominator += 1e-8 * np.eye(vec_length)

        # Use lstsq for better numerical stability
        gamma_vec, *_ = np.linalg.lstsq(denominator, numerator, rcond=None)

        # Reshape to matrix form
        Gamma_new = pd.DataFrame(
            gamma_vec.reshape((self.L, self.K)),
            index=self.characteristics_names,
            columns=factors.index
        )

        # Apply orthonormalization constraint
        Gamma_new = self._orthonormalize_gamma(Gamma_new, factors)

        return Gamma_new
    
    def _orthonormalize_gamma(self, Gamma: pd.DataFrame,
                             factors: pd.DataFrame) -> pd.DataFrame:
        """Apply orthonormalization constraint to Gamma using QR decomposition."""
        # Use QR decomposition for numerical stability
        Q, R = np.linalg.qr(Gamma.values)

        # Keep only first K columns (for orthonormal basis)
        Gamma_orth = Q[:, :self.K]

        return pd.DataFrame(
            Gamma_orth,
            index=Gamma.index,
            columns=Gamma.columns
        )
    
    def _sign_convention(self, Gamma: pd.DataFrame, 
                        factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply sign convention: factors have positive means."""
        factor_means = factors.mean(axis=1)
        signs = np.where(factor_means >= 0, 1, -1)
        
        # Apply signs
        factors_signed = factors.mul(signs, axis=0)
        Gamma_signed = Gamma.mul(signs, axis=1)
        
        return Gamma_signed, factors_signed
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get estimation results.

        Returns
        -------
        Dict containing:
            - 'Gamma': characteristic loadings matrix
            - 'factors': time-varying factor returns
            - 'Lambda': average factor returns
            - 'objective_history': objective function values per iteration
            - 'n_iterations': number of iterations until convergence
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        return {
            'Gamma': self.Gamma,
            'factors': self.factors,
            'Lambda': self.Lambda,
            'objective_history': np.array(self.objective_history),
            'n_iterations': self.n_iterations
        }


def prepare_ipca_data(returns: pd.DataFrame, 
                      characteristics: pd.DataFrame, 
                      lag_chars: int = 1) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.Series]:
    """
    Prepare data for ALS IPCA estimation.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with MultiIndex (date, asset_id)
        
    characteristics : pd.DataFrame  
        Characteristics with MultiIndex (date, asset_id)
        
    lag_chars : int, default 1
        Number of periods to lag characteristics
        
    Returns
    -------
    X : pd.DataFrame (L x T)
        Managed portfolio returns
        
    W : Dict[str, pd.DataFrame]
        Characteristic second moments for each time period
        
    N_valid : pd.Series
        Number of valid observations per time period
    """
    # Get unique dates and sort
    dates = sorted(returns.index.get_level_values(0).unique())
    char_names = characteristics.columns.tolist()
    
    X = pd.DataFrame(index=char_names, columns=dates)
    W = {}
    N_valid = pd.Series(index=dates)
    
    for date in dates:
        # Get returns and characteristics for this date
        try:
            ret_t = returns.loc[date].dropna()
            
            # Get lagged characteristics
            char_date_idx = dates.index(date)
            if char_date_idx >= lag_chars:
                char_date = dates[char_date_idx - lag_chars]
                char_t = characteristics.loc[char_date].dropna()
                
                # Align assets (inner join)
                common_assets = ret_t.index.intersection(char_t.index)
                if len(common_assets) == 0:
                    continue
                    
                ret_aligned = ret_t.loc[common_assets]
                char_aligned = char_t.loc[common_assets]
                
                # Calculate managed portfolios and second moments
                n_valid = len(common_assets)
                X[date] = (char_aligned.T @ ret_aligned) / n_valid
                W[date] = (char_aligned.T @ char_aligned) / n_valid
                N_valid[date] = n_valid
                
        except KeyError:
            continue
    
    # Remove dates with no valid data
    valid_dates = X.columns[X.notna().any()]
    X = X[valid_dates]
    W = {date: W[date] for date in valid_dates if date in W}
    N_valid = N_valid[valid_dates].dropna()

    return X, W, N_valid


if __name__ == '__main__':
    from src.generate.ipca import generate_ipca_data
    from matplotlib import pyplot as plt

    # Same parameters as ipca_opt.py
    seed = 6890
    np.random.seed(seed)
    num_assets = 100  # N
    num_fact = 5      # k
    num_charact = 25  # m (L in ALS terminology)
    win_len = 21      # T
    include_intercept = False

    # Generate data using the same function as ipca_opt
    data, truth = generate_ipca_data(
        T=win_len,
        N=num_assets,
        m=num_charact,
        k=num_fact,
        include_intercept=include_intercept,
        seed=seed
    )

    W_star = truth['W_star']

    # =========================================================================
    # Fit ALS IPCA
    # =========================================================================
    print("=" * 60)
    print("ALS IPCA Estimation")
    print("=" * 60)
    print(f"Parameters: T={win_len}, N={num_assets}, m={num_charact}, k={num_fact}")
    print(f"Seed: {seed}")
    print("=" * 60)

    model = ALSIPCA(num_assets, num_fact, num_charact, win_len)
    Gamma_als, history_als = model.fit(
        data, max_iter=5000, tol=1e-6, verbose=True, seed=seed, W_star=W_star
    )

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nInitial obj = {history_als[0]:.6f}, Final obj = {history_als[-1]:.6f}, "
          f"Iterations = {len(history_als)-1}")
    print(f"Final subspace error: {model.subspace_error_history[-1]:.6f}")

    # Recovery report
    recovery_report(Gamma_als, truth)

    # =========================================================================
    # Plot objective function and subspace error
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    iterations = range(len(history_als))

    ax1.plot(iterations, history_als)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Objective function")
    ax1.set_title("IPCA Objective (ALS)")
    ax1.set_yscale('log')

    ax2.plot(iterations, model.subspace_error_history)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Subspace error")
    ax2.set_title("Subspace Error vs True W*")
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.show()

