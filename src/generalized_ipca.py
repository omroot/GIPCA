"""
Generalized Instrumented Principal Component Analysis (GIPCA)

Extension of IPCA to incorporate macroeconomic factors, following the same
methodology as als_ipca.py for consistency and performance.

Based on:
Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics are covariances.
Extended with macro-factor linkage: f_t = Lambda' m_t + nu_t

The GIPCA model:
    x_{i,t} = beta_{i,t}' f_t + epsilon_{i,t}
    beta_{i,t} = Gamma' c_{i,t}
    f_t = Lambda' m_t + nu_t

Objective function (normalized):
    Q = (1/N) sum_t ||x_t - Z_t Gamma f_t||^2 + alpha * (1/K) sum_t ||f_t - Lambda' m_t||^2
"""

import numpy as np
import pandas as pd
import scipy.linalg as sla
from typing import Dict, Tuple, Optional


class GIPCA:
    """
    Generalized Instrumented PCA with Macroeconomic Factors.

    Extends IPCA by allowing latent factors to be driven by macro variables.
    Uses the same ALS methodology as ALSIPCA for consistency.

    Parameters
    ----------
    num_assets : int
        Number of assets (N)
    num_fact : int
        Number of latent factors (K)
    num_charact : int
        Number of characteristics (L)
    num_macro : int
        Number of macro variables (R)
    win_len : int
        Number of time periods (T)
    alpha : float, default 1.0
        Weight on macro equation. Controls the trade-off between return fit and
        factor predictability:
        - Lower alpha (0.1): Higher in-sample R², but factors less predictable
        - Higher alpha (1.0): Lower in-sample R², but factors more predictable
        For forecasting applications, alpha=1.0 is recommended as it enforces
        the macro constraint needed for genuine out-of-sample prediction.
    """

    def __init__(self, num_assets: int, num_fact: int, num_charact: int,
                 num_macro: int, win_len: int, alpha: float = 1.0):
        self.num_assets = num_assets  # N
        self.K = num_fact             # K
        self.L = num_charact          # L
        self.R = num_macro            # R
        self.T = win_len              # T
        self.alpha = alpha

        # Parameters to be estimated
        self.Gamma = None   # L x K
        self.Lambda = None  # R x K
        self.factors = None # K x T

        self._fitted = False
        self.objective_history = []
        self.n_iterations = 0

    def loss_fct(self, Gamma: np.ndarray, Lambda: np.ndarray,
                 factors: np.ndarray, data: list) -> float:
        """
        Compute the GIPCA loss function.

        Parameters
        ----------
        Gamma : np.ndarray (L x K)
            Characteristic-to-loading map
        Lambda : np.ndarray (R x K)
            Macro-to-factor map
        factors : np.ndarray (K x T)
            Factor realizations
        data : list
            [rets, Z, M] where rets is (T, N), Z is (T, N, L), M is (T, R)

        Returns
        -------
        float
            Objective function value
        """
        rets, Z, M = data
        T, N = rets.shape

        obj_cross = 0.0
        obj_macro = 0.0

        for t in range(T):
            Z_t = Z[t, :, :]  # (N, L)
            r_t = rets[t, :]  # (N,)
            m_t = M[t, :]     # (R,)
            f_t = factors[:, t]  # (K,)

            # Cross-sectional loss
            pred = Z_t @ Gamma @ f_t
            obj_cross += np.sum((r_t - pred) ** 2)

            # Macro loss
            pred_f = Lambda.T @ m_t
            obj_macro += np.sum((f_t - pred_f) ** 2)

        # Normalize both terms so they are O(T)
        # Cross-sectional: divide by N
        # Macro: divide by K
        K = factors.shape[0]
        return obj_cross / N + self.alpha * obj_macro / K

    def fit(self, data: list, max_iter: int = 1000, tol: float = 1e-6,
            verbose: bool = True, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the GIPCA model using alternating least squares.

        Parameters
        ----------
        data : list
            [rets, Z, M] where:
            - rets: (T, N) asset returns
            - Z: (T, N, L) characteristics
            - M: (T, R) macro variables
        max_iter : int
            Maximum ALS iterations
        tol : float
            Convergence tolerance (max parameter change)
        verbose : bool
            Print progress
        seed : int, optional
            Random seed for initialization

        Returns
        -------
        Gamma : np.ndarray (L x K)
        objective_history : np.ndarray
        """
        rets, Z, M = data

        # Validate dimensions
        assert rets.shape == (self.T, self.num_assets), \
            f"rets shape {rets.shape} != expected {(self.T, self.num_assets)}"
        assert Z.shape == (self.T, self.num_assets, self.L), \
            f"Z shape {Z.shape} != expected {(self.T, self.num_assets, self.L)}"
        assert M.shape == (self.T, self.R), \
            f"M shape {M.shape} != expected {(self.T, self.R)}"

        # Store data
        self._returns = rets
        self._characteristics = Z
        self._macro = M
        self._data = data

        # Pre-compute managed portfolios
        self._X, self._W, self._N_valid = self._compute_managed_portfolios(rets, Z)

        # Pre-compute macro moments
        self._M_sum = M.T @ M  # R x R

        self.times = list(range(self.T))
        self.char_names = [f'char_{l}' for l in range(self.L)]
        self.factor_names = [f'Factor_{k+1}' for k in range(self.K)]
        self.macro_names = [f'macro_{r}' for r in range(self.R)]

        # Reset history
        self.objective_history = []

        # Initialize parameters
        Gamma_old, Lambda_old, factors_old = self._initialize(seed=seed)

        # Initial objective
        obj_init = self._compute_objective(Gamma_old, Lambda_old, factors_old)
        self.objective_history.append(obj_init)

        if verbose:
            print(f"Iteration    0: Objective = {obj_init:.6f}")

        # ALS iterations
        for iteration in range(max_iter):
            # Step 1: Update factors given Gamma, Lambda
            factors_new = self._update_factors(Gamma_old, Lambda_old)

            # Step 2: Update Gamma given factors
            Gamma_new = self._update_gamma(factors_new)

            # Step 3: Update Lambda given factors
            Lambda_new = self._update_lambda(factors_new)

            # Compute objective
            obj = self._compute_objective(Gamma_new, Lambda_new, factors_new)
            self.objective_history.append(obj)

            # Check convergence (max parameter change)
            gamma_change = np.max(np.abs(Gamma_new.values - Gamma_old.values))
            lambda_change = np.max(np.abs(Lambda_new.values - Lambda_old.values))
            factor_change = np.max(np.abs(factors_new.values - factors_old.values))
            max_change = max(gamma_change, lambda_change, factor_change)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:4d}: Objective = {obj:.6f}, "
                      f"Max change = {max_change:.2e}")

            if max_change < tol:
                if verbose:
                    print(f"\nConverged after {iteration + 1} iterations")
                    print(f"Initial objective: {self.objective_history[0]:.6f}")
                    print(f"Final objective:   {obj:.6f}")
                    print(f"Reduction:         {(1 - obj/self.objective_history[0])*100:.2f}%")
                break

            Gamma_old = Gamma_new
            Lambda_old = Lambda_new
            factors_old = factors_new

        # Store results
        self.Gamma = Gamma_new
        self.Lambda = Lambda_new
        self.factors = factors_new
        self._fitted = True
        self.n_iterations = iteration + 1

        return self.Gamma.values, np.array(self.objective_history)

    def _compute_managed_portfolios(self, rets: np.ndarray,
                                    Z: np.ndarray) -> Tuple[pd.DataFrame, Dict, pd.Series]:
        """
        Compute managed portfolios X_t = Z_t' r_t / N and second moments W_t = Z_t' Z_t / N.
        """
        T, N, L = Z.shape

        X = pd.DataFrame(index=self.char_names if hasattr(self, 'char_names')
                         else [f'char_{l}' for l in range(L)],
                         columns=list(range(T)), dtype=float)
        W = {}
        N_valid = pd.Series(index=list(range(T)), dtype=float)

        char_names = X.index.tolist()

        for t in range(T):
            Z_t = Z[t, :, :]  # (N, L)
            r_t = rets[t, :]  # (N,)

            X[t] = (Z_t.T @ r_t) / N
            W[t] = pd.DataFrame((Z_t.T @ Z_t) / N, index=char_names, columns=char_names)
            N_valid[t] = N

        return X, W, N_valid

    def _initialize(self, seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Initialize Gamma, Lambda, and factors.
        Uses best-of-population approach for Gamma (same as ALSIPCA).
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize Gamma via best-of-population on Grassmannian
        dim = self.L * self.K
        pop_size = min(5 * dim, 500)  # Cap population size

        best_obj = np.inf
        best_Gamma = None
        best_factors = None

        # Temporary Lambda for initialization (OLS on random factors)
        temp_Lambda = np.zeros((self.R, self.K))

        for i in range(pop_size):
            # Random Gamma on Grassmannian
            A = np.random.uniform(-1.0, 1.0, size=(self.L, self.K))
            Q, _ = np.linalg.qr(A)
            Gamma_arr = Q[:, :self.K]

            # Estimate factors given Gamma (no macro constraint initially)
            factors_arr = np.zeros((self.K, self.T))
            for t in range(self.T):
                Z_t = self._characteristics[t, :, :]
                r_t = self._returns[t, :]
                Lambda_t = Z_t @ Gamma_arr  # (N, K)
                f_t, *_ = np.linalg.lstsq(Lambda_t, r_t, rcond=None)
                factors_arr[:, t] = f_t

            # Compute objective (with zero Lambda initially)
            obj = self.loss_fct(Gamma_arr, temp_Lambda, factors_arr, self._data)

            if obj < best_obj:
                best_obj = obj
                best_Gamma = Gamma_arr
                best_factors = factors_arr

        # Initialize Lambda via OLS
        # Lambda' = (sum f_t m_t') @ (sum m_t m_t')^-1
        # Solving: M'M @ Lambda = (F @ M).T  =>  Lambda = (M'M)^-1 @ (F @ M).T
        sum_fm = best_factors @ self._macro  # K x R
        Lambda_arr = sla.solve(self._M_sum + 1e-8 * np.eye(self.R),
                               sum_fm.T, assume_a='sym')  # R x K

        # Wrap in DataFrames
        Gamma = pd.DataFrame(best_Gamma, index=self.char_names, columns=self.factor_names)
        Lambda = pd.DataFrame(Lambda_arr, index=self.macro_names, columns=self.factor_names)
        factors = pd.DataFrame(best_factors, index=self.factor_names, columns=self.times)

        return Gamma, Lambda, factors

    def _update_factors(self, Gamma: pd.DataFrame,
                        Lambda: pd.DataFrame) -> pd.DataFrame:
        """
        Update factors given Gamma and Lambda.

        With normalized objective:
        Q = (1/N) ||r_t - Z_t Gamma f_t||^2 + (alpha/K) ||f_t - Lambda' m_t||^2

        Solution:
        f_t = (Gamma' W_t Gamma + (alpha/K) * I)^{-1}
              * (Gamma' X_t + (alpha/K) * Lambda' m_t)
        """
        factors_new = pd.DataFrame(
            index=self.factor_names,
            columns=self.times,
            dtype=float
        )

        Gamma_arr = Gamma.values
        Lambda_arr = Lambda.values
        alpha_normalized = self.alpha / self.K  # Normalize by K

        for t in self.times:
            # Terms from cross-sectional equation
            GWG = Gamma.T @ self._W[t] @ Gamma  # K x K
            GX = Gamma.T @ self._X[t]           # K x 1

            # Terms from macro equation (with normalized alpha)
            m_t = self._macro[t, :]
            macro_rhs = alpha_normalized * Lambda_arr.T @ m_t  # K

            # Combined system
            lhs = GWG.values + alpha_normalized * np.eye(self.K)
            rhs = GX.values + macro_rhs

            factors_new[t], *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

        return factors_new

    def _update_gamma(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Update Gamma given factors using vectorized Kronecker product approach.
        Same as ALSIPCA but uses only cross-sectional terms (macro doesn't affect Gamma).
        """
        vec_length = self.L * self.K
        numerator = np.zeros(vec_length)
        denominator = np.zeros((vec_length, vec_length))

        for t in self.times:
            F_t = factors[t].values  # K
            X_t = self._X[t].values  # L
            W_t = self._W[t].values  # L x L
            n_t = self._N_valid[t]

            FF = np.outer(F_t, F_t)  # K x K
            numerator += np.kron(X_t, F_t) * n_t
            denominator += np.kron(W_t, FF) * n_t

        # Regularize and solve
        denominator += 1e-8 * np.eye(vec_length)
        gamma_vec, *_ = np.linalg.lstsq(denominator, numerator, rcond=None)

        # Reshape and orthonormalize
        Gamma_arr = gamma_vec.reshape((self.L, self.K))
        Q, _ = np.linalg.qr(Gamma_arr)
        Gamma_arr = Q[:, :self.K]

        return pd.DataFrame(Gamma_arr, index=self.char_names, columns=self.factor_names)

    def _update_lambda(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Update Lambda given factors.

        Lambda' = (sum_t f_t m_t') (sum_t m_t m_t')^{-1}
        """
        factors_arr = factors.values  # K x T

        # sum_t f_t m_t' = F @ M where F is K x T, M is T x R
        sum_fm = factors_arr @ self._macro  # K x R

        # Solve (M'M) Lambda = (sum_fm)'  =>  Lambda = (M'M)^-1 (sum_fm)'
        Lambda_T = sla.solve(self._M_sum + 1e-8 * np.eye(self.R),
                             sum_fm.T, assume_a='sym')  # R x K

        return pd.DataFrame(Lambda_T, index=self.macro_names, columns=self.factor_names)

    def _compute_objective(self, Gamma: pd.DataFrame, Lambda: pd.DataFrame,
                          factors: pd.DataFrame) -> float:
        """Compute objective function."""
        return self.loss_fct(Gamma.values, Lambda.values, factors.values, self._data)

    def predict(self, Z: np.ndarray, M: np.ndarray = None,
                factors: np.ndarray = None) -> np.ndarray:
        """
        Predict returns.

        Parameters
        ----------
        Z : np.ndarray (T_new, N, L) or (N, L)
            Characteristics
        M : np.ndarray (T_new, R), optional
            Macro variables (used to predict factors if factors not provided)
        factors : np.ndarray (K, T_new) or (K,), optional
            Factor realizations (if known)

        Returns
        -------
        np.ndarray
            Predicted returns
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        Gamma_arr = self.Gamma.values
        Lambda_arr = self.Lambda.values

        # Handle single time period
        if Z.ndim == 2:
            Z = Z[np.newaxis, :, :]
            single_period = True
        else:
            single_period = False

        T_new = Z.shape[0]
        N = Z.shape[1]

        # Get factors
        if factors is not None:
            if factors.ndim == 1:
                factors = factors[:, np.newaxis]
            f_arr = factors
        elif M is not None:
            # Predict factors from macro: f_t = Lambda' m_t
            f_arr = (M @ Lambda_arr).T  # K x T_new
        else:
            raise ValueError("Either factors or M must be provided")

        # Predict returns: r_t = Z_t @ Gamma @ f_t
        preds = np.zeros((T_new, N))
        for t in range(T_new):
            preds[t, :] = Z[t, :, :] @ Gamma_arr @ f_arr[:, t]

        if single_period:
            return preds[0, :]
        return preds

    def transform(self, Z: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Get factor predictions from macro variables.

        Parameters
        ----------
        Z : np.ndarray (T_new, N, L)
            Characteristics (not used, for API consistency)
        M : np.ndarray (T_new, R)
            Macro variables

        Returns
        -------
        np.ndarray (T_new, K)
            Predicted factors
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        return M @ self.Lambda.values

    def score(self, data: list) -> float:
        """
        Compute R-squared on given data.

        Parameters
        ----------
        data : list
            [rets, Z, M]

        Returns
        -------
        float
            R-squared value
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        rets, Z, M = data

        # Predict factors from macro
        pred_factors = (M @ self.Lambda.values).T  # K x T

        # Predict returns
        preds = self.predict(Z, factors=pred_factors)

        # R-squared
        ss_res = np.sum((rets - preds) ** 2)
        ss_tot = np.sum((rets - np.mean(rets)) ** 2)

        return 1 - ss_res / ss_tot

    def get_results(self) -> Dict:
        """
        Get estimation results.

        Returns
        -------
        Dict containing:
            - 'Gamma': characteristic loadings (L x K)
            - 'Lambda': macro-factor map (R x K)
            - 'factors': factor realizations (K x T)
            - 'objective_history': objective values per iteration
            - 'n_iterations': iterations until convergence
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        return {
            'Gamma': self.Gamma,
            'Lambda': self.Lambda,
            'factors': self.factors,
            'objective_history': np.array(self.objective_history),
            'n_iterations': self.n_iterations
        }

    def factor_macro_r2(self) -> np.ndarray:
        """
        Compute R-squared of macro equation for each factor.

        Returns
        -------
        np.ndarray (K,)
            R-squared for each factor
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        factors_arr = self.factors.values  # K x T
        Lambda_arr = self.Lambda.values    # R x K

        r2_values = np.zeros(self.K)

        for k in range(self.K):
            f_k = factors_arr[k, :]  # T
            pred_f_k = self._macro @ Lambda_arr[:, k]  # T

            ss_res = np.sum((f_k - pred_f_k) ** 2)
            ss_tot = np.sum((f_k - np.mean(f_k)) ** 2)

            r2_values[k] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return r2_values


# =============================================================================
# Example usage and testing
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GIPCA Test")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)

    T = 100   # Time periods
    N = 50    # Assets
    L = 10    # Characteristics
    K = 3     # Factors
    R = 5     # Macro variables

    # True parameters
    true_Gamma = np.random.randn(L, K)
    true_Gamma, _ = np.linalg.qr(true_Gamma)
    true_Gamma = true_Gamma[:, :K]

    true_Lambda = np.random.randn(R, K) * 0.5

    # Generate macro variables (persistent)
    M = np.cumsum(np.random.randn(T, R) * 0.1, axis=0)

    # Generate factors: f_t = Lambda' m_t + noise
    noise_f = np.random.randn(T, K) * 0.3
    true_factors = M @ true_Lambda + noise_f  # T x K

    # Generate characteristics
    Z = np.random.randn(T, N, L) * 0.5
    Z[:, :, 0] = 1  # Intercept

    # Generate returns: r_{i,t} = c_{i,t}' Gamma f_t + noise
    rets = np.zeros((T, N))
    for t in range(T):
        loadings = Z[t, :, :] @ true_Gamma  # N x K
        rets[t, :] = loadings @ true_factors[t, :] + np.random.randn(N) * 0.5

    # Fit GIPCA
    print(f"\nData: T={T}, N={N}, L={L}, K={K}, R={R}")
    print(f"Alpha = 1.0")
    print("-" * 60)

    model = GIPCA(
        num_assets=N,
        num_fact=K,
        num_charact=L,
        num_macro=R,
        win_len=T,
        alpha=1.0
    )

    data = [rets, Z, M]
    Gamma_est, history = model.fit(data, max_iter=500, tol=1e-6, verbose=True, seed=42)

    # Results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    results = model.get_results()
    print(f"\nGamma shape: {results['Gamma'].shape}")
    print(f"Lambda shape: {results['Lambda'].shape}")
    print(f"Factors shape: {results['factors'].shape}")

    # R-squared
    r2 = model.score(data)
    print(f"\nOverall R²: {r2:.4f}")

    # Macro R² per factor
    macro_r2 = model.factor_macro_r2()
    print(f"Macro R² per factor: {np.round(macro_r2, 4)}")

    # Subspace comparison with true Gamma
    Q1, _ = np.linalg.qr(true_Gamma)
    Q2, _ = np.linalg.qr(Gamma_est)
    _, s, _ = np.linalg.svd(Q1.T @ Q2)
    principal_angles = np.arccos(np.clip(s, -1, 1))

    print(f"\nSubspace comparison (True vs Estimated Gamma):")
    print(f"  Principal angles (degrees): {np.round(np.degrees(principal_angles), 2)}")
    print(f"  Grassmann distance: {np.linalg.norm(principal_angles):.4f}")

    # Factor correlation
    est_factors = results['factors'].values  # K x T
    factor_corr = np.corrcoef(est_factors, true_factors.T)[:K, K:]
    print(f"\nFactor correlations with true factors:")
    print(np.round(np.abs(factor_corr), 3))

    print("\n" + "=" * 60)
    print("GIPCA test complete!")
    print("=" * 60)
