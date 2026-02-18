"""
ALS GIPCA — Generalized Instrumented PCA with Latent Factors

Full model with latent factor residuals:
    r_t = Z_t Γ f_t + ε_t
    f_t = f⁰_t + Λ' m_t

where f⁰_t is a free/latent factor component estimated from data.

Objective (profiled, no penalty):
    min_{Γ, Λ, f}  (1/N) Σ_t ‖r_t − Z_t Γ f_t‖²    s.t.  Γ'Γ = I_K

ALS alternates between:
    f-step : Cross-sectional OLS for each t: f_t = (Λ_t'Λ_t)⁻¹ Λ_t' r_t
    Γ-step : Kronecker normal equations with f_t, SVD orthonormalization
    Λ-step : Time-series OLS: regress f_t on m_t

Intercept convention:
    The caller should prepend a column of ones to the macro matrix M so that
    Λ' m̃_t = λ₀ + Λ_macro' m_t.  All code below treats the intercept as an
    ordinary macro variable (the first column of M).

Relation to other methods:
    - Hard GIPCA (α → ∞): Forces f⁰_t = 0, so f_t = Λ' m_t exactly
    - Soft GIPCA (finite α): Penalizes ||f_t - Λ' m_t||² with weight α
    - This version (α = 0): No penalty, f⁰_t is free
"""

import numpy as np
import pandas as pd
import scipy.linalg as sla
from typing import Dict, Tuple, Optional


class HardGIPCA:
    """
    ALS GIPCA with Latent Factors.

    Full model: f_t = f⁰_t + Λ' m_t, where f⁰_t is a free latent component.
    Estimates Γ, Λ, and the full factor series f_t via ALS.

    Parameters
    ----------
    num_assets : int
        Number of assets (N).
    num_fact : int
        Number of factors (K).
    num_charact : int
        Number of characteristics (L).
    num_macro : int
        Number of macro variables (R), including the intercept if prepended.
    win_len : int
        Number of time periods (T).
    ridge : float, default 1e-8
        Ridge regularisation added to Kronecker normal matrices for stability.
    """

    def __init__(self, num_assets: int, num_fact: int, num_charact: int,
                 num_macro: int, win_len: int, ridge: float = 1e-8):
        self.num_assets = num_assets  # N
        self.K = num_fact             # K
        self.L = num_charact          # L
        self.R = num_macro            # R
        self.T = win_len              # T
        self.ridge = ridge

        # Parameters to be estimated
        self.Gamma = None    # L × K  (DataFrame)
        self.Lambda = None   # R × K  (DataFrame)
        self.factors = None  # K × T  (DataFrame) — full factors f_t

        self._fitted = False
        self.objective_history = []
        self.n_iterations = 0

    # ──────────────────────────────────────────────────────────
    #  Loss function
    # ──────────────────────────────────────────────────────────
    def loss_fct(self, Gamma: np.ndarray, factors: np.ndarray,
                 data: list) -> float:
        """
        GIPCA loss with explicit factors:  (1/T) Σ_t ‖r_t − Z_t Γ f_t‖².

        Parameters
        ----------
        Gamma : np.ndarray (L × K)
        factors : np.ndarray (K × T)
        data : list  [rets (T,N), Z (T,N,L), M (T,R)]

        Returns
        -------
        float
        """
        rets, Z, M = data
        T, _ = rets.shape

        obj = 0.0
        for t in range(T):
            f_t = factors[:, t]                       # K
            pred = Z[t, :, :] @ Gamma @ f_t           # N
            obj += np.sum((rets[t, :] - pred) ** 2)

        return obj / T

    # ──────────────────────────────────────────────────────────
    #  Fit via ALS
    # ──────────────────────────────────────────────────────────
    def fit(self, data: list, max_iter: int = 1000, min_iter: int = 100,
            tol: float = 1e-6, verbose: bool = True, seed: int = None,
            init_Gamma: np.ndarray = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the GIPCA model via ALS (f-step, Γ-step, Λ-step).

        Parameters
        ----------
        data : list
            [rets (T,N), Z (T,N,L), M (T,R)]
        max_iter : int
            Maximum number of iterations.
        min_iter : int
            Minimum iterations before convergence check (default 100).
        tol : float
            Convergence tolerance (relative objective change).
        verbose : bool
        seed : int, optional
        init_Gamma : np.ndarray (L × K), optional
            Initial Gamma matrix. If provided, skips best-of-population search.
            Must be orthonormal (Γ'Γ = I_K).

        Returns
        -------
        Gamma_arr : np.ndarray (L × K)
        objective_history : np.ndarray
        """
        rets, Z, M = data

        # ── Validate ──
        assert rets.shape == (self.T, self.num_assets), \
            f"rets shape {rets.shape} != expected {(self.T, self.num_assets)}"
        assert Z.shape == (self.T, self.num_assets, self.L), \
            f"Z shape {Z.shape} != expected {(self.T, self.num_assets, self.L)}"
        assert M.shape == (self.T, self.R), \
            f"M shape {M.shape} != expected {(self.T, self.R)}"

        # ── Store data ──
        self._returns = rets
        self._characteristics = Z
        self._macro = M
        self._data = data

        # ── Pre-compute managed portfolios ──
        self._X, self._W, self._N_valid = self._compute_managed_portfolios(rets, Z)

        # ── Pre-compute macro moments ──
        self._M_sum = M.T @ M  # R × R

        # ── Names ──
        self.times = list(range(self.T))
        self.char_names = [f'char_{l}' for l in range(self.L)]
        self.factor_names = [f'Factor_{k+1}' for k in range(self.K)]
        self.macro_names = [f'macro_{r}' for r in range(self.R)]

        # ── Reset history ──
        self.objective_history = []

        # ── Initialise ──
        if init_Gamma is not None:
            # Use provided initialization
            Gamma_arr = np.asarray(init_Gamma)
            assert Gamma_arr.shape == (self.L, self.K), \
                f"init_Gamma shape {Gamma_arr.shape} != expected {(self.L, self.K)}"
            Gamma_old = pd.DataFrame(Gamma_arr,
                                     index=self.char_names, columns=self.factor_names)
            factors_old = self._update_factors(Gamma_old)
            Lambda_old = self._update_lambda(factors_old)
            if verbose:
                print("Using provided init_Gamma")
        else:
            # Best-of-population search
            Gamma_old, Lambda_old, factors_old = self._initialize(seed=seed)

        obj_init = self._compute_objective(Gamma_old, factors_old)
        self.objective_history.append(obj_init)
        if verbose:
            print(f"Iteration    0: Objective = {obj_init:.6f}")

        # ── ALS loop ──
        obj_old = obj_init
        for iteration in range(max_iter):
            # Step 1 — f-step: update factors given Gamma (cross-sectional OLS)
            factors_new = self._update_factors(Gamma_old)

            # Step 2 — Γ-step: update Gamma given factors (with Procrustes alignment)
            Gamma_new = self._update_gamma(factors_new, Gamma_old=Gamma_old)

            # Step 3 — Λ-step: update Lambda given factors (time-series OLS)
            Lambda_new = self._update_lambda(factors_new)

            # Objective
            obj = self._compute_objective(Gamma_new, factors_new)
            self.objective_history.append(obj)

            # Convergence: use RELATIVE objective change
            obj_change = abs(obj_old - obj) / (abs(obj_old) + 1e-10)

            # Also track parameter change for diagnostics
            gamma_change = np.max(np.abs(Gamma_new.values - Gamma_old.values))
            factor_change = np.max(np.abs(factors_new.values - factors_old.values))
            max_change = max(gamma_change, factor_change)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:4d}: Objective = {obj:.6f}, "
                      f"Obj change = {obj_change:.2e}, Param change = {max_change:.2e}")

            # Check convergence (only after min_iter iterations)
            if iteration + 1 >= min_iter and obj_change < tol:
                if verbose:
                    print(f"\nConverged after {iteration + 1} iterations")
                break

            Gamma_old = Gamma_new
            Lambda_old = Lambda_new
            factors_old = factors_new
            obj_old = obj

        # ── Final summary ──
        if verbose and iteration == max_iter - 1:
            print(f"\nCompleted {max_iter} iterations (did not converge)")
        if verbose:
            print(f"Initial objective: {self.objective_history[0]:.6f}")
            print(f"Final objective:   {obj:.6f}")
            print(f"Reduction:         "
                  f"{(1 - obj / self.objective_history[0]) * 100:.2f}%")

        # ── Store results ──
        self.Gamma = Gamma_new
        self.Lambda = Lambda_new
        self.factors = factors_new
        self._fitted = True
        self.n_iterations = iteration + 1

        # ── Identification: decompose f = Delta*mu + f0, check f0'mu = 0 ──
        f0, Delta = self._decompose_factors(factors_new, Lambda_new)
        self.f0 = f0          # K × T  (residual factors)
        self.Delta = Delta    # K × R  (macro loading)

        # Identification check: || f0^T @ mu || should be ~0
        ident_norm = np.linalg.norm(f0 @ self._macro)  # f0 is K×T, mu is T×R
        if verbose:
            print(f"Identification check: || f0^T @ mu || = {ident_norm:.2e}")

        return self.Gamma.values, np.array(self.objective_history)

    # ──────────────────────────────────────────────────────────
    #  Managed portfolios (same as soft GIPCA)
    # ──────────────────────────────────────────────────────────
    def _compute_managed_portfolios(self, rets, Z):
        """X_t = Z_t' r_t / N  and  W_t = Z_t' Z_t / N."""
        T, N, L = Z.shape
        char_names = [f'char_{l}' for l in range(L)]

        X = pd.DataFrame(index=char_names, columns=list(range(T)), dtype=float)
        W = {}
        N_valid = pd.Series(index=list(range(T)), dtype=float)

        for t in range(T):
            Z_t = Z[t, :, :]
            r_t = rets[t, :]
            X[t] = (Z_t.T @ r_t) / N
            W[t] = pd.DataFrame((Z_t.T @ Z_t) / N,
                                index=char_names, columns=char_names)
            N_valid[t] = N

        return X, W, N_valid

    # ──────────────────────────────────────────────────────────
    #  IPCA profiled loss (same as grassmanian_gipca_fully_identified)
    # ──────────────────────────────────────────────────────────
    def _ipca_profiled_loss(self, Gamma: np.ndarray) -> float:
        """
        IPCA profiled loss (same as GrassmannManifoldGIPCAEstimator.loss_fct):

            L(Γ) = (1/T) Σ_t min_f ‖r_t − Z_t Γ f‖²

        For each t, the factor f_t is profiled out via cross-sectional OLS.
        """
        rets, Z, _ = self._data

        obj = 0.0
        for t in range(self.T):
            Lambda_t = Z[t] @ Gamma              # (N, K)
            f_t, *_ = np.linalg.lstsq(Lambda_t, rets[t], rcond=None)
            resid = rets[t] - Lambda_t @ f_t      # (N,)
            obj += float(resid @ resid)

        return obj / self.T

    # ──────────────────────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────────────────────
    def _initialize(self, seed=None):
        """
        Best-of-population initialisation on the Grassmannian (aligned with de_gipca).

        Generates pop_size random orthonormal Gamma matrices via QR decomposition,
        evaluates the PROFILED LOSS (same as de_gipca) for each, and keeps the best.

        This matches de_gipca's approach exactly:
        - Same pop_size = 5 * L * K
        - Same random initialization (uniform(-1, 1) then QR)
        - Same loss function (profiled over Delta)
        """
        # Note: Does NOT reset the random seed here to match de_gipca's behavior,
        # which uses the random state after data generation.

        dim = self.L * self.K
        pop_size = 5 * dim  # Same as de_gipca: pop_size = 5 * self.dim

        best_obj = np.inf
        best_Gamma = None

        for _ in range(pop_size):
            # Random orthonormal Gamma (same as de_gipca's population init)
            A = np.random.uniform(-1.0, 1.0, size=(self.L, self.K))
            Q, _ = np.linalg.qr(A)
            Gamma_cand = Q[:, :self.K]

            # Use PROFILED LOSS (same as de_gipca)
            obj = self._ipca_profiled_loss(Gamma_cand)
            if obj < best_obj:
                best_obj = obj
                best_Gamma = Gamma_cand

        # Now compute factors and Lambda for the best Gamma
        M_sum = self._macro.T @ self._macro  # R × R

        factors_arr = np.zeros((self.K, self.T))
        for t in range(self.T):
            A_t = self._characteristics[t, :, :] @ best_Gamma
            factors_arr[:, t], *_ = np.linalg.lstsq(
                A_t, self._returns[t, :], rcond=None)

        sum_fm = factors_arr @ self._macro          # K × R
        Lambda_arr = sla.solve(
            M_sum + 1e-8 * np.eye(self.R),
            sum_fm.T, assume_a='sym')               # R × K

        Gamma = pd.DataFrame(best_Gamma,
                             index=self.char_names, columns=self.factor_names)
        Lambda = pd.DataFrame(Lambda_arr,
                              index=self.macro_names, columns=self.factor_names)
        factors = pd.DataFrame(factors_arr,
                               index=self.factor_names, columns=self.times)

        print(f"Init IPCA profiled loss: {best_obj:.6f}")
        return Gamma, Lambda, factors

    # ──────────────────────────────────────────────────────────
    #  Procrustes alignment (resolves rotation ambiguity)
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _procrustes_align(Gamma_new: np.ndarray, Gamma_old: np.ndarray) -> np.ndarray:
        """
        Align Gamma_new to Gamma_old via orthogonal Procrustes.

        Finds R = argmin_R ||Gamma_new @ R - Gamma_old||_F  s.t. R'R = I
        Returns Gamma_new @ R.
        """
        # Solve Procrustes: R = V @ U' where Gamma_old' @ Gamma_new = U @ S @ V'
        M = Gamma_old.T @ Gamma_new  # K × K
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt  # optimal rotation
        return Gamma_new @ R

    # ──────────────────────────────────────────────────────────
    #  f-step: Update factors given Gamma (cross-sectional OLS)
    # ──────────────────────────────────────────────────────────
    def _update_factors(self, Gamma: pd.DataFrame) -> pd.DataFrame:
        """
        Update factors given Gamma via cross-sectional OLS for each t.

        f_t = (Λ_t' Λ_t)^{-1} Λ_t' r_t  where Λ_t = Z_t @ Γ
        """
        Gamma_arr = Gamma.values  # L × K
        factors_arr = np.zeros((self.K, self.T))

        for t in self.times:
            Z_t = self._characteristics[t, :, :]  # N × L
            r_t = self._returns[t, :]              # N
            Lambda_t = Z_t @ Gamma_arr             # N × K

            # OLS: f_t = (Λ_t' Λ_t)^{-1} Λ_t' r_t
            factors_arr[:, t], *_ = np.linalg.lstsq(Lambda_t, r_t, rcond=None)

        return pd.DataFrame(factors_arr,
                            index=self.factor_names, columns=self.times)

    # ──────────────────────────────────────────────────────────
    #  Γ-step: Update Gamma given factors
    # ──────────────────────────────────────────────────────────
    def _update_gamma(self, factors: pd.DataFrame,
                      Gamma_old: pd.DataFrame = None) -> pd.DataFrame:
        """
        Update Gamma given factors.

        Row-major Kronecker normal equation (LK × LK):
            [Σ_t n_t · kron(W_t, f_t f_t')] vec_C(Γ) = Σ_t n_t · kron(X_t, f_t)

        Then orthonormalise via SVD:  Γ_unc = UΔV' → Γ = UV'.
        Finally, align with previous Gamma via Procrustes to avoid cycling.
        """
        factors_arr = factors.values  # K × T

        vec_length = self.L * self.K
        numerator = np.zeros(vec_length)
        denominator = np.zeros((vec_length, vec_length))

        for t in self.times:
            f_t = factors_arr[:, t]             # K

            X_t = self._X[t].values             # L
            W_t = self._W[t].values             # L × L
            n_t = self._N_valid[t]

            ff = np.outer(f_t, f_t)             # K × K
            numerator += np.kron(X_t, f_t) * n_t
            denominator += np.kron(W_t, ff) * n_t

        # Regularise and solve
        denominator += self.ridge * np.eye(vec_length)
        gamma_vec, *_ = np.linalg.lstsq(denominator, numerator, rcond=None)

        # Reshape and orthonormalise via SVD  (Γ_unc = UΔV' → Γ = UV')
        Gamma_unc = gamma_vec.reshape((self.L, self.K))
        U, _, Vt = np.linalg.svd(Gamma_unc, full_matrices=False)
        Gamma_arr = U @ Vt  # L × K  with  Γ'Γ = I_K

        # Procrustes alignment to avoid rotation cycling
        if Gamma_old is not None:
            Gamma_arr = self._procrustes_align(Gamma_arr, Gamma_old.values)

        return pd.DataFrame(Gamma_arr,
                            index=self.char_names, columns=self.factor_names)

    # ──────────────────────────────────────────────────────────
    #  Λ-step: Update Lambda given factors (time-series OLS)
    # ──────────────────────────────────────────────────────────
    def _update_lambda(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Update Lambda given factors via time-series OLS.

        Regress f_t on m_t:  Λ = (M'M)^{-1} M' F'

        where F is K × T (factors) and M is T × R (macro).
        """
        factors_arr = factors.values  # K × T

        # OLS: Λ = (M'M)^{-1} M' F'
        # F @ M gives K × R, we want R × K
        sum_fm = factors_arr @ self._macro          # K × R
        Lambda_arr = sla.solve(
            self._M_sum + self.ridge * np.eye(self.R),
            sum_fm.T, assume_a='sym')               # R × K

        return pd.DataFrame(Lambda_arr,
                            index=self.macro_names, columns=self.factor_names)

    # ──────────────────────────────────────────────────────────
    #  Identification: decompose factors into Delta*mu + f0
    # ──────────────────────────────────────────────────────────
    def _decompose_factors(self, factors: pd.DataFrame, Lambda: pd.DataFrame):
        """
        Decompose estimated factors into macro-explained and residual parts.

        Given f_t and Lambda (= Delta in the GIPCA notation):
            f0_t = f_t - Lambda' mu_t

        By construction of the OLS Lambda-step, f0^T @ mu = 0 (identified).

        Returns
        -------
        f0 : np.ndarray (K × T) — residual factors
        Delta : np.ndarray (K × R) — macro loading (= Lambda^T)
        """
        factors_arr = factors.values       # K × T
        Lambda_arr = Lambda.values         # R × K

        # Macro-explained part: Lambda' @ mu_t for each t
        # Lambda_arr is R × K, mu is T × R → mu @ Lambda_arr = T × K → transpose = K × T
        macro_part = (self._macro @ Lambda_arr).T  # K × T

        # Residual factors
        f0 = factors_arr - macro_part  # K × T

        # Delta = Lambda^T (K × R)
        Delta = Lambda_arr.T

        return f0, Delta

    # ──────────────────────────────────────────────────────────
    #  Objective wrapper
    # ──────────────────────────────────────────────────────────
    def _compute_objective(self, Gamma, factors):
        """Use profiled loss (same as de_gipca) for fair comparison."""
        return self._ipca_profiled_loss(Gamma.values)

    # ──────────────────────────────────────────────────────────
    #  Predict
    # ──────────────────────────────────────────────────────────
    def predict(self, Z: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Predict returns:  r̂_t = Z_t Γ Λ' m_t.

        Parameters
        ----------
        Z : np.ndarray  (T_new, N, L) or (N, L)
        M : np.ndarray  (T_new, R) or (R,)

        Returns
        -------
        np.ndarray  (T_new, N) or (N,)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        Gamma_arr = self.Gamma.values
        Lambda_arr = self.Lambda.values

        single = Z.ndim == 2
        if single:
            Z = Z[np.newaxis, :, :]
            M = M[np.newaxis, :]

        T_new = Z.shape[0]
        N = Z.shape[1]
        preds = np.zeros((T_new, N))

        for t in range(T_new):
            g_t = Lambda_arr.T @ M[t, :]
            preds[t, :] = Z[t, :, :] @ Gamma_arr @ g_t

        return preds[0, :] if single else preds

    # ──────────────────────────────────────────────────────────
    #  Transform (macro → factors)
    # ──────────────────────────────────────────────────────────
    def transform(self, M: np.ndarray) -> np.ndarray:
        """
        Macro-implied factors:  g_t = Λ' m_t.

        Parameters
        ----------
        M : np.ndarray (T_new, R)

        Returns
        -------
        np.ndarray (T_new, K)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        return M @ self.Lambda.values

    # ──────────────────────────────────────────────────────────
    #  Score (R²)
    # ──────────────────────────────────────────────────────────
    def score(self, data: list) -> float:
        """R² on given data (using macro-implied factors)."""
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        rets, Z, M = data
        preds = self.predict(Z, M)

        ss_res = np.sum((rets - preds) ** 2)
        ss_tot = np.sum((rets - np.mean(rets)) ** 2)
        return 1 - ss_res / ss_tot

    # ──────────────────────────────────────────────────────────
    #  Results dict
    # ──────────────────────────────────────────────────────────
    def get_results(self) -> Dict:
        """Return estimation results."""
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        return {
            'Gamma': self.Gamma,
            'Lambda': self.Lambda,
            'factors': self.factors,  # Estimated factors (includes latent component)
            'objective_history': np.array(self.objective_history),
            'n_iterations': self.n_iterations,
        }

    # ──────────────────────────────────────────────────────────
    #  Macro R² per factor
    # ──────────────────────────────────────────────────────────
    def factor_macro_r2(self) -> np.ndarray:
        """
        Compute R² of macro regression for each factor.

        For each factor k:  R²_k = 1 - SS_res / SS_tot
        where SS_res = Σ_t (f_kt - Λ'_k m_t)² and SS_tot = Σ_t (f_kt - mean(f_k))²
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        factors_arr = self.factors.values    # K × T
        Lambda_arr = self.Lambda.values      # R × K

        r2_values = np.zeros(self.K)
        for k in range(self.K):
            f_k = factors_arr[k, :]                    # T
            pred_f_k = self._macro @ Lambda_arr[:, k]  # T

            ss_res = np.sum((f_k - pred_f_k) ** 2)
            ss_tot = np.sum((f_k - np.mean(f_k)) ** 2)
            r2_values[k] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return r2_values


def generate_random_gamma(L: int, K: int, seed: int = None) -> np.ndarray:
    """
    Generate a random orthonormal Gamma matrix on the Grassmannian.

    This function can be used to generate a shared initialization for
    both hard_gipca (ALS) and de_gipca (DE) for fair comparison.

    Parameters
    ----------
    L : int
        Number of characteristics (rows).
    K : int
        Number of factors (columns).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Gamma : np.ndarray (L × K)
        Orthonormal matrix with Γ'Γ = I_K.
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.uniform(-1.0, 1.0, size=(L, K))
    Q, _ = np.linalg.qr(A)
    return Q[:, :K]


def generate_gipca_data(
        T: int = 252,
        N: int = 500,
        m: int = 20,          # TOTAL characteristics (includes intercept if include_intercept=True)
        k: int = 5,           # factors
        num_macro: int = 3,   # l (macro dimension)
        seed: int = 124,

        # Factor residual dynamics: f0_t (k,)
        f0_rho: float = 0.9,
        sigma_f0: float = 0.5,

        # Macro dynamics: mu_t (num_macro,)
        mu_rho: float = 0.8,          # AR(1) persistence (componentwise)
        sigma_mu: float = 0.5,        # innovation scale
        mu_corr: float = 0.3,         # contemporaneous correlation across macro dims (Toeplitz)
        mu_drift_scale: float = 0.00, # optional random-walk drift in macro means

        # True Delta_* scale
        delta_scale: float = 0.5,

        # Characteristic structure
        z_rho: float = 0.4,
        z_scale: float = 1.0,

        # Returns noise
        heavy_tail_df: float = 5.0,   # Student-t dof; set to np.inf for Gaussian
        sigma_eps_base: float = 0.5,
        hetero_strength: float = 0.5,

        # Missingness
        missing_prob: float = 0.05,
        missing_mode: str = "mcAR",   # "mcAR" or "tail"
        impute: str = "zero",         # "zero" or "mean"

        # Intercept
        include_intercept: bool = True,

        # Optional: time-varying characteristic drift
        z_drift_scale: float = 0.02,

        # Identification
        enforce_identification: bool = True,
        ident_ridge: float = 1e-12,
):
    """
    Generates synthetic GIPCA-like data:
      - True W_* on Grassmann (orthonormal columns) in R^{m_eff x k}
      - Cross-sectionally correlated characteristics Z_t (N x m_eff)
      - Macro series mu_t (T x num_macro) with AR(1) dynamics + correlated innovations
      - True Delta_* (k x num_macro)
      - Factor residuals f0_t (T x k) with AR(1) dynamics
      - Returns: r_t = (Z_t W_*) ( f0_t + Delta_* mu_t ) + eps_t
      - Heteroskedastic idiosyncratic noise across assets
      - Optional heavy-tailed noise (Student t)
      - Missingness in characteristics + simple imputation

    Returns:
      data = (rets, Z, mu) where
         rets: (T, N)
         Z:    (T, N, m_eff)
         mu:   (T, num_macro)
      truth dict with W_star, Delta_star, f0, f_full, etc.
    """
    rng = np.random.default_rng(seed)

    # ----- handle intercept in Z -----
    if include_intercept:
        if m < 2:
            raise ValueError("Need m>=2 if include_intercept=True (1 intercept + at least 1 feature).")
        m_eff = m
        m_core = m - 1
    else:
        m_eff = m
        m_core = m

    # ----- True W_* (m_eff x k), orthonormal columns -----
    A = rng.normal(size=(m_eff, k))
    W_star, _ = np.linalg.qr(A)  # Grassmann representative

    # ----- True Delta_* (k x num_macro) -----
    Delta_star = delta_scale * rng.normal(size=(k, num_macro))

    # ----- Feature covariance for Z core (Toeplitz) -----
    if m_core > 0:
        idx = np.arange(m_core)
        Sigma_z = z_rho ** np.abs(idx[:, None] - idx[None, :])
        Sigma_z = Sigma_z + 1e-10 * np.eye(m_core)
        Lz = np.linalg.cholesky(Sigma_z)
    else:
        Lz = None

    # ----- Time-varying drift in characteristic means (core only) -----
    mean_z_t = np.zeros((T, m_core))
    for t in range(1, T):
        mean_z_t[t] = mean_z_t[t - 1] + rng.normal(scale=z_drift_scale, size=m_core)

    # ----- Generate Z (T, N, m_eff) -----
    Z = np.empty((T, N, m_eff), dtype=float)
    for t in range(T):
        if m_core > 0:
            Z_core = (rng.normal(size=(N, m_core)) @ Lz.T) * z_scale
            Z_core = Z_core + mean_z_t[t]
        else:
            Z_core = np.empty((N, 0), dtype=float)

        if include_intercept:
            Z[t, :, 0] = 1.0
            Z[t, :, 1:] = Z_core
        else:
            Z[t, :, :] = Z_core

    # ----- Missingness mask for Z (core columns only; never intercept) -----
    mask = np.ones_like(Z, dtype=bool)

    if missing_prob > 0.0 and m_core > 0:
        core_slice = slice(1, m_eff) if include_intercept else slice(0, m_eff)
        Z_core_all = Z[:, :, core_slice]

        if missing_mode.lower() == "mcar":
            miss = rng.uniform(size=Z_core_all.shape) < missing_prob
        elif missing_mode.lower() == "tail":
            scaled_abs = np.abs(Z_core_all) / (np.std(Z_core_all) + 1e-12)
            p = missing_prob * (1.0 + 0.5 * scaled_abs)
            p = np.clip(p, 0.0, 0.5)
            miss = rng.uniform(size=Z_core_all.shape) < p
        else:
            raise ValueError("missing_mode must be 'mcAR' or 'tail'")

        Z[:, :, core_slice][miss] = np.nan
        mask[:, :, core_slice][miss] = False

        if impute.lower() == "zero":
            Z[:, :, core_slice] = np.nan_to_num(Z[:, :, core_slice], nan=0.0)
        elif impute.lower() == "mean":
            Z_imp = Z[:, :, core_slice]
            for t in range(T):
                col_means = np.nanmean(Z_imp[t], axis=0)
                inds = np.where(~np.isfinite(Z_imp[t]))
                if inds[0].size > 0:
                    Z_imp[t][inds] = col_means[inds[1]]
            Z[:, :, core_slice] = Z_imp
        else:
            raise ValueError("impute must be 'zero' or 'mean'")

    # ----- Macro innovations covariance (Toeplitz) -----
    if num_macro < 1:
        raise ValueError("num_macro must be >= 1.")
    idxm = np.arange(num_macro)
    Sigma_mu = mu_corr ** np.abs(idxm[:, None] - idxm[None, :])
    Sigma_mu = Sigma_mu + 1e-10 * np.eye(num_macro)
    Lmu = np.linalg.cholesky(Sigma_mu)

    # ----- Optional drift in macro means -----
    mean_mu_t = np.zeros((T, num_macro))
    for t in range(1, T):
        mean_mu_t[t] = mean_mu_t[t - 1] + rng.normal(scale=mu_drift_scale, size=num_macro)

    # ----- Generate mu_t (T, num_macro): AR(1) with correlated innovations -----
    mu = np.zeros((T, num_macro), dtype=float)
    mu[0] = mean_mu_t[0] + (rng.normal(size=num_macro) @ Lmu.T) * sigma_mu
    for t in range(1, T):
        innov = (rng.normal(size=num_macro) @ Lmu.T) * sigma_mu
        mu[t] = mean_mu_t[t] + mu_rho * (mu[t - 1] - mean_mu_t[t - 1]) + innov

    # ----- Generate f0_t (T, k): AR(1) -----
    f0 = np.zeros((T, k), dtype=float)
    f0[0] = rng.normal(scale=sigma_f0, size=k)
    for t in range(1, T):
        f0[t] = f0_rho * f0[t - 1] + rng.normal(scale=sigma_f0, size=k)

    # ----- Enforce identification: f0^T mu = 0 -----
    ident_B = None
    if enforce_identification:
        S_mm = mu.T @ mu + ident_ridge * np.eye(num_macro)
        ident_B = (f0.T @ mu) @ np.linalg.inv(S_mm)  # (k, num_macro)
        f0 = f0 - mu @ ident_B.T                       # project out mu component

    # ----- Cross-sectional heteroskedastic idiosyncratic scales -----
    u = rng.normal(size=N)
    sigma_i = sigma_eps_base * np.exp(hetero_strength * u)
    sigma_i = np.clip(sigma_i, 1e-4, np.percentile(sigma_i, 99.5))

    # ----- Generate returns -----
    rets = np.empty((T, N), dtype=float)
    use_t = np.isfinite(heavy_tail_df) and heavy_tail_df < 1e9

    # also record full factors f_t = f0_t + Delta mu_t
    f_full = np.zeros((T, k), dtype=float)

    for t in range(T):
        Lambda_t = Z[t] @ W_star  # (N,k)

        f_full[t] = f0[t] + Delta_star @ mu[t]  # (k,)
        signal = Lambda_t @ f_full[t]           # (N,)

        if use_t:
            df = heavy_tail_df
            if df <= 2:
                raise ValueError("heavy_tail_df must be > 2 for finite variance.")
            eps = rng.standard_t(df, size=N) * np.sqrt((df - 2) / df)
        else:
            eps = rng.normal(size=N)

        rets[t] = signal + sigma_i * eps

    data = [rets, Z, mu]
    truth = {
        "W_star": W_star,
        "Delta_star": Delta_star,
        "mu": mu,
        "f0": f0,
        "f_full": f_full,
        "sigma_i": sigma_i,
        "mask": mask,
        "m_eff": m_eff,
        "include_intercept": include_intercept,
        "ident_B": ident_B,
        "params": {
            "f0_rho": f0_rho,
            "sigma_f0": sigma_f0,
            "mu_rho": mu_rho,
            "sigma_mu": sigma_mu,
            "mu_corr": mu_corr,
            "mu_drift_scale": mu_drift_scale,
            "delta_scale": delta_scale,
            "z_rho": z_rho,
            "z_scale": z_scale,
            "heavy_tail_df": heavy_tail_df,
            "sigma_eps_base": sigma_eps_base,
            "hetero_strength": hetero_strength,
            "missing_prob": missing_prob,
            "missing_mode": missing_mode,
            "impute": impute,
            "z_drift_scale": z_drift_scale,
            "enforce_identification": enforce_identification,
            "ident_ridge": ident_ridge,
        },
    }
    return data, truth




# ______________________________________________________________________________________________________________

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test code
    seed = 6890
    np.random.seed(seed)
    num_assets = 40     # N
    num_fact = 5        # k
    num_charact = 25    # m
    num_macro = 3       # l
    win_len = 21        # T
    max_iter = 2_000

    # Generate hard data
    include_intercept = False
    data, truth = generate_gipca_data(T=win_len, N=num_assets, m=num_charact, k=num_fact, num_macro=num_macro,
                                 include_intercept=include_intercept, seed=seed)

    est = HardGIPCA(num_assets, num_fact, num_charact, num_macro, win_len)
    Gamma_hat, obj_history = est.fit(data, max_iter=max_iter)

    print("Gamma_hat:", Gamma_hat)
    print("Delta_hat:", est.Delta)
    print("f0_hat shape:", est.f0.shape)
    print("Final loss:", obj_history[-1])

    # Quick check of f0^T mu ~ 0
    _, _, mu = data
    orth = np.linalg.norm(est.f0 @ mu)
    print("|| f0_hat^T mu ||:", orth)

    # Convergence history
    plt.figure(figsize=(8, 5))
    plt.plot(obj_history)
    plt.title("Hard GIPCA convergence (profiled IPCA loss)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.show()