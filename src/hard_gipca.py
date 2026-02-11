"""
Hard Macro Factor Model (Method A) — Generalized Instrumented PCA

Factors are fully determined by the macro state: f_t ≡ Λ' m_t.
No free/latent factors are estimated.

Objective (Method A from Missaoui & Lesniewski, 2026):
    min_{Γ, Λ}  (1/N) Σ_t ‖x_t − Z_t Γ (Λ' m_t)‖²    s.t.  Γ'Γ = I_K

ALS alternates between:
    Γ-step : Kronecker normal equations with g_t = Λ'm_t, SVD orthonormalization
    Λ-step : Kronecker normal equations in the projected space (KR × KR system)

Intercept convention:
    The caller should prepend a column of ones to the macro matrix M so that
    Λ' m̃_t = λ₀ + Λ_macro' m_t.  All code below treats the intercept as an
    ordinary macro variable (the first column of M).

Relation to soft GIPCA (Method B):
    This is the α → ∞ limit of the soft objective.

Λ-step derivation (projected Kronecker system)
-----------------------------------------------
With Γ fixed, define Ã_t = Z_tΓ (N×K), P_t = Γ'W_tΓ (K×K), q_t = Γ'X_t (K).
The row-major vectorisation θ = vec_C(Λ') ∈ ℝ^{KR} satisfies:

    [Σ_t n_t · kron(P_t, m_t m_t')] θ = Σ_t n_t · kron(q_t, m_t)

where kron is np.kron (row-major Kronecker).
After solving, Λ = reshape(θ, K, R).T gives R × K.
"""

import numpy as np
import pandas as pd
import scipy.linalg as sla
from typing import Dict, Tuple, Optional


class HardGIPCA:
    """
    Hard Macro Factor Model (Method A).

    Factors are fully determined by the macro state: f_t = Λ' m_t.
    No free/latent factor time series is estimated.

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
        self.Gamma = None   # L × K  (DataFrame)
        self.Lambda = None  # R × K  (DataFrame)

        self._fitted = False
        self.objective_history = []
        self.n_iterations = 0

    # ──────────────────────────────────────────────────────────
    #  Loss function
    # ──────────────────────────────────────────────────────────
    def loss_fct(self, Gamma: np.ndarray, Lambda: np.ndarray,
                 data: list) -> float:
        """
        Hard GIPCA loss:  (1/N) Σ_t ‖x_t − Z_t Γ (Λ' m_t)‖².

        Parameters
        ----------
        Gamma : np.ndarray (L × K)
        Lambda : np.ndarray (R × K)
        data : list  [rets (T,N), Z (T,N,L), M (T,R)]

        Returns
        -------
        float
        """
        rets, Z, M = data
        T, N = rets.shape

        obj = 0.0
        for t in range(T):
            g_t = Lambda.T @ M[t, :]                 # K
            pred = Z[t, :, :] @ Gamma @ g_t           # N
            obj += np.sum((rets[t, :] - pred) ** 2)

        return obj / N

    # ──────────────────────────────────────────────────────────
    #  Fit via ALS
    # ──────────────────────────────────────────────────────────
    def fit(self, data: list, max_iter: int = 1000, tol: float = 1e-6,
            verbose: bool = True, seed: int = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the Hard GIPCA model via ALS (Γ-step, Λ-step).

        Parameters
        ----------
        data : list
            [rets (T,N), Z (T,N,L), M (T,R)]
        max_iter : int
        tol : float
            Convergence tolerance (max absolute parameter change).
        verbose : bool
        seed : int, optional

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
        Gamma_old, Lambda_old = self._initialize(seed=seed)

        obj_init = self._compute_objective(Gamma_old, Lambda_old)
        self.objective_history.append(obj_init)
        if verbose:
            print(f"Iteration    0: Objective = {obj_init:.6f}")

        # ── ALS loop ──
        for iteration in range(max_iter):
            # Step 1 — Γ-step:  update Gamma given Lambda
            Gamma_new = self._update_gamma(Lambda_old)

            # Step 2 — Λ-step:  update Lambda given Gamma
            Lambda_new = self._update_lambda(Gamma_new)

            # Objective
            obj = self._compute_objective(Gamma_new, Lambda_new)
            self.objective_history.append(obj)

            # Convergence
            gamma_change = np.max(np.abs(Gamma_new.values - Gamma_old.values))
            lambda_change = np.max(np.abs(Lambda_new.values - Lambda_old.values))
            max_change = max(gamma_change, lambda_change)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:4d}: Objective = {obj:.6f}, "
                      f"Max change = {max_change:.2e}")

            if max_change < tol:
                if verbose:
                    print(f"\nConverged after {iteration + 1} iterations")
                    print(f"Initial objective: {self.objective_history[0]:.6f}")
                    print(f"Final objective:   {obj:.6f}")
                    print(f"Reduction:         "
                          f"{(1 - obj / self.objective_history[0]) * 100:.2f}%")
                break

            Gamma_old = Gamma_new
            Lambda_old = Lambda_new

        # ── Store results ──
        self.Gamma = Gamma_new
        self.Lambda = Lambda_new
        self._fitted = True
        self.n_iterations = iteration + 1

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
    #  Initialisation
    # ──────────────────────────────────────────────────────────
    def _initialize(self, seed=None):
        """
        Best-of-population initialisation on the Grassmannian.

        For each random Gamma candidate:
          1.  Cross-sectional OLS → pseudo-factors f̂_t.
          2.  Regress f̂_t on m_t → Lambda_cand.
          3.  Evaluate the hard loss with g_t = Lambda_cand' m_t.
        Keep the (Gamma, Lambda) pair with the lowest loss.
        """
        if seed is not None:
            np.random.seed(seed)

        dim = self.L * self.K
        pop_size = min(5 * dim, 500)

        best_obj = np.inf
        best_Gamma = None
        best_Lambda = None

        M_sum = self._macro.T @ self._macro  # R × R

        for _ in range(pop_size):
            # Random orthonormal Gamma
            A = np.random.uniform(-1.0, 1.0, size=(self.L, self.K))
            Q, _ = np.linalg.qr(A)
            Gamma_cand = Q[:, :self.K]

            # Cross-sectional pseudo-factors
            factors_arr = np.zeros((self.K, self.T))
            for t in range(self.T):
                A_t = self._characteristics[t, :, :] @ Gamma_cand
                factors_arr[:, t], *_ = np.linalg.lstsq(
                    A_t, self._returns[t, :], rcond=None)

            # OLS: Lambda = (M'M)^{-1} M' F'  where F is K × T
            sum_fm = factors_arr @ self._macro          # K × R
            Lambda_cand = sla.solve(
                M_sum + 1e-8 * np.eye(self.R),
                sum_fm.T, assume_a='sym')               # R × K

            obj = self.loss_fct(Gamma_cand, Lambda_cand, self._data)
            if obj < best_obj:
                best_obj = obj
                best_Gamma = Gamma_cand
                best_Lambda = Lambda_cand

        Gamma = pd.DataFrame(best_Gamma,
                             index=self.char_names, columns=self.factor_names)
        Lambda = pd.DataFrame(best_Lambda,
                              index=self.macro_names, columns=self.factor_names)
        return Gamma, Lambda

    # ──────────────────────────────────────────────────────────
    #  Γ-step  (Algorithm 1, lines 3-8)
    # ──────────────────────────────────────────────────────────
    def _update_gamma(self, Lambda: pd.DataFrame) -> pd.DataFrame:
        """
        Update Gamma given Lambda.

        Uses macro-implied factors g_t = Λ' m_t.

        Row-major Kronecker normal equation (LK × LK):
            [Σ_t n_t · kron(W_t, g_t g_t')] vec_C(Γ) = Σ_t n_t · kron(X_t, g_t)

        Then orthonormalise via SVD:  Γ_unc = UΔV' → Γ = UV'.
        """
        Lambda_arr = Lambda.values  # R × K

        vec_length = self.L * self.K
        numerator = np.zeros(vec_length)
        denominator = np.zeros((vec_length, vec_length))

        for t in self.times:
            m_t = self._macro[t, :]             # R
            g_t = Lambda_arr.T @ m_t            # K   (macro-implied factor)

            X_t = self._X[t].values             # L
            W_t = self._W[t].values             # L × L
            n_t = self._N_valid[t]

            gg = np.outer(g_t, g_t)             # K × K
            numerator += np.kron(X_t, g_t) * n_t
            denominator += np.kron(W_t, gg) * n_t

        # Regularise and solve
        denominator += self.ridge * np.eye(vec_length)
        gamma_vec, *_ = np.linalg.lstsq(denominator, numerator, rcond=None)

        # Reshape and orthonormalise via SVD  (Γ_unc = UΔV' → Γ = UV')
        Gamma_unc = gamma_vec.reshape((self.L, self.K))
        U, _, Vt = np.linalg.svd(Gamma_unc, full_matrices=False)
        Gamma_arr = U @ Vt  # L × K  with  Γ'Γ = I_K

        return pd.DataFrame(Gamma_arr,
                            index=self.char_names, columns=self.factor_names)

    # ──────────────────────────────────────────────────────────
    #  Λ-step  (Algorithm 1, lines 9-12)
    # ──────────────────────────────────────────────────────────
    def _update_lambda(self, Gamma: pd.DataFrame) -> pd.DataFrame:
        """
        Update Lambda given Gamma.

        Works in the K-dimensional projected space:
            P_t = Γ' W_t Γ   (K×K)
            q_t = Γ' X_t     (K)

        Row-major Kronecker normal equation (KR × KR):
            [Σ_t n_t · kron(P_t, m_t m_t')  +  ridge · I] θ
                = Σ_t n_t · kron(q_t, m_t)

        where θ = vec_C(Λ') ∈ ℝ^{KR}.
        After solving: Λ = reshape(θ, K, R).T   →  R × K.
        """
        Gamma_arr = Gamma.values  # L × K

        system_dim = self.K * self.R
        denominator = np.zeros((system_dim, system_dim))
        numerator = np.zeros(system_dim)

        for t in self.times:
            m_t = self._macro[t, :]                # R
            W_t = self._W[t].values                # L × L
            X_t = self._X[t].values                # L
            n_t = self._N_valid[t]

            P_t = Gamma_arr.T @ W_t @ Gamma_arr   # K × K
            q_t = Gamma_arr.T @ X_t               # K

            mm = np.outer(m_t, m_t)                # R × R

            denominator += np.kron(P_t, mm) * n_t
            numerator += np.kron(q_t, m_t) * n_t

        # Ridge regularisation
        denominator += self.ridge * np.eye(system_dim)

        # Solve and reshape
        theta, *_ = np.linalg.lstsq(denominator, numerator, rcond=None)
        Lambda_T = theta.reshape((self.K, self.R))  # K × R  =  Λ'
        Lambda_arr = Lambda_T.T                      # R × K  =  Λ

        return pd.DataFrame(Lambda_arr,
                            index=self.macro_names, columns=self.factor_names)

    # ──────────────────────────────────────────────────────────
    #  Objective wrapper
    # ──────────────────────────────────────────────────────────
    def _compute_objective(self, Gamma, Lambda):
        return self.loss_fct(Gamma.values, Lambda.values, self._data)

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

        # Compute deterministic factor series
        factors_arr = (self._macro @ self.Lambda.values).T  # K × T
        factors = pd.DataFrame(factors_arr,
                               index=self.factor_names, columns=self.times)

        return {
            'Gamma': self.Gamma,
            'Lambda': self.Lambda,
            'factors': factors,
            'objective_history': np.array(self.objective_history),
            'n_iterations': self.n_iterations,
        }

    # ──────────────────────────────────────────────────────────
    #  Macro R² (trivially 1.0 for hard model)
    # ──────────────────────────────────────────────────────────
    def factor_macro_r2(self) -> np.ndarray:
        """
        Macro R² per factor.  Always 1.0 by construction (f_t ≡ Λ' m_t).
        Provided for API compatibility with the soft GIPCA.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        return np.ones(self.K)



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
    mu[0] = mean_mu_t[0] + rng.normal(scale=sigma_mu, size=num_macro) @ Lmu.T
    for t in range(1, T):
        innov = (rng.normal(size=num_macro) @ Lmu.T) * sigma_mu
        mu[t] = mean_mu_t[t] + mu_rho * (mu[t - 1] - mean_mu_t[t - 1]) + innov

    # ----- Generate f0_t (T, k): AR(1) -----
    f0 = np.zeros((T, k), dtype=float)
    f0[0] = rng.normal(scale=sigma_f0, size=k)
    for t in range(1, T):
        f0[t] = f0_rho * f0[t - 1] + rng.normal(scale=sigma_f0, size=k)

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
        },
    }
    return data, truth




# =====================================================================
#  Test harness
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HardGIPCA (Method A) — Test Suite")
    print("=" * 70)

    # ── 1.  Synthetic data using generate_gipca_data  ────────
    seed = 6890
    np.random.seed(seed)

    num_assets = 40   # N
    num_fact = 5      # K
    num_charact = 25  # L
    num_macro = 3     # R
    win_len = 21      # T
    max_iter = 500
    include_intercept = False

    # Generate synthetic GIPCA data
    data, truth = generate_gipca_data(
        T=win_len,
        N=num_assets,
        m=num_charact,
        k=num_fact,
        num_macro=num_macro,
        include_intercept=include_intercept,
        seed=seed
    )

    # Extract ground truth
    true_Gamma = truth['W_star']       # m × k (called Gamma in HardGIPCA)
    true_Delta = truth['Delta_star']   # k × num_macro
    true_f0 = truth['f0']              # T × k
    true_f_full = truth['f_full']      # T × k

    # ── 2.  Fit Hard GIPCA  ──────────────────────────────────
    print(f"\nData: T={win_len}, N={num_assets}, L={num_charact}, "
          f"K={num_fact}, R={num_macro}")
    print("-" * 70)

    model = HardGIPCA(
        num_assets=num_assets,
        num_fact=num_fact,
        num_charact=num_charact,
        num_macro=num_macro,
        win_len=win_len,
    )

    Gamma_est, history = model.fit(data, max_iter=max_iter, tol=1e-6,
                                   verbose=True, seed=seed)

    # ── 3.  Results  ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    results = model.get_results()
    print(f"\nGamma shape:   {results['Gamma'].shape}")
    print(f"Lambda shape:  {results['Lambda'].shape}")
    print(f"Factors shape: {results['factors'].shape}")

    r2 = model.score(data)
    print(f"\nOverall R²:    {r2:.4f}")

    # Subspace comparison (true W_star vs estimated Gamma)
    Q1, _ = np.linalg.qr(true_Gamma)
    Q2, _ = np.linalg.qr(Gamma_est)
    _, s, _ = np.linalg.svd(Q1.T @ Q2)
    principal_angles = np.arccos(np.clip(s, -1, 1))

    print(f"\nSubspace comparison (True W_star vs Estimated Γ):")
    print(f"  Principal angles (°): {np.round(np.degrees(principal_angles), 2)}")
    print(f"  Grassmann distance:   {np.linalg.norm(principal_angles):.4f}")

    # Factor correlation (estimated vs true full factors)
    est_factors = results['factors'].values  # K × T
    factor_corr = np.corrcoef(est_factors, true_f_full.T)[:num_fact, num_fact:]
    print(f"\n|Factor correlations| with true factors:")
    print(np.round(np.abs(factor_corr), 3))

    # ── 4.  Ground truth comparison  ─────────────────────────
    print("\n" + "=" * 70)
    print("Ground Truth Comparison")
    print("=" * 70)
    print(f"\nTrue W_star (Gamma):\n{true_Gamma[:5, :]}  ... (first 5 rows)")
    print(f"\nEstimated Gamma:\n{Gamma_est[:5, :]}  ... (first 5 rows)")
    print(f"\nTrue Delta:\n{true_Delta}")
    print(f"\nEstimated Lambda:\n{results['Lambda'].values}")

    # ── 5.  Monotonicity check  ──────────────────────────────
    diffs = np.diff(history)
    n_increases = np.sum(diffs > 1e-10)
    print(f"\nObjective monotonicity: "
          f"{'PASS ✓' if n_increases == 0 else f'FAIL ✗ ({n_increases} increases)'}")

    print("\n" + "=" * 70)
    print("HardGIPCA test complete!")
    print("=" * 70)