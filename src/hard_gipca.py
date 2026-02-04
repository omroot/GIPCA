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


# =====================================================================
#  Test harness
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HardGIPCA (Method A) — Test Suite")
    print("=" * 70)

    # ── 1.  Synthetic data  ──────────────────────────────────
    np.random.seed(42)

    T = 100   # time periods
    N = 50    # assets
    L = 10    # characteristics
    K = 3     # factors
    R = 5     # macro variables (not counting intercept)

    # True parameters
    true_Gamma = np.random.randn(L, K)
    true_Gamma, _ = np.linalg.qr(true_Gamma)
    true_Gamma = true_Gamma[:, :K]

    true_Lambda = np.random.randn(R + 1, K) * 0.5  # +1 for intercept

    # Macro variables (persistent)
    M_raw = np.cumsum(np.random.randn(T, R) * 0.1, axis=0)
    M = np.column_stack([np.ones(T), M_raw])  # T × (R+1), intercept first
    R_total = R + 1

    # True factors:  f_t = Lambda' m̃_t  (hard DGP — no noise on factors)
    true_factors = M @ true_Lambda  # T × K

    # Characteristics
    Z = np.random.randn(T, N, L) * 0.5
    Z[:, :, 0] = 1  # intercept characteristic

    # Returns:  r_{i,t} = c_{i,t}' Gamma f_t + ε
    rets = np.zeros((T, N))
    for t in range(T):
        loadings = Z[t, :, :] @ true_Gamma
        rets[t, :] = loadings @ true_factors[t, :] + np.random.randn(N) * 0.5

    data = [rets, Z, M]

    # ── 2.  Fit Hard GIPCA  ──────────────────────────────────
    print(f"\nData: T={T}, N={N}, L={L}, K={K}, R={R_total} (incl. intercept)")
    print("-" * 70)

    model = HardGIPCA(
        num_assets=N,
        num_fact=K,
        num_charact=L,
        num_macro=R_total,
        win_len=T,
    )

    Gamma_est, history = model.fit(data, max_iter=500, tol=1e-6,
                                   verbose=True, seed=42)

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

    # Subspace comparison
    Q1, _ = np.linalg.qr(true_Gamma)
    Q2, _ = np.linalg.qr(Gamma_est)
    _, s, _ = np.linalg.svd(Q1.T @ Q2)
    principal_angles = np.arccos(np.clip(s, -1, 1))

    print(f"\nSubspace comparison (True vs Estimated Γ):")
    print(f"  Principal angles (°): {np.round(np.degrees(principal_angles), 2)}")
    print(f"  Grassmann distance:   {np.linalg.norm(principal_angles):.4f}")

    # Factor correlation
    est_factors = results['factors'].values  # K × T
    factor_corr = np.corrcoef(est_factors, true_factors.T)[:K, K:]
    print(f"\n|Factor correlations| with true factors:")
    print(np.round(np.abs(factor_corr), 3))

    # ── 4.  Compare with soft GIPCA  ─────────────────────────
    print("\n" + "=" * 70)
    print("Comparison: Hard GIPCA  vs  Soft GIPCA (α = 1)")
    print("=" * 70)

    import sys
    sys.path.insert(0, '/mnt/user-data/uploads')
    try:
        from generalized_ipca import GIPCA

        soft_model = GIPCA(
            num_assets=N,
            num_fact=K,
            num_charact=L,
            num_macro=R_total,
            win_len=T,
            alpha=1.0,
        )
        _, soft_history = soft_model.fit(data, max_iter=500, tol=1e-6,
                                         verbose=False, seed=42)
        soft_r2 = soft_model.score(data)
        soft_macro_r2 = soft_model.factor_macro_r2()

        # Soft Gamma subspace angle
        Q3, _ = np.linalg.qr(soft_model.Gamma.values)
        _, s3, _ = np.linalg.svd(Q1.T @ Q3)
        soft_angles = np.arccos(np.clip(s3, -1, 1))

        print(f"\n{'Metric':35s} {'Hard':>10s}  {'Soft(α=1)':>10s}")
        print("-" * 60)
        print(f"{'Overall R²':35s} {r2:10.4f}  {soft_r2:10.4f}")
        print(f"{'Iterations':35s} {model.n_iterations:10d}  "
              f"{soft_model.n_iterations:10d}")
        print(f"{'Final objective (own loss)':35s} {history[-1]:10.4f}  "
              f"{soft_history[-1]:10.4f}")
        print(f"{'Macro R² (mean)':35s} {'1.0000':>10s}  "
              f"{np.mean(soft_macro_r2):10.4f}")
        print(f"{'Grassmann dist to true Γ':35s} "
              f"{np.linalg.norm(principal_angles):10.4f}  "
              f"{np.linalg.norm(soft_angles):10.4f}")
    except Exception as e:
        print(f"Could not load soft GIPCA for comparison: {e}")

    # ── 5.  Monotonicity check  ──────────────────────────────
    diffs = np.diff(history)
    n_increases = np.sum(diffs > 1e-10)
    print(f"\nObjective monotonicity: "
          f"{'PASS ✓' if n_increases == 0 else f'FAIL ✗ ({n_increases} increases)'}")

    print("\n" + "=" * 70)
    print("HardGIPCA test complete!")
    print("=" * 70)