import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.function import autograd as pymanopt_autograd
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions
from scipy.linalg import subspace_angles

# from parsifal.manifold.differential_evolution import JDifferentialEvolution

#______________________________________________________________________________________________________________


class GrassmannDiffEvolGIPCAEstimator:
    """
    For fixed Gamma (represented by w = Gamma^T, shape (m,k)):
      1) Cross-sectional factors:
            f_hat[t] = argmin_f || r_t - (Z_t w) f ||^2
      2) Macro loading (time-series regression):
            Delta_hat = (sum_t f_hat[t] mu_t^T) (sum_t mu_t mu_t^T)^{-1}
      3) Residual factors:
            f0_hat[t] = f_hat[t] - Delta_hat mu_t
         which implies sum_t f0_hat[t] mu_t^T = 0.

    Optimization over w uses the profiled IPCA loss:
        L(w) = (1/T) sum_t min_f || r_t - (Z_t w) f ||^2
    which is Grassmann-invariant and is what you used in pymanopt.
    """

    def __init__(self, num_assets, num_fact, num_charact, num_macro, win_len):
        self.num_assets = num_assets      # N
        self.grass_n = num_charact        # m
        self.grass_k = num_fact           # k
        self.num_macro = num_macro        # l
        self.dim = self.grass_n * self.grass_k
        self.win_len = win_len

    # ------------------------------------------------------------------
    # Core estimators given w
    # ------------------------------------------------------------------
    def estimate_f(self, w: np.ndarray, data) -> np.ndarray:
        """
        Cross-sectional factor realizations:
            f_hat[t] = argmin_f || r_t - (Z_t w) f ||^2

        Returns:
            f_hat: (T, k)
        """
        rets, Z, mu = data
        T = self.win_len
        k = self.grass_k

        f_hat = np.zeros((T, k), dtype=float)
        for t in range(T):
            Lambda_t = Z[t] @ w                 # (N, k)
            f_hat[t], *_ = np.linalg.lstsq(Lambda_t, rets[t], rcond=None)
        return f_hat

    def estimate_delta(self, f_hat: np.ndarray, mu: np.ndarray, ridge_mu: float = 0.0) -> np.ndarray:
        """
        Time-series regression of factors on macro:
            Delta_hat = (F^T M) (M^T M)^{-1}
        where F is (T,k), M is (T,l). Returns (k,l).

        ridge_mu adds ridge to (M^T M) for stability if needed.
        """
        k = self.grass_k
        l = self.num_macro
        assert f_hat.shape == (self.win_len, k)
        assert mu.shape == (self.win_len, l)

        S_fm = f_hat.T @ mu                    # (k, l)
        S_mm = mu.T @ mu                       # (l, l)
        if ridge_mu > 0.0:
            S_mm = S_mm + ridge_mu * np.eye(l)

        Delta_hat = S_fm @ np.linalg.inv(S_mm)  # (k, l)
        return Delta_hat

    def estimate_f0(self, f_hat: np.ndarray, Delta_hat: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Residual factors:
            f0_hat[t] = f_hat[t] - Delta_hat mu_t
        """
        assert Delta_hat.shape == (self.grass_k, self.num_macro)
        f0_hat = f_hat - (mu @ Delta_hat.T)     # (T,k)
        return f0_hat

    # ------------------------------------------------------------------
    # Objective for optimizing w (Gamma^T)
    # ------------------------------------------------------------------
    def loss_fct(self, w, data):
        """
        Profiled IPCA objective:
            L(w) = (1/T) sum_t min_f || r_t - (Z_t w) f ||^2
        """
        w = np.asarray(w)
        if w.ndim == 1:
            w = w.reshape(self.grass_n, self.grass_k)
        assert w.shape == (self.grass_n, self.grass_k)

        rets, Z, mu = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)
        assert mu.shape == (self.win_len, self.num_macro)

        obj = 0.0
        for t in range(self.win_len):
            Lambda_t = Z[t] @ w
            f_t, *_ = np.linalg.lstsq(Lambda_t, rets[t], rcond=None)
            resid = rets[t] - (Lambda_t @ f_t)
            obj += float(resid @ resid)
        return obj / self.win_len

    # ------------------------------------------------------------------
    # Fit using your DE engine (kept for parity)
    # ------------------------------------------------------------------
    def fit(self, data, max_gen=500, ridge_mu: float = 0.0):
        """
        Optimize w using Grassmann DE, then compute:
          f_hat, Delta_hat, f0_hat

        Returns:
          W (m,k), Delta_hat (k,l), f0_hat (T,k), f_hat (T,k), history
        """
        rets, Z, mu = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)
        assert mu.shape == (self.win_len, self.num_macro)

        w_min = np.full(self.dim, -1.0)
        w_max = np.ones(self.dim)
        pop_size = 5 * self.dim

        de = JDifferentialEvolution(
            w_min, w_max, pop_size,
            model='grassmannian',
            grass_k=self.grass_k
        )

        w_opt, max_dist, max_gen, history = de.optimize(
            self.loss_fct, params=data,
            eps=1e-3, max_gen=max_gen,
            history=True, verbose=True
        )

        W = np.asarray(w_opt).reshape(self.grass_n, self.grass_k)

        # Post-estimation of factors / Delta / f0
        f_hat = self.estimate_f(W, data)
        Delta_hat = self.estimate_delta(f_hat, mu, ridge_mu=ridge_mu)
        f0_hat = self.estimate_f0(f_hat, Delta_hat, mu)

        print(f"W (Gamma^T):\n{W}")
        print(f"Delta_hat:\n{Delta_hat}")
        print(f"f_hat: {f_hat}")
        print(f"f0_hat: {f0_hat}")
        print(f"max_dist: {max_dist}")
        print(f"max_gen: {max_gen}")
        print(f"objective function: {self.loss_fct(W, data=data)}")

        plt.figure(figsize=(8, 5))
        plt.plot(history, label='Objective function')
        plt.title("GIPCA (identified Delta and f^0; optimize Gamma only)")
        plt.xlabel("Generation (g)")
        plt.ylabel("Objective function (f)")
        plt.legend()
        plt.show()

        return W, Delta_hat, f0_hat, f_hat, history


#______________________________________________________________________________________________________________


class GrassmannManifoldGIPCAEstimator:
    """
    Optimization variable:
        W = Gamma^T in R^{m x k} with orthonormal columns (Grassmann representative)

    Objective (profiled IPCA loss):
        L(W) = (1/T) sum_t min_f || r_t - (Z_t W) f ||^2

    Post-estimation:
        1) f_hat[t] = argmin_f || r_t - (Z_t W) f ||^2
        2) Delta_hat = (F^T M) (M^T M)^{-1}
        3) f0_hat[t] = f_hat[t] - Delta_hat mu_t

    Returns:
        W, Delta_hat, f0_hat, f_hat, history
    """

    def __init__(self, num_assets, num_fact, num_charact, num_macro, win_len):
        self.num_assets = num_assets      # N
        self.grass_n = num_charact        # m
        self.grass_k = num_fact           # k
        self.num_macro = num_macro        # l
        self.win_len = win_len

    # ------------------------------------------------------------------
    # Post-estimation helpers (NumPy)
    # ------------------------------------------------------------------
    def estimate_f(self, W: np.ndarray, data) -> np.ndarray:
        rets, Z, mu = data
        T = self.win_len
        k = self.grass_k
        f_hat = np.zeros((T, k), dtype=float)
        for t in range(T):
            Lambda_t = Z[t] @ W
            f_hat[t], *_ = np.linalg.lstsq(Lambda_t, rets[t], rcond=None)
        return f_hat

    def estimate_delta(self, f_hat: np.ndarray, mu: np.ndarray, ridge_mu: float = 0.0) -> np.ndarray:
        k = self.grass_k
        l = self.num_macro
        assert f_hat.shape == (self.win_len, k)
        assert mu.shape == (self.win_len, l)

        S_fm = f_hat.T @ mu          # (k,l)
        S_mm = mu.T @ mu             # (l,l)
        if ridge_mu > 0.0:
            S_mm = S_mm + ridge_mu * np.eye(l)

        return S_fm @ np.linalg.inv(S_mm)

    def estimate_f0(self, f_hat: np.ndarray, Delta_hat: np.ndarray, mu: np.ndarray) -> np.ndarray:
        # f0_hat[t] = f_hat[t] - Delta_hat mu_t
        return f_hat - (mu @ Delta_hat.T)

    # ------------------------------------------------------------------
    # Pymanopt fit (current pymanopt: no callback, use log_verbosity)
    # ------------------------------------------------------------------
    def fit(
        self,
        data,
        optimizer: str = "ConjugateGradient",
        ridge_mu: float = 0.0,
        max_iterations: int = 200,
        verbosity: int = 1,
        log_verbosity: int = 1,
        initial_point: np.ndarray | None = None,
        reuse_line_searcher: bool = False,
        return_pymanopt_result: bool = False,
    ):
        """
        Fit using current pymanopt API.

        Key points (current pymanopt):
          - Problem() has no verbosity kwarg.
          - Optimizers have verbosity + log_verbosity.
          - ConjugateGradient.run() does not accept callback.
          - History is obtained from result.log when log_verbosity >= 1.
        """
        rets, Z, mu = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)
        assert mu.shape == (self.win_len, self.num_macro)

        # --- objective in autograd.numpy ---
        def ipca_profiled_loss_autograd(W, rets_, Z_):
            T_, _ = rets_.shape
            obj = 0.0
            for t in range(T_):
                Zt = Z_[t]               # (N,m)
                rt = rets_[t]            # (N,)
                Lt = anp.dot(Zt, W)      # (N,k)
                XtX = anp.dot(Lt.T, Lt)  # (k,k)
                Xty = anp.dot(Lt.T, rt)  # (k,)
                bt = anp.linalg.solve(XtX, Xty)
                resid = rt - anp.dot(Lt, bt)
                obj = obj + anp.dot(resid, resid)
            return obj / T_

        manifold = Grassmann(self.grass_n, self.grass_k)
        rets_ag = anp.asarray(rets)
        Z_ag = anp.asarray(Z)

        @pymanopt_autograd(manifold)
        def cost(W):
            return ipca_profiled_loss_autograd(W, rets_ag, Z_ag)

        problem = Problem(manifold=manifold, cost=cost)

        opt = optimizer.lower().replace("_", "")
        if opt in {"cg", "conjugategradient"}:
            solver = ConjugateGradient(
                max_iterations=max_iterations,
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs = {"reuse_line_searcher": reuse_line_searcher}
        elif opt in {"sd", "steepestdescent"}:
            solver = SteepestDescent(
                max_iterations=max_iterations,
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs = {}
        elif opt in {"tr", "trustregions"}:
            solver = TrustRegions(
                max_iterations=max_iterations,
                verbosity=verbosity,
                log_verbosity=log_verbosity,
            )
            run_kwargs = {}
        else:
            raise ValueError("optimizer must be one of: ConjugateGradient, SteepestDescent, TrustRegions")

        if initial_point is not None:
            initial_point = np.asarray(initial_point)
            assert initial_point.shape == (self.grass_n, self.grass_k)
            result = solver.run(problem, initial_point=initial_point, **run_kwargs)
        else:
            result = solver.run(problem, **run_kwargs)

        Wopt = np.asarray(result.point)

        # ---- history extraction (current pymanopt stores it in result.log) ----
        history = []
        if getattr(result, "log", None) is not None:
            iters = result.log.get("iterations", None)
            if isinstance(iters, dict) and "cost" in iters:
                history = [float(c) for c in iters["cost"]]

        if not history:
            # fallback: at least store final cost
            history = [float(cost(Wopt))]

        # ---- post-estimation ----
        f_hat = self.estimate_f(Wopt, data)
        Delta_hat = self.estimate_delta(f_hat, mu, ridge_mu=ridge_mu)
        f0_hat = self.estimate_f0(f_hat, Delta_hat, mu)

        if return_pymanopt_result:
            return Wopt, Delta_hat, f0_hat, f_hat, history, result
        return Wopt, Delta_hat, f0_hat, f_hat, history


#______________________________________________________________________________________________________________


def grassmann_recovery_report(W_hat: np.ndarray, truth: dict, Delta_hat: np.ndarray | None = None):
    """
    Quick validation report vs synthetic truth from generate_gipca_data().

    Prints:
      - principal angles between subspaces span(W_hat) and span(W_star)
      - a simple subspace distance summary (max angle, RMS angle)
      - (optional) Delta error if Delta_hat is provided
    """
    W_star = truth["W_star"]
    assert W_hat.shape == W_star.shape, f"W_hat {W_hat.shape} vs W_star {W_star.shape}"

    # principal angles (in radians)
    ang = subspace_angles(W_hat, W_star)
    ang_deg = np.degrees(ang)

    max_ang = float(np.max(ang_deg))
    rms_ang = float(np.sqrt(np.mean(ang_deg ** 2)))

    print("\n=== Grassmann recovery report ===")
    print("Principal angles (deg):", np.array2string(ang_deg, precision=4, separator=", "))
    print(f"Max principal angle (deg): {max_ang:.6f}")
    print(f"RMS principal angle (deg): {rms_ang:.6f}")

    # Optional: Delta recovery
    if Delta_hat is not None and "Delta_star" in truth:
        Delta_star = truth["Delta_star"]
        if Delta_hat.shape != Delta_star.shape:
            print(f"Delta shapes differ: Delta_hat {Delta_hat.shape} vs Delta_star {Delta_star.shape}")
        else:
            rel = np.linalg.norm(Delta_hat - Delta_star) / (np.linalg.norm(Delta_star) + 1e-12)
            print(f"Relative Delta error ||Δhat-Δ*||/||Δ*||: {rel:.6e}")

    # Optional: check Option A orthogonality for f0 if available
    if "mu" in truth and "f0" in truth:
        mu = truth["mu"]
        f0 = truth["f0"]
        ortho = np.linalg.norm(f0.T @ mu)
        print(f"Truth check ||f0^T mu|| (should be ~0 if enforced): {ortho:.3e}")


#______________________________________________________________________________________________________________



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

        enforce_identification: bool = True,
        ident_ridge: float = 1e-12,
        check_identification: bool = False,
        ident_tol: float = 1e-8,
):
    """
    Generates synthetic GIPCA-like data:
      - True W_* on Grassmann (orthonormal columns) in R^{m_eff x k}
      - Cross-sectionally correlated characteristics Z_t (N x m_eff)
      - Macro series mu_t (T x num_macro) with AR(1) dynamics + correlated innovations
      - True Delta_* (k x num_macro)
      - Factor residuals f0_t (T x k) with AR(1) dynamics
      - Optionally enforces identification in-sample:
            sum_t f0_t mu_t^T = 0   i.e. f0^T mu = 0
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
    if num_macro < 1:
        raise ValueError("num_macro must be >= 1.")
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

    # Enforce identification: f0 orthogonal to mu in-sample
    # Condition: sum_t f0_t mu_t^T = 0  <=>  f0^T mu = 0  (k x num_macro)
    ident_B = None
    if enforce_identification:
        S_mm = mu.T @ mu
        S_mm = S_mm + ident_ridge * np.eye(num_macro)  # for stability
        ident_B = (f0.T @ mu) @ np.linalg.inv(S_mm)     # (k, num_macro)
        f0 = f0 - (mu @ ident_B.T)                      # (T, k)

        if check_identification:
            resid = f0.T @ mu
            rel = np.linalg.norm(resid) / (np.linalg.norm(f0.T @ f0) + 1e-12)
            if np.linalg.norm(resid) > ident_tol:
                raise RuntimeError(
                    f"Identification check failed: ||f0^T mu||={np.linalg.norm(resid):.3e}, "
                    f"relative={rel:.3e}"
                )

    # ----- Cross-sectional heteroskedastic idiosyncratic scales -----
    u = rng.normal(size=N)
    sigma_i = sigma_eps_base * np.exp(hetero_strength * u)
    sigma_i = np.clip(sigma_i, 1e-4, np.percentile(sigma_i, 99.5))

    # ----- Generate returns -----
    rets = np.empty((T, N), dtype=float)
    use_t = np.isfinite(heavy_tail_df) and heavy_tail_df < 1e9

    # record full factors f_t = f0_t + Delta mu_t
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
        "ident_B": ident_B,  # regression coeffs used to orthogonalize f0 (None if not enforced)
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


#______________________________________________________________________________________________________________

if __name__ == '__main__':

    # Test code
    seed = 6890
    # seed = 156
    np.random.seed(seed)
    num_assets = 40     # N
    num_fact = 5        # k
    num_charact = 25    # m
    num_macro = 3       # l
    win_len = 21        # T

    # Generate simple data
    # rets = np.random.normal(size=(win_len, num_assets))
    # Z = np.random.uniform(size=(win_len, num_assets, num_charact))
    # mu = np.random.uniform(size=(win_len, num_macro))
    # data = [rets, Z, mu]

    # Generate hard data
    include_intercept = False
    data, truth = generate_gipca_data(T=win_len, N=num_assets, m=num_charact, k=num_fact, num_macro=num_macro,
                                 include_intercept=include_intercept, seed=seed)


    # Differential Evolution estimator
    # est_de = GrassmannDiffEvolGIPCAEstimator(num_assets=num_assets,
    #                                          num_fact=num_fact,
    #                                          num_charact=num_charact,
    #                                          num_macro=num_macro,
    #                                          win_len=win_len)
    # max_gen = 3_000
    # est_de.fit(data, max_gen=max_gen)

    # GIPCA Manifold estimator
    est_manifold = GrassmannManifoldGIPCAEstimator(
        num_assets=num_assets,
        num_fact=num_fact,
        num_charact=num_charact,
        num_macro=num_macro,
        win_len=win_len
    )

    Wopt, Delta_hat, f0_hat, f_hat, history = est_manifold.fit(
        data=data,
        optimizer="ConjugateGradient",
        max_iterations=200,
        verbosity=2,
        log_verbosity=1,
    )

    print("Wopt:", Wopt)  # (m, k)
    print("Delta_hat:", Delta_hat)  # (k, num_macro)
    print("f0_hat:", f0_hat)  # (T, k)
    print("f_hat:", f_hat)  # (T, k)
    print("Final loss:", history[-1])

    # Quick check of f0^T mu ~ 0
    _, _, mu = data
    orth = np.linalg.norm(f0_hat.T @ mu)
    print("|| f0_hat^T mu ||:", orth)

    # Convergence history
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.title("Pymanopt convergence (profiled IPCA loss)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.show()

    # Recovery report
    # grassmann_recovery_report(Wopt, truth, Delta_hat=Delta_hat)
