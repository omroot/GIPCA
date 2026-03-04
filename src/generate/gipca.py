
import numpy as np


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

    # ----- Generate f0_t (T, k): AR(1) then orthogonalize w.r.t. mu -----
    f0 = np.zeros((T, k), dtype=float)
    f0[0] = rng.normal(scale=sigma_f0, size=k)
    for t in range(1, T):
        f0[t] = f0_rho * f0[t - 1] + rng.normal(scale=sigma_f0, size=k)

    # Identification: f0 ⊥ mu  (project out the mu component)
    coef = np.linalg.lstsq(mu, f0, rcond=None)[0]  # (num_macro, k)
    f0 = f0 - mu @ coef

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

