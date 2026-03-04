import numpy as np


def generate_ipca_data(
        T: int = 252,  # time points
        N: int = 500,  # assets
        m: int = 20,  # TOTAL characteristics (includes intercept if include_intercept=True)
        k: int = 5,  # factors
        seed: int = 124,

        # Factor dynamics
        beta_rho: float = 0.9,  # AR(1) persistence
        sigma_beta: float = 0.5,  # innovation scale

        # Characteristic structure
        z_rho: float = 0.4,  # feature correlation in Toeplitz covariance
        z_scale: float = 1.0,  # overall scale

        # Returns noise
        heavy_tail_df: float = 5.0,  # Student-t dof; set to np.inf for Gaussian
        sigma_eps_base: float = 0.5,  # base idiosyncratic scale
        hetero_strength: float = 0.5,  # cross-sectional heteroskedasticity strength in [0, ~2]

        # Missingness
        missing_prob: float = 0.05,  # per entry missingness probability in Z (non-intercept cols)
        missing_mode: str = "mcAR",  # "mcAR" or "tail" (more missing for extreme values)
        impute: str = "zero",  # "zero" or "mean"

        # Intercept
        include_intercept: bool = True,

        # Optional: time-varying characteristic drift
        z_drift_scale: float = 0.02,
):
    """
    Generates synthetic IPCA-like data with:
      - True W_* on Grassmann (orthonormal columns) in R^{(m(+1)) x k}
      - Cross-sectionally correlated characteristics Z_t (N x m)
      - Optional intercept column (all ones)
      - AR(1) factor returns beta_t
      - Heteroskedastic idiosyncratic noise across assets
      - Optional heavy-tailed noise (Student t)
      - Missingness in characteristics + simple imputation
    Returns:
      data = (rets, Z) where rets is (T,N) and Z is (T,N,m_eff)
      truth dict with W_star, beta, masks, sigmas, etc.
    """

    rng = np.random.default_rng(seed)

    if include_intercept:
        if m < 2:
            raise ValueError("Need m>=2 if include_intercept=True (1 intercept + at least 1 feature).")
        m_eff = m  # total columns returned
        m_core = m - 1  # random feature columns
    else:
        m_eff = m
        m_core = m

    # ----- True W_* (m_eff x k), orthonormal columns -----
    A = rng.normal(size=(m_eff, k))
    W_star, _ = np.linalg.qr(A)  # Grassmann representative

    # ----- Feature covariance for Z core (Toeplitz) -----
    if m_core > 0:
        idx = np.arange(m_core)
        Sigma_z = z_rho ** np.abs(idx[:, None] - idx[None, :])
        Sigma_z = Sigma_z + 1e-10 * np.eye(m_core)
        Lz = np.linalg.cholesky(Sigma_z)
    else:
        Lz = None  # no core features

    # ----- Time-varying drift in characteristic means (core only) -----
    mean_t = np.zeros((T, m_core))
    for t in range(1, T):
        mean_t[t] = mean_t[t - 1] + rng.normal(scale=z_drift_scale, size=m_core)

    # ----- Generate Z (T, N, m_eff) -----
    Z = np.empty((T, N, m_eff), dtype=float)
    for t in range(T):
        if m_core > 0:
            Z_core = (rng.normal(size=(N, m_core)) @ Lz.T) * z_scale
            Z_core = Z_core + mean_t[t]
        else:
            Z_core = np.empty((N, 0), dtype=float)

        if include_intercept:
            Z[t, :, 0] = 1.0
            Z[t, :, 1:] = Z_core
        else:
            Z[t, :, :] = Z_core

    # ----- Factor returns beta_t (T,k): AR(1) -----
    beta = np.zeros((T, k), dtype=float)
    beta[0] = rng.normal(scale=sigma_beta, size=k)
    for t in range(1, T):
        beta[t] = beta_rho * beta[t - 1] + rng.normal(scale=sigma_beta, size=k)

    # ----- Cross-sectional heteroskedastic idiosyncratic scales -----
    u = rng.normal(size=N)
    sigma_i = sigma_eps_base * np.exp(hetero_strength * u)
    sigma_i = np.clip(sigma_i, 1e-4, np.percentile(sigma_i, 99.5))

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

    # ----- Generate returns: r_t = (Z_t W_*) beta_t + eps_t -----
    rets = np.empty((T, N), dtype=float)
    use_t = np.isfinite(heavy_tail_df) and heavy_tail_df < 1e9

    for t in range(T):
        Lambda_t = Z[t] @ W_star  # (N,k)
        signal = Lambda_t @ beta[t]  # (N,)

        if use_t:
            df = heavy_tail_df
            if df <= 2:
                raise ValueError("heavy_tail_df must be > 2 for finite variance.")
            eps = rng.standard_t(df, size=N) * np.sqrt((df - 2) / df)
        else:
            eps = rng.normal(size=N)

        rets[t] = signal + sigma_i * eps

    data = [rets, Z]
    truth = {
        "W_star": W_star,
        "beta": beta,
        "sigma_i": sigma_i,
        "mask": mask,
        "m_eff": m_eff,
        "include_intercept": include_intercept,
        "params": {
            "beta_rho": beta_rho,
            "sigma_beta": sigma_beta,
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
