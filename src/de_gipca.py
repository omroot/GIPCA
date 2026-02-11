import numpy as np
from typing import Callable, List, Optional, Tuple
from matplotlib import pyplot as plt

# from parsifal.manifold.differential_evolution import JDifferentialEvolution


class JDifferentialEvolution:
    """
    Differential Evolution optimizer with support for constrained manifolds.

    Supports optimization on hypercubes, simplices, spheres, and Grassmannian
    manifolds via projection after mutation/crossover steps.

    Parameters
    ----------
    x_min : np.ndarray
        Lower bounds for each dimension
    x_max : np.ndarray
        Upper bounds for each dimension
    pop_size : int
        Population size (must be >= 4)
    model : str, default 'cube'
        Constraint model: 'cube', 'affine simplex', 'prob simplex',
        'sphere', or 'grassmannian'
    grass_k : int, default 0
        Number of columns for Grassmannian model (k in Gr(n, k))
    """

    def __init__(self, 
                 x_min: np.ndarray, 
                 x_max: np.ndarray, 
                 pop_size: int,
                 model: str = 'cube', 
                 grass_k: int = 0):
        
        self.x_min: np.ndarray = x_min
        self.x_max: np.ndarray = x_max
        self.pop_size: int = pop_size
        self.dim: int = len(x_min)
        self.population: np.ndarray = self.initialize_population()
        self.model: str = model
        self.grass_k: int = grass_k

    def initialize_population(self) -> np.ndarray:
        """Initialize population uniformly within bounds."""
        if self.pop_size < 4:
            raise ValueError("Population size has to be at least 4")
        pop = np.random.uniform(self.x_min, self.x_max, size=(self.pop_size, self.dim))
        return pop

    def crossover(self,
                   v: np.ndarray,
                   x: np.ndarray, 
                   cw: float) -> np.ndarray:
        """
        Binomial crossover between donor vector v and target vector x.

        Parameters
        ----------
        v : np.ndarray
            Donor vector
        x : np.ndarray
            Target vector
        cw : float
            Crossover probability

        Returns
        -------
        np.ndarray
            Trial vector
        """
        R = np.random.randint(self.dim)
        mask = (np.random.uniform(size=self.dim) <= cw)
        mask[R] = True
        u = np.where(mask, v, x)
        return u

    def mutate(self, 
               x1: np.ndarray, 
               x2: np.ndarray, 
               x3: np.ndarray,
               f: np.ndarray) -> np.ndarray:
        """
        DE/rand/1 mutation: x1 + f * (x2 - x3).

        Parameters
        ----------
        x1 : np.ndarray
            Base vector(s)
        x2 : np.ndarray
            Difference vector 1
        x3 : np.ndarray
            Difference vector 2
        f : np.ndarray
            Mutation scale factor(s)

        Returns
        -------
        np.ndarray
            Mutant vector(s)
        """
        return x1 + f * (x2 - x3)

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project population onto the constraint manifold.

        Parameters
        ----------
        x : np.ndarray (pop_size x dim)
            Population matrix to project

        Returns
        -------
        np.ndarray
            Projected population
        """
        if self.model == 'cube':
            return x
        elif self.model == 'affine simplex':
            x /= (np.sum(x, axis=1, keepdims=True) + 1e-10)
            return x
        elif self.model == 'prob simplex':
            x = np.maximum(x, 0.0)
            x /= (np.sum(x, axis=1, keepdims=True) + 1e-10)
            return x
        elif self.model == 'sphere':
            x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
            return x
        elif self.model == 'grassmannian':
            if self.dim % self.grass_k != 0:
                raise ValueError('Wrong dimensions for the Grassmannian model')
            n = self.dim // self.grass_k
            x_reshaped = x.reshape(self.pop_size, n, self.grass_k)
            q_matrices = np.array([np.linalg.qr(a)[0] for a in x_reshaped])
            return q_matrices.reshape(self.pop_size, self.dim)
        else:
            raise ValueError("Invalid model")

    def optimize(self, fitness: Callable, params: Optional[List[np.ndarray]] = None,
                 max_gen: int = 10_000, tau1: float = 0.1, tau2: float = 0.1,
                 eps: float = 1e-3, history: bool = False,
                 verbose: bool = True,
                 callback: Optional[Callable[[np.ndarray], None]] = None) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """
        Run the Differential Evolution optimization loop.

        Parameters
        ----------
        fitness : Callable
            Objective function with signature fitness(x, params) -> float
        params : Optional[List[np.ndarray]], default None
            Additional parameters passed to the fitness function
        max_gen : int, default 10000
            Maximum number of generations
        tau1 : float, default 0.1
            Probability of updating mutation scale factor F
        tau2 : float, default 0.1
            Probability of updating crossover weight CR
        eps : float, default 1e-3
            Convergence tolerance on max population spread
        history : bool, default False
            Whether to record best fitness per generation
        verbose : bool, default True
            Whether to print progress every 100 generations
        callback : Optional[Callable[[np.ndarray], None]], default None
            Called each generation with the current best solution vector

        Returns
        -------
        Tuple[np.ndarray, float, int, np.ndarray]
            (best solution, max population spread, generations run, fitness history)
        """
        f_low, f_up = 0.1, 0.9
        pop_size, dim = self.pop_size, self.dim
        cw = np.full(pop_size, 0.9)
        f = np.full(pop_size, 0.1)
        gen, max_dist = 0, 1.0
        hist_vec = []
        self.population = np.copy(self.population)
        self.population = self.project(self.population)
        while max_dist > eps and gen < max_gen:
            scores = np.apply_along_axis(fitness, 1, self.population, params)
            r1, r2, r3 = self._select_random_indices(pop_size)
            v = self.mutate(self.population[r1], self.population[r2], self.population[r3], f[:, np.newaxis])
            v = self.project(v)
            u = np.array([self.crossover(v[i], self.population[i], cw[i]) for i in range(pop_size)])
            u = self.project(u)
            new_scores = np.apply_along_axis(fitness, 1, u, params)
            improved_mask = new_scores < scores
            self.population[improved_mask] = u[improved_mask]
            scores[improved_mask] = new_scores[improved_mask]

            u_rand = np.random.uniform(size=(pop_size, 4))
            f_update_mask = u_rand[:, 1] < tau1
            cw_update_mask = u_rand[:, 3] < tau2
            f[f_update_mask] = f_low + u_rand[f_update_mask, 0] * f_up
            cw[cw_update_mask] = u_rand[cw_update_mask, 2]

            pos_min = np.argmin(scores)
            max_dist = np.max(np.linalg.norm(self.population - self.population[pos_min], axis=1))
            if verbose:
                if gen % 100 == 0:
                    # print(f'gen = {gen}, max_dist = {max_dist:.6f}, best_solution = {self.population[pos_min]}')
                    # print(f'gen = {gen}, \t max_dist = {max_dist:.6f}')
                    print(f'gen = {gen}, \t fitness = {fitness(self.population[pos_min], params)}')
                    # print(f'gen = {gen}')
            if history:
                hist_vec.append(fitness(self.population[pos_min], params))
            if callback is not None:
                callback(self.population[pos_min])
            gen += 1
        return self.population[pos_min], max_dist, gen, np.array(hist_vec)

    def _select_random_indices(self, pop_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select three mutually exclusive random index arrays for DE mutation."""
        r1 = np.random.choice(pop_size, pop_size, replace=False)
        r2 = np.random.choice(pop_size, pop_size, replace=False)
        r3 = np.random.choice(pop_size, pop_size, replace=False)

        # Ensure unique indices
        while np.any((r1 == r2) | (r1 == r3) | (r2 == r3)):
            mask = (r1 == r2) | (r1 == r3)
            r1[mask] = np.random.choice(pop_size, np.sum(mask), replace=False)
            mask = (r2 == r3)
            r2[mask] = np.random.choice(pop_size, np.sum(mask), replace=False)

        return r1, r2, r3




class GrassmannGIPCAEstimator:
    def __init__(self, num_assets, num_fact, num_charact, num_macro, win_len):
        self.num_assets = num_assets      # N
        self.grass_n = num_charact        # m
        self.grass_k = num_fact           # k
        self.num_macro = num_macro        # l
        self.dim = self.grass_n * self.grass_k
        self.win_len = win_len

    @staticmethod
    def _proj_complement(Lambda_t: np.ndarray) -> np.ndarray:
        N, k = Lambda_t.shape
        XtX = Lambda_t.T @ Lambda_t
        XtX_inv = np.linalg.inv(XtX)
        P = Lambda_t @ XtX_inv @ Lambda_t.T
        return np.eye(N) - P

    def _estimate_delta(self, w: np.ndarray, data, ridge: float = 0.0):
        rets, Z, mu = data
        T = self.win_len
        k = self.grass_k
        l = self.num_macro

        G = np.zeros((k * l, k * l), dtype=float)
        B = np.zeros((k, l), dtype=float)
        c = 0.0

        for t in range(T):
            Z_t = Z[t, :, :]          # (N, m)
            r_t = rets[t, :]          # (N,)
            mu_t = mu[t, :]           # (l,)

            Lambda_t = Z_t @ w        # (N, k)
            Q_t = self._proj_complement(Lambda_t)

            c += r_t @ (Q_t @ r_t)

            A_t = Lambda_t.T @ Q_t @ Lambda_t       # (k, k)
            d_t = Lambda_t.T @ Q_t @ r_t            # (k,)

            B += np.outer(d_t, mu_t)                # (k, l)
            G += np.kron(np.outer(mu_t, mu_t), A_t) # (kl, kl)

        if ridge > 0.0:
            G = G + ridge * np.eye(k * l)

        b = B.reshape(k * l, order="F")  # vec(B)

        vec_Delta = np.linalg.solve(G, b)
        Delta_hat = vec_Delta.reshape((k, l), order="F")

        return Delta_hat, G, b, c

    def estimate_f0(self, w: np.ndarray, Delta: np.ndarray, data):
        """
        Compute f0_hat[t] = (Lambda_t^T Lambda_t)^{-1} Lambda_t^T (r_t - Lambda_t Delta mu_t)
        for each t, given w (=Gamma^T) and Delta.

        Returns:
            f0_hat: (T, k)
        """
        rets, Z, mu = data
        T = self.win_len
        k = self.grass_k

        assert Delta.shape == (k, self.num_macro)

        f0_hat = np.zeros((T, k), dtype=float)

        for t in range(T):
            Z_t = Z[t, :, :]          # (N, m)
            r_t = rets[t, :]          # (N,)
            mu_t = mu[t, :]           # (l,)

            Lambda_t = Z_t @ w        # (N, k)
            rhs = r_t - (Lambda_t @ (Delta @ mu_t))  # (N,)

            # Numerically stable LS solve for Lambda_t f0 ≈ rhs
            # (equivalent to (Lambda^T Lambda)^{-1} Lambda^T rhs when full rank)
            f0_hat[t, :], *_ = np.linalg.lstsq(Lambda_t, rhs, rcond=None)

        return f0_hat

    def loss_fct(self, w, data):
        w = np.asarray(w)
        if w.ndim == 1:
            w = w.reshape(self.grass_n, self.grass_k)
        assert w.shape == (self.grass_n, self.grass_k)

        rets, Z, mu = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)
        assert mu.shape == (self.win_len, self.num_macro)

        try:
            Delta_hat, G, b, c = self._estimate_delta(w, data)
        except np.linalg.LinAlgError:
            return 1e18

        vec_Delta = Delta_hat.reshape(self.grass_k * self.num_macro, order="F")
        quad = float(b @ vec_Delta)  # b^T G^{-1} b

        return (c - quad) / self.win_len

    def fit(self, data, max_gen=500, ridge=0.0):
        rets, Z, mu = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)
        assert mu.shape == (self.win_len, self.num_macro)

        w_min = np.full(self.dim, -1.0)
        w_max = np.ones(self.dim)
        pop_size = 5 * self.dim

        def loss_wrapped(w, params):
            if ridge == 0.0:
                return self.loss_fct(w, params)
            # ridge-stabilized variant
            w_arr = np.asarray(w).reshape(self.grass_n, self.grass_k)
            try:
                Delta_hat, G, b, c = self._estimate_delta(w_arr, params, ridge=ridge)
            except np.linalg.LinAlgError:
                return 1e18
            vec_Delta = Delta_hat.reshape(self.grass_k * self.num_macro, order="F")
            quad = float(b @ vec_Delta)
            return (c - quad) / self.win_len

        de = JDifferentialEvolution(
            w_min, w_max, pop_size,
            model='grassmannian',
            grass_k=self.grass_k
        )

        w_opt, max_dist, max_gen, history = de.optimize(
            loss_wrapped, params=data,
            eps=1e-3, max_gen=max_gen,
            history=True, verbose=True
        )

        W = np.asarray(w_opt).reshape(self.grass_n, self.grass_k)

        # Final Delta and f0 at optimum
        Delta_hat, *_ = self._estimate_delta(W, data, ridge=ridge)
        f0_hat = self.estimate_f0(W, Delta_hat, data)

        print(f"W (Gamma^T):\n{W}")
        print(f"Delta_hat:\n{Delta_hat}")
        print(f"f0_hat shape: {f0_hat.shape}")
        print(f"max_dist: {max_dist}")
        print(f"max_gen: {max_gen}")
        print(f"profiled objective: {self.loss_fct(W, data=data)}")

        plt.figure(figsize=(8, 5))
        plt.plot(history, label='Objective function')
        plt.title("GIPCA (profiled over Delta and f^0)")
        plt.xlabel("Generation (g)")
        plt.ylabel("Objective function (f)")
        plt.legend()
        plt.show()

        # Return all three: Gamma^T, Delta, and the profiled factor series f0
        return W, Delta_hat, f0_hat, history


    def loss_with_ridge(self, w, data, ridge=1e-6):
        w = np.asarray(w).reshape(self.grass_n, self.grass_k)
        try:
            Delta_hat, G, b, c = self._estimate_delta(w, data, ridge=ridge)
        except np.linalg.LinAlgError:
            return 1e18

        vec_Delta = Delta_hat.reshape(self.grass_k * self.num_macro, order="F")
        quad = float(b @ vec_Delta)
        return (c - quad) / self.win_len


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


# ______________________________________________________________________________________________________________

if __name__ == '__main__':

    # Test code
    seed = 6890
    np.random.seed(seed)
    num_assets = 40  # N
    num_fact = 5  # k
    num_charact = 25    # m
    num_macro = 3   # l
    win_len = 21  # T
    max_gen = 2_000

    # Simple data
    # rets = np.random.normal(size=(win_len, num_assets))
    # Z = np.random.uniform(size=(win_len, num_assets, num_charact))
    # mu = np.random.uniform(size=(win_len, num_macro))
    # data = [rets, Z, mu]

    # Hard data
    include_intercept = False
    data, _ = generate_gipca_data(T=win_len, N=num_assets, m=num_charact, k=num_fact, num_macro=num_macro,
                                 include_intercept=include_intercept, seed=seed)

    est = GrassmannGIPCAEstimator(num_assets, num_fact, num_charact, num_macro, win_len)
    est.fit(data, max_gen=max_gen)
