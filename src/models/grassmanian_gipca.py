import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.function import autograd as pymanopt_autograd
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions
from src.generate.gipca import generate_gipca_data
from src.utils import subspace_error, recovery_report

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
        log_verbosity: int = 2,
        initial_point: np.ndarray | None = None,
        reuse_line_searcher: bool = False,
        return_pymanopt_result: bool = False,
        truth: dict = None,
    ):
        """
        Fit using current pymanopt API.

        Key points (current pymanopt):
          - Problem() has no verbosity kwarg.
          - Optimizers have verbosity + log_verbosity.
          - ConjugateGradient.run() does not accept callback.
          - History is obtained from result.log when log_verbosity >= 2
            (also stores intermediate points for recovery tracking).

        Parameters
        ----------
        truth : dict, optional
            Truth dictionary from generate_gipca_data. If provided (and
            log_verbosity >= 2), tracks Gamma subspace error and Delta
            relative error at each iteration.
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
        self.gamma_error_history = []
        self.delta_error_history = []

        if getattr(result, "log", None) is not None:
            iters = result.log.get("iterations", None)
            if isinstance(iters, dict):
                if "cost" in iters:
                    history = [float(c) for c in iters["cost"]]

                # Compute recovery errors from logged iterates
                if truth is not None and "point" in iters:
                    W_star = truth["W_star"]
                    Delta_star = truth.get("Delta_star")
                    for W_iter in iters["point"]:
                        W_np = np.asarray(W_iter)
                        self.gamma_error_history.append(
                            subspace_error(W_np, W_star))
                        if Delta_star is not None:
                            f_iter = self.estimate_f(W_np, data)
                            Delta_iter = self.estimate_delta(f_iter, mu, ridge_mu=ridge_mu)
                            self.delta_error_history.append(
                                subspace_error(Delta_iter, Delta_star))

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
        truth=truth,
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

    # Recovery report
    recovery_report(Wopt, truth, Delta_hat=Delta_hat)

    # =========================================================================
    # Plots: objective + Gamma error + Delta error over iterations
    # =========================================================================
    iters = range(len(history))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.plot(iters, history)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Objective")
    ax1.set_title("Pymanopt GIPCA convergence")

    ax2.plot(range(len(est_manifold.gamma_error_history)),
             est_manifold.gamma_error_history)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Grassmann distance")
    ax2.set_title("Gamma subspace error vs W*")
    ax2.set_yscale("log")

    ax3.plot(range(len(est_manifold.delta_error_history)),
             est_manifold.delta_error_history)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Grassmann distance")
    ax3.set_title("Delta subspace error vs Delta*")
    ax3.set_yscale("log")

    plt.tight_layout()
    plt.show()
