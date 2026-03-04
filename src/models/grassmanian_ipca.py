import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, Optional, Tuple, List
import autograd.numpy as anp
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.function import autograd as pymanopt_autograd
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions
from src.generate.ipca import generate_ipca_data
from src.utils import subspace_error, recovery_report


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



class GrassmannIPCAEstimator:
    def __init__(self, num_assets, num_fact, num_charact, win_len):
        self.num_assets = num_assets  # N
        self.grass_n = num_charact  # m
        self.grass_k = num_fact  # k
        self.dim = self.grass_n * self.grass_k
        self.win_len = win_len

    def loss_fct(self, w, data):
        w = np.asarray(w)
        if w.ndim == 1:
            w = w.reshape(self.grass_n, self.grass_k)
        if w.ndim == 2:
            assert w.shape == (self.grass_n, self.grass_k)

        rets, Z = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)

        obj = 0.0
        for t in range(self.win_len):
            Z_t = Z[t, :, :]  # (N, m)
            Lambda_t = Z_t @ w  # (N, k)
            beta, *_ = np.linalg.lstsq(Lambda_t, rets[t, :], rcond=None)  # (k,)
            fit = Lambda_t @ beta  # (N,)
            resid = rets[t, :] - fit
            obj += resid @ resid

        return obj / self.win_len

    def fit(self, data, max_gen=500):
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
        print(f"W:\n{W}")
        print(f"max_dist: {max_dist}")
        print(f"max_gen: {max_gen}")
        print(f"objective function: {self.loss_fct(W, data=data)}")

        plt.figure(figsize=(8, 5))
        plt.plot(history, label='Objective function')
        plt.title("IPCA")
        plt.xlabel("Generation (g)")
        plt.ylabel("Objective function (f)")
        plt.legend()
        plt.show()

        return W, history


#______________________________________________________________________________________________________________


class GrassmannManifoldIPCAEstimator:
    """
    IPCA on the Grassmann manifold using pymanopt.

    Optimization variable:
        W = Gamma^T in R^{m x k} with orthonormal columns (Grassmann representative)

    Objective (profiled IPCA loss):
        L(W) = (1/T) sum_t min_f || r_t - (Z_t W) f ||^2

    Post-estimation:
        1) f_hat[t] = argmin_f || r_t - (Z_t W) f ||^2  (cross-sectional LS)
        2) beta_t (optional) = f_hat[t]  (in your codebase, this is the factor return time series)

    Returns (mirrors the GIPCA manifold estimator style):
        W, f_hat, history
    Optionally also returns the raw pymanopt result object.
    """

    def __init__(self, num_assets, num_fact, num_charact, win_len):
        self.num_assets = num_assets      # N
        self.grass_n = num_charact        # m
        self.grass_k = num_fact           # k
        self.win_len = win_len

    # ------------------------------------------------------------------
    # Post-estimation helper (NumPy)
    # ------------------------------------------------------------------
    def estimate_f(self, W: np.ndarray, data) -> np.ndarray:
        rets, Z = data
        T = self.win_len
        k = self.grass_k
        f_hat = np.zeros((T, k), dtype=float)
        for t in range(T):
            Lambda_t = Z[t] @ W
            f_hat[t], *_ = np.linalg.lstsq(Lambda_t, rets[t], rcond=None)
        return f_hat

    # ------------------------------------------------------------------
    # Pymanopt fit (current pymanopt API)
    # ------------------------------------------------------------------
    def fit(
        self,
        data,
        optimizer: str = "ConjugateGradient",
        max_iterations: int = 200,
        verbosity: int = 1,
        log_verbosity: int = 2,
        initial_point: np.ndarray | None = None,
        reuse_line_searcher: bool = False,
        return_pymanopt_result: bool = False,
        truth: dict = None,
    ):
        """
        Fit using pymanopt.

        Parameters:
          data: [rets, Z]
              rets: (T, N)
              Z:    (T, N, m)
          truth: dict, optional
              Truth dictionary from generate_ipca_data. If provided (and
              log_verbosity >= 2), tracks Gamma subspace error at each iteration.

        Returns:
          Wopt: (m, k)
          f_hat: (T, k)
          history: list of costs (best effort; from result.log if available)
          (optionally) result
        """
        rets, Z = data
        assert rets.shape == (self.win_len, self.num_assets)
        assert Z.shape == (self.win_len, self.num_assets, self.grass_n)

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

        if getattr(result, "log", None) is not None:
            iters = result.log.get("iterations", None)
            if isinstance(iters, dict):
                if "cost" in iters:
                    history = [float(c) for c in iters["cost"]]

                # Compute Gamma recovery from logged iterates
                if truth is not None and "point" in iters:
                    W_star = truth["W_star"]
                    for W_iter in iters["point"]:
                        self.gamma_error_history.append(
                            subspace_error(np.asarray(W_iter), W_star))

        if not history:
            history = [float(cost(Wopt))]

        # ---- post-estimation ----
        f_hat = self.estimate_f(Wopt, data)

        if return_pymanopt_result:
            return Wopt, f_hat, history, result
        return Wopt, f_hat, history


#______________________________________________________________________________________________________________

if __name__ == '__main__':

    # Test code
    seed = 6890
    # seed = 156
    np.random.seed(seed)
    num_assets = 500  # N
    num_fact = 5  # k
    num_charact = 25  # m
    win_len = 63  # T

    # Generate simple data ()
    # rets = np.random.normal(size=(win_len, num_assets))
    # Z = np.random.uniform(size=(win_len, num_assets, num_charact))
    # data = [rets, Z]

    # Generate hard data
    include_intercept = False
    data, truth = generate_ipca_data(T=win_len, N=num_assets, m=num_charact, k=num_fact,
                                      include_intercept=include_intercept, seed=seed)

    # Differential Evolution estimator
    # est_de = GrassmannIPCAEstimator(num_assets, num_fact, num_charact, win_len)
    # max_gen = 3_000
    # est_de.fit(data, max_gen=max_gen)

    # IPCA Manifold estimator
    est_manifold = GrassmannManifoldIPCAEstimator(
        num_assets=num_assets,
        num_fact=num_fact,
        num_charact=num_charact,
        win_len=win_len
    )

    Wopt, f_hat, history = est_manifold.fit(
        data=data,
        optimizer="ConjugateGradient",
        max_iterations=200,
        verbosity=2,
        truth=truth,
    )

    print("Wopt:", Wopt)  # (m, k)
    print("f_hat:", f_hat)  # (T, k)
    print("Final loss:", history[-1])

    # Recovery report
    recovery_report(Wopt, truth)

    # =========================================================================
    # Plots: objective + Gamma error over iterations
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(len(history)), history)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Objective")
    ax1.set_title("Pymanopt IPCA convergence")

    ax2.plot(range(len(est_manifold.gamma_error_history)),
             est_manifold.gamma_error_history)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Grassmann distance")
    ax2.set_title("Gamma subspace error vs W*")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()

