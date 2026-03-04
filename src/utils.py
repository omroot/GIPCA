
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import svd, subspace_angles
from sklearn.metrics import mean_squared_error, r2_score


def subspace_error(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the subspace error (Grassmann distance) between column spaces of A and B.

    Parameters
    ----------
    A : np.ndarray
        First matrix
    B : np.ndarray
        Second matrix (same number of columns as A)

    Returns
    -------
    float
        Grassmann distance between the column spaces
    """
    # Orthonormalize both matrices
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)

    # Compute singular values of Qa' @ Qb
    s = svd(Qa.T @ Qb, compute_uv=False)

    # Clamp for numerical stability
    s = np.clip(s, -1, 1)

    # Grassmann distance
    return np.sqrt(np.sum(np.arccos(s)**2))


def recovery_report(W_hat: np.ndarray, truth: dict, Delta_hat: np.ndarray = None) -> Dict[str, Any]:
    """
    Print recovery metrics comparing estimated parameters to ground truth.

    Parameters
    ----------
    W_hat : np.ndarray
        Estimated Gamma matrix (L x K)
    truth : dict
        Truth dictionary from generate_ipca_data or generate_gipca_data.
        Must contain 'W_star'. May contain 'Delta_star'.
    Delta_hat : np.ndarray, optional
        Estimated Delta matrix (K x R). If provided and truth contains
        'Delta_star', prints relative Delta error.

    Returns
    -------
    Dict with keys: 'grassmann_dist', 'principal_angles_deg', 'rel_delta_error' (if applicable)
    """
    W_star = truth["W_star"]

    # Grassmann distance
    grass_dist = subspace_error(W_hat, W_star)

    # Principal angles (radians -> degrees)
    ang = subspace_angles(W_hat, W_star)
    ang_deg = np.degrees(ang)

    print("\n=== Recovery Report ===")
    print(f"Grassmann distance:      {grass_dist:.6f}")
    print(f"Principal angles (deg):  {np.array2string(ang_deg, precision=4, separator=', ')}")
    print(f"Max principal angle:     {float(np.max(ang_deg)):.6f} deg")
    print(f"RMS principal angle:     {float(np.sqrt(np.mean(ang_deg**2))):.6f} deg")

    result = {
        "grassmann_dist": grass_dist,
        "principal_angles_deg": ang_deg,
    }

    # Delta recovery (Grassmann distance — rotation-invariant)
    if Delta_hat is not None and "Delta_star" in truth:
        Delta_star = truth["Delta_star"]
        if Delta_hat.shape != Delta_star.shape:
            print(f"Delta shape mismatch: hat {Delta_hat.shape} vs star {Delta_star.shape}")
        else:
            delta_dist = subspace_error(Delta_hat, Delta_star)
            print(f"Delta Grassmann dist:    {delta_dist:.6f}")
            result["delta_grassmann_dist"] = delta_dist

    return result

