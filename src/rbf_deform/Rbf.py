'''
Created by Ewan Courts on 28/05/2026
Building the RBF 
Uses Wendland C2 basis function 
Method - Solves influence matrix
'''

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


""" Building Wendland compact basis functions - expanded for orders C0,C2,C4"""
def wendland(r: np.ndarray, order: str = "C2") -> np.ndarray:
    r = np.asarray(r, dtype=float)
    phi = np.zeros_like(r)
    inside = r < 1.0
    rr = r[inside]
    one_minus = 1.0 - rr
    if order == "C0":
        phi[inside] = one_minus ** 2
    elif order == "C2":
        phi[inside] = one_minus ** 4 * (4.0 * rr + 1.0)
    elif order == "C4":
        phi[inside] = one_minus ** 6 * (35.0 * rr ** 2 + 18.0 * rr + 3.0) / 3.0
    else:
        raise ValueError(f"unknown Wendland order '{order}' (use C0, C2 or C4)")
    return phi
 
 
""" Dont break existing code """
def wendland_c2(r: np.ndarray) -> np.ndarray:
    return wendland(r, "C2")


""" Build the influence Matrix C """
def build_C(control_pts: np.ndarray, R: float) -> np.ndarray:
    dist = cdist(control_pts, control_pts)
    return wendland_c2(dist / R)


""" Solve for RBF weights - gamma """
def solve_weights(C: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.linalg.solve(C, f)


""" Evaluate the interpolation at all points to find deformation """
def evaluate(eval_pts: np.ndarray, control_pts: np.ndarray, gamma: np.ndarray, R: float) -> np.ndarray:
    dist = cdist(eval_pts, control_pts)
    A = wendland_c2(dist / R)
    return A @ gamma


""" Apply deformation to the mesh  """
def deform(volume_pts: np.ndarray, control_pts: np.ndarray, surface_disp: np.ndarray, R: float):
    C = build_C(control_pts, R)
    gamma = solve_weights(C, surface_disp)
    volume_disp = evaluate(volume_pts, control_pts, gamma, R)
    deformed = volume_pts + volume_disp

    """ Condition number  """
    cond = np.linalg.cond(C)

    info = {"R": R, "cond": cond, "N": control_pts.shape[0]}
    return deformed, info