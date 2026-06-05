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


""" Build the control-point influence matrix C """
def build_C(control_pts, R, order="C2"):
    dist = cdist(control_pts, control_pts)
    return wendland(dist / R, order)


""" Solve for RBF weights - gamma """
def solve_weights(C: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.linalg.solve(C, f)


""" Evaluate the interpolation at all points to find deformation """
def evaluate(eval_pts, control_pts, gamma, R, order="C2"):
    dist = cdist(eval_pts, control_pts)
    A = wendland(dist / R, order)
    return A @ gamma


""" Apply deformation to the mesh  """
def deform(volume_pts, control_pts, surface_disp, R, order="C2"):
    C = build_C(control_pts, R, order)
    cond = np.linalg.cond(C)
    gamma = solve_weights(C, surface_disp)
    volume_disp = evaluate(volume_pts, control_pts, gamma, R, order)
    deformed = volume_pts + volume_disp
    info = {"R": R, "cond": cond, "N": control_pts.shape[0], "order": order}
    return deformed, info
