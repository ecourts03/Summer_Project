'''
Baseline RBF mesh deformation (OBJ1)

Follows the Module 2 formulation (Allen & Poole):
  - Wendland C2 basis, radius normalised by support radius R   (slide 189)
        phi(r) = (1-r)^4 (4r+1)   for r < 1,   else 0,   r = ||x - xi|| / R
  - System  C gamma = f  solved for weights, one solve per coordinate  (slide 180/181)
  - Evaluate anywhere:  s(x) = sum_i gamma_i phi(||x - xi|| / R)        (slide 182)
  - No polynomial term: compact support keeps farfield untouched when R < gap
'''

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


""" Wendland C2 basis function (slide 189) """
def wendland_c2(r: np.ndarray) -> np.ndarray:
    """ r already normalised by support radius. Zero beyond r >= 1 (compact). """
    r = np.asarray(r, dtype=float)
    phi = np.zeros_like(r)
    inside = r < 1.0
    rr = r[inside]
    phi[inside] = (1.0 - rr) ** 4 * (4.0 * rr + 1.0)
    return phi


""" Build the control-point influence matrix C (slide 180) """
def build_C(control_pts: np.ndarray, R: float) -> np.ndarray:
    """ C[i,j] = phi(||xi - xj|| / R). Symmetric, N x N. """
    dist = cdist(control_pts, control_pts)
    return wendland_c2(dist / R)


""" Solve for RBF weights, one column of f per coordinate (slide 181) """
def solve_weights(C: np.ndarray, f: np.ndarray) -> np.ndarray:
    """ C gamma = f. f is (N, nd) displacements, returns gamma (N, nd). """
    return np.linalg.solve(C, f)


""" Evaluate the interpolation at arbitrary points (slide 182) """
def evaluate(eval_pts: np.ndarray, control_pts: np.ndarray, gamma: np.ndarray, R: float) -> np.ndarray:
    """ s = A gamma, where A[e,i] = phi(||xE_e - xi|| / R). Returns (EN, nd). """
    dist = cdist(eval_pts, control_pts)
    A = wendland_c2(dist / R)
    return A @ gamma


""" Full baseline deform: prescribe surface displacement, move all volume points """
def deform(volume_pts: np.ndarray, control_pts: np.ndarray, surface_disp: np.ndarray, R: float):
    """
    volume_pts   : (M, nd) all mesh points to move
    control_pts  : (N, nd) surface points (the RBF control points)
    surface_disp : (N, nd) prescribed displacement of each control point
    R            : support radius

    Returns (deformed_volume_pts, info) where info holds the condition number etc.
    """
    C = build_C(control_pts, R)

    """ Condition number for the support-radius study (slide 192) """
    cond = np.linalg.cond(C)

    gamma = solve_weights(C, surface_disp)
    volume_disp = evaluate(volume_pts, control_pts, gamma, R)
    deformed = volume_pts + volume_disp

    info = {"R": R, "cond": cond, "N": control_pts.shape[0]}
    return deformed, info