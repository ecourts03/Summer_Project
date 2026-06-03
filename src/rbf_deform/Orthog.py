'''
Created by Ewan Courts on 03/06/2026
Calculating Mesh quality - orthogonality following methodology found in  (Rendall & Allen 2008).
'''

from __future__ import annotations

import numpy as np


""" Find cos^2 of the angle between two vectors """
def _cos2(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    dot = np.sum(va * vb, axis=-1)
    na2 = np.sum(va * va, axis=-1)
    nb2 = np.sum(vb * vb, axis=-1)
    return dot ** 2 / (na2 * nb2)


""" Local orthogonality q 2D for now """
def orthogonality_field(grid_2d: np.ndarray, wrap_i: bool = True) -> np.ndarray:
    nj, ni, _ = grid_2d.shape

    """ Surface and farfield rings excluded """
    j = np.arange(1, nj - 1)

    if wrap_i:
        ip = (np.arange(ni) + 1) % ni
        im = (np.arange(ni) - 1) % ni
        ip[ni - 1] = 1       
        im[0] = ni - 2        
        icols = np.arange(ni)
    else:
        icols = np.arange(1, ni - 1)
        ip = icols + 1
        im = icols - 1

    centre = grid_2d[np.ix_(j, icols)]
    v1 = grid_2d[np.ix_(j, ip)] - centre        
    v3 = grid_2d[np.ix_(j, im)] - centre        
    v2 = grid_2d[np.ix_(j + 1, icols)] - centre  
    v4 = grid_2d[np.ix_(j - 1, icols)] - centre  

    """  sum cos^2 """
    q_plane = _cos2(v1, v2) + _cos2(v2, v3) + _cos2(v3, v4) + _cos2(v4, v1)

    """ 2D: one plane """
    q_local = 1.0 - q_plane
    return q_local


""" Global orthogonality Q """
def global_orthogonality(grid_2d: np.ndarray, wrap_i: bool = True) -> float:
    return float(np.mean(orthogonality_field(grid_2d, wrap_i)))


""" Change in orthogonality from original """
def orthogonality_change(orig_grid_2d: np.ndarray, deformed_grid_2d: np.ndarray,
                         wrap_i: bool = True, return_field: bool = False):
    q0 = orthogonality_field(orig_grid_2d, wrap_i)
    q1 = orthogonality_field(deformed_grid_2d, wrap_i)
    dq = q1 - q0  

    info = {
        "Q_orig": float(np.mean(q0)),
        "Q_deformed": float(np.mean(q1)),
        "Q_change_mean": float(np.mean(dq)),
        "Q_change_max_abs": float(np.max(np.abs(dq))),
        "q_min_deformed": float(np.min(q1)),   
    }
    if return_field:
        return info, dq
    return info