'''
Created by Ewan Courts on 26/05/2026
Files into arrays etc 
Should work for 2D and 3D

'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

""" Create shortcuts using data class  """
@dataclass
class VolumeMesh:

    points: np.ndarray 
    ni: int
    nj: int
    nk: int = 1

    @property
    def is_3d(self) -> bool:
        return self.nk > 1

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def grid(self) -> np.ndarray:
        """ Reshape to (nk, nj, ni, 3) """
        return self.points.reshape(self.nk, self.nj, self.ni, 3)

    @property
    def surface(self) -> np.ndarray:
        """ Get aerofoil (j=0) """
        block = self.grid[:, 0, :, :]
        if not self.is_3d:
            return block[0]
        return block

    @property
    def farfield(self) -> np.ndarray:
        """ Get boundary (j=nj-1)"""
        block = self.grid[:, -1, :, :]
        if not self.is_3d:
            return block[0]
        return block

    @property
    def surface_points(self) -> np.ndarray:
        """ Aerofoil as flat (M,3) list - RBF control points"""
        return self.surface.reshape(-1, 3)

    @property
    def farfield_points(self) -> np.ndarray:
        """ Boundary as flat (M,3) list """
        return self.farfield.reshape(-1, 3)

""" Read O mesh file """
def load_plt(path: str | Path) -> VolumeMesh:

    path = Path(path)
    with path.open() as f:
        lines = f.readlines()

    ni = nj = nk = None
    data_start = 0
    for idx, line in enumerate(lines):
        upper = line.upper()
        if "ZONE" in upper:
            """ Get mesh I, J, K correctly formatted """
            ni = _extract_dim(upper, "I") 
            nj = _extract_dim(upper, "J") 
            nk = _extract_dim(upper, "K")
            data_start = idx + 1
            break

    
    if nk is None:
        nk = 1

    rows = [ln.split() for ln in lines[data_start:] if ln.strip()]
    points = np.array(rows, dtype=float)

    """ Check the file isn't broken"""
    if points.shape[0] != ni * nj * nk:
        raise ValueError(f"expected {ni*nj*nk} points (I={ni} J={nj} K={nk}), read {points.shape[0]}")

    return VolumeMesh(points=points, ni=ni, nj=nj, nk=nk)


""" Read Surface Points NACA0012"""
def load_xyz(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open() as f:
        lines = f.readlines()

    declared = int(lines[0].split()[0])
    rows = [ln.split() for ln in lines[1:] if ln.strip()]
    pts = np.array(rows, dtype=float)

    return pts

""" Files sent in 3D with no data change to 2D (also check if 3D data available )"""
def to_2d(points: np.ndarray) -> np.ndarray:
    if not np.allclose(points[:, 1], 0.0):
        raise ValueError()
    return points[:, [0, 2]]

"""Helper function for extracting I/J/K"""
def _extract_dim(zone_line_upper: str, key: str) -> int | None:
    toks = zone_line_upper.replace("=", " = ").split()
    for k, tok in enumerate(toks):
        if tok == key and k + 2 < len(toks) and toks[k + 1] == "=":
            return int(toks[k + 2])
    return None