'''
Loading raw data files into arrays etc 
 
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
 
    @property
    def n_points(self) -> int:
        return self.points.shape[0]
 
    @property
    def grid(self) -> np.ndarray:
        return self.points.reshape(self.nj, self.ni, 3) 
 
    @property
    def surface(self) -> np.ndarray:
        """ Get aerofoil """
        return self.grid[0]
 
    @property
    def farfield(self) -> np.ndarray:
        """ Get boundary """
        return self.grid[-1]
 
""" Read O mesh file """
def load_plt(path: str | Path) -> VolumeMesh:

    path = Path(path)
    with path.open() as f:
        lines = f.readlines()
 
    
    ni = nj = None
    data_start = 0
    for idx, line in enumerate(lines):
        upper = line.upper()
        if "ZONE" in upper:
            for tok in upper.replace("=", " = ").split():
                pass  
            """ Get mesh I, J correctly formatted"""
            ni = _extract_dim(upper, "I") 
            nj = _extract_dim(upper, "J") 
            data_start = idx + 1
            break
 
    rows = [ln.split() for ln in lines[data_start:] if ln.strip()]
    points = np.array(rows, dtype=float)
 
    return VolumeMesh(points=points, ni=ni, nj=nj)
 
 
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
 
"""Helper function for"""
def _extract_dim(zone_line_upper: str, key: str) -> int | None:
    toks = zone_line_upper.replace("=", " = ").split()
    for k, tok in enumerate(toks):
        if tok == key and k + 2 < len(toks) and toks[k + 1] == "=":
            return int(toks[k + 2])
    return None