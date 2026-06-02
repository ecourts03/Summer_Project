"""
Writing numpy arrays into tecplot Formatting

"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_plt(
    path: str | Path,
    points: np.ndarray,
    ni: int,
    nj: int,
    nk: int = 1,
    extra_vars: dict[str, np.ndarray] | None = None,
    zone_title: str | None = None,
) -> None:
    """Write a structured grid as an ASCII Tecplot POINT-format file.

    Parameters
    ----------
    path : str or Path
        Output file path (conventionally ending in ``.plt`` or ``.dat``).
    points : (N, 3) float array
        Grid points in file order (i fastest, then j, then k). Must have
        ``N == ni * nj * nk`` rows. Pass deformed coordinates here.
    ni, nj, nk : int
        Structured grid dimensions. ``nk`` defaults to 1 for the 2D case.
    extra_vars : dict of name -> (N,) array, optional
        Additional per-point scalar fields to write as extra variables, so
        Tecplot can contour/colour by them. Each array must have N entries in
        the same point order as ``points``.
    zone_title : str, optional
        Optional title for the zone (purely cosmetic in Tecplot).

    Notes
    -----
    Point ordering is preserved exactly as given -- this function does not
    reorder anything. It is the caller's responsibility to keep points in
    i-fastest order, which is what :func:`load_plt` and the ``VolumeMesh``
    reshape already maintain.
    """
    path = Path(path)
    points = np.asarray(points, dtype=float)

    n_expected = ni * nj * nk
    if points.shape[0] != n_expected:
        raise ValueError(
            f"point count {points.shape[0]} != ni*nj*nk = {n_expected} "
            f"(I={ni}, J={nj}, K={nk})"
        )
    if points.shape[1] != 3:
        raise ValueError(f"points must have 3 columns (X Y Z), got {points.shape[1]}")

    # Variable names: the three coordinates, plus any extras.
    var_names = ["X", "Y", "Z"]
    extra_cols: list[np.ndarray] = []
    if extra_vars:
        for name, arr in extra_vars.items():
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.shape[0] != n_expected:
                raise ValueError(
                    f"extra variable '{name}' has {arr.shape[0]} values, "
                    f"expected {n_expected}"
                )
            var_names.append(name)
            extra_cols.append(arr)

    # Full data block: coordinates then any extra columns, side by side.
    if extra_cols:
        block = np.column_stack([points, *extra_cols])
    else:
        block = points

    # Header.
    vars_str = " ".join(f'"{v}"' for v in var_names)
    zone_line = f"ZONE I={ni}    J={nj}    K={nk}    F=POINT"
    if zone_title:
        zone_line = f'ZONE T="{zone_title}"    I={ni}    J={nj}    K={nk}    F=POINT'

    # Write. Full double precision (matches the supplied file's 16 decimals).
    with path.open("w") as f:
        f.write(f"VARIABLES = {vars_str}\n")
        f.write(zone_line + "\n")
        for row in block:
            f.write("    " + "    ".join(f"{val:.16f}" for val in row) + "\n")