"""Inspect the NACA 0012 mesh: load both files, print a summary, and plot.

Run from the repo root:
    python scripts/01_inspect_mesh.py

Produces two figures:
  * the full grid (every Nth line, so it's legible) with the airfoil highlighted
  * a zoom on the airfoil surface
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rbf_deform.io import load_plt, load_xyz, to_2d

MESH_DIR = Path(__file__).resolve().parents[1] / "Mesh files"


def main() -> None:
    mesh = load_plt(MESH_DIR / "NACA0012257x129.plt")
    surf = load_xyz(MESH_DIR / "surface257.xyz")

    print(f"Volume grid : {mesh.ni} x {mesh.nj} = {mesh.n_points} points")
    print(f"Surface file: {surf.shape[0]} points")
    g = to_2d(mesh.points).reshape(mesh.nj, mesh.ni, 2)
    print(f"Domain extent: x [{g[..., 0].min():.1f}, {g[..., 0].max():.1f}], "
          f"y [{g[..., 1].min():.1f}, {g[..., 1].max():.1f}]")

    # --- Figure 1: full grid, decimated for legibility ---
    fig, ax = plt.subplots(figsize=(7, 7))
    step_i, step_j = 8, 4
    for j in range(0, mesh.nj, step_j):
        ax.plot(g[j, :, 0], g[j, :, 1], color="0.7", lw=0.4)
    for i in range(0, mesh.ni, step_i):
        ax.plot(g[:, i, 0], g[:, i, 1], color="0.7", lw=0.4)
    ax.plot(g[0, :, 0], g[0, :, 1], color="C3", lw=1.5, label="airfoil (j=0)")
    ax.set_aspect("equal")
    ax.set_title("NACA 0012 C-grid (decimated)")
    ax.legend()
    fig.savefig("mesh_full.png", dpi=130, bbox_inches="tight")

    # --- Figure 2: zoom on the airfoil ---
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    s2d = to_2d(surf)
    ax2.plot(s2d[:, 0], s2d[:, 1], "-o", color="C3", ms=2, lw=0.8)
    ax2.set_aspect("equal")
    ax2.set_title("Airfoil surface: 257 control-point candidates")
    fig2.savefig("airfoil_zoom.png", dpi=130, bbox_inches="tight")

    print("Saved mesh_full.png and airfoil_zoom.png")


if __name__ == "__main__":
    main()