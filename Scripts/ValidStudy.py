'''
Support-radius and basis-order study (next major step after OBJ1)

For each Wendland order (C0, C2, C4) and a sweep of support radius R, records:
  - condition number of the RBF matrix      (numerical stability)
  - orthogonality of the deformed mesh        (mesh quality, before/after)

Outputs:
  1. CSV of all results (log and linear R spacing)        -> results/
  2. Plots: condition number vs R, orthogonality vs R      -> results/
  3. Tecplot contour files of the per-node orthogonality
     for a chosen handful of (R, order) points             -> results/
     (so the orthogonality field can be visualised in Tecplot)

Deformation is passed in as a function, so the study runs on any prescribed
surface motion unchanged.

Run from the repo root:
    python Scripts/04_support_radius_study.py
'''

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rbf_deform import (load_plt, load_xyz, to_2d, deform, write_plt, orthogonality_change)

MESH_DIR = Path(__file__).resolve().parents[1] / "Mesh files"
VOLUME_FILE = MESH_DIR / "NACA0012257x129.plt"
SURFACE_FILE = MESH_DIR / "surface256.xyz"
OUT_DIR = Path(__file__).resolve().parents[1] / "results"

ORDERS = ["C0", "C2", "C4"]
R_LOG = np.geomspace(0.2, 50.0, 20)
R_LIN = np.linspace(0.2, 50.0, 20)

""" Chosen (order, R) points to dump as Tecplot orthogonality contours.
    Picked to show the trade-off: local / sweet-spot / near-breakdown, plus
    a basis-order comparison at fixed R. Edit freely. """
CONTOUR_POINTS = [
    ("C2", 0.5),    # small R - local support, expect distortion
    ("C2", 5.0),    # mid R - plateau sweet spot
    ("C2", 30.0),   # large R - smooth but near conditioning breakdown
    ("C0", 5.0),    # basis-order comparison at fixed R
    ("C4", 5.0),
]

ANGLE = -5.0        # deformation: rotation angle (deg). Change as needed.


""" Rotation deformation -> returns surface displacement """
def rotation(ctrl, angle_deg, centre=np.array([0.0, 0.0])):
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    shifted = ctrl - centre
    rot = np.empty_like(shifted)
    rot[:, 0] = shifted[:, 0] * c - shifted[:, 1] * s
    rot[:, 1] = shifted[:, 0] * s + shifted[:, 1] * c
    return (rot + centre) - ctrl


""" Run the R x order sweep, return rows of results """
def run_sweep(vol, ctrl, orig_grid, nj, ni, disp, R_values, spacing_label):
    rows = []
    for order in ORDERS:
        for R in R_values:
            deformed, info = deform(vol, ctrl, disp, R, order=order)
            def_grid = deformed.reshape(nj, ni, 2)
            q = orthogonality_change(orig_grid, def_grid)
            rows.append({
                "spacing": spacing_label, "order": order, "R": R,
                "cond": info["cond"],
                "Q_orig": q["Q_orig"], "Q_deformed": q["Q_deformed"],
                "Q_change_mean": q["Q_change_mean"],
                "Q_change_max_abs": q["Q_change_max_abs"],
                "q_min_deformed": q["q_min_deformed"],
            })
    return rows


""" Write rows to CSV """
def write_csv(rows, path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV  -> {path}")


""" Plot a quantity vs R, one curve per order (log x-axis) """
def plot_vs_R(rows, ykey, ylabel, title, path, logy=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    for order in ORDERS:
        sub = [r for r in rows if r["order"] == order]
        sub.sort(key=lambda r: r["R"])
        Rs = [r["R"] for r in sub]
        ys = [r[ykey] for r in sub]
        ax.plot(Rs, ys, "-o", ms=3, label=order)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Support radius R")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {path}")


""" Build a full-grid orthogonality field (boundary rings padded) for Tecplot.
    orthogonality_field covers interior nodes only; pad the two boundary rings
    with the nearest interior value so every node has a value and the grid
    renders intact. """
def padded_ortho_field(def_grid, nj, ni):
    from rbf_deform import orthogonality_field
    q_int = orthogonality_field(def_grid)          # (nj-2, ni)
    full = np.empty((nj, ni))
    full[1:nj - 1, :] = q_int
    full[0, :] = q_int[0, :]        # surface ring = first interior layer
    full[nj - 1, :] = q_int[-1, :]  # farfield ring = last interior layer
    return full.reshape(-1)         # flat, matching point order


def main():
    OUT_DIR.mkdir(exist_ok=True)

    mesh = load_plt(VOLUME_FILE)
    vol = to_2d(mesh.points)
    ctrl = to_2d(load_xyz(SURFACE_FILE))
    orig_grid = vol.reshape(mesh.nj, mesh.ni, 2)

    disp = rotation(ctrl, ANGLE)
    tag = f"rot{abs(ANGLE):.0f}deg"
    print(f"Deformation: {ANGLE} deg rotation, N = {ctrl.shape[0]}")

    """ --- 1. Sweeps + CSV --- """
    print("\nLog sweep:")
    rows_log = run_sweep(vol, ctrl, orig_grid, mesh.nj, mesh.ni, disp, R_LOG, "log")
    write_csv(rows_log, OUT_DIR / f"study_{tag}_log.csv")

    print("Linear sweep:")
    rows_lin = run_sweep(vol, ctrl, orig_grid, mesh.nj, mesh.ni, disp, R_LIN, "linear")
    write_csv(rows_lin, OUT_DIR / f"study_{tag}_linear.csv")

    """ --- 2. Plots (use log-spaced data) --- """
    print("Plots:")
    plot_vs_R(rows_log, "cond", "Condition number",
              f"Condition number vs R ({ANGLE} deg)",
              OUT_DIR / f"plot_{tag}_cond.png", logy=True)
    plot_vs_R(rows_log, "Q_deformed", "Deformed orthogonality Q",
              f"Orthogonality vs R ({ANGLE} deg)",
              OUT_DIR / f"plot_{tag}_ortho.png")
    plot_vs_R(rows_log, "Q_change_max_abs", "Max |orthogonality change|",
              f"Worst-node orthogonality change vs R ({ANGLE} deg)",
              OUT_DIR / f"plot_{tag}_maxchange.png", logy=True)

    """ --- 3. Tecplot orthogonality contours for chosen points --- """
    print("Tecplot contour dumps:")
    out_xyz = np.zeros((vol.shape[0], 3))
    for order, R in CONTOUR_POINTS:
        deformed, info = deform(vol, ctrl, disp, R, order=order)
        def_grid = deformed.reshape(mesh.nj, mesh.ni, 2)
        qfull = padded_ortho_field(def_grid, mesh.nj, mesh.ni)
        out_xyz[:, 0] = deformed[:, 0]
        out_xyz[:, 2] = deformed[:, 1]
        fname = OUT_DIR / f"ortho_{tag}_{order}_R{R:g}.plt"
        write_plt(fname, out_xyz, mesh.ni, mesh.nj, mesh.nk,
                  extra_vars={"Orthogonality": qfull},
                  zone_title=f"{order}_R{R:g}")
        print(f"  {order} R={R:<5g} cond={info['cond']:.2e} -> {fname.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()