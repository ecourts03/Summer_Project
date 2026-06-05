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

from rbf_deform import (load_plt, load_xyz, to_2d, deform, write_plt,
                        orthogonality_change)

MESH_DIR = Path(__file__).resolve().parents[1] / "Mesh files"
VOLUME_FILE = MESH_DIR / "NACA0012257x129.plt"
SURFACE_FILE = MESH_DIR / "surface257.xyz"
OUT_DIR = Path(__file__).resolve().parents[1] / "results"

ORDERS = ["C0", "C2", "C4"]

""" Support-radius sampling. The interesting behaviour (steep collapse in
    orthogonality, explosion in conditioning) all happens at low R, so sample
    finely there and coarsely across the flat plateau beyond. """
R_MIN = 0.2          # smallest support radius
R_SPLIT = 3.0        # boundary between the steep region and the plateau
R_MAX = 50.0         # largest support radius
N_FINE = 7           # points in the steep region (R_MIN .. R_SPLIT)
N_COARSE = 10        # points on the plateau (R_SPLIT .. R_MAX)

""" Linear: dense low-R, sparse plateau """
R_LIN = np.unique(np.concatenate([
    np.linspace(R_MIN, R_SPLIT, N_FINE),
    np.linspace(R_SPLIT, R_MAX, N_COARSE),
]))

""" Log: naturally dense at low R already, but include the split point too """
R_LOG = np.unique(np.concatenate([
    np.geomspace(R_MIN, R_SPLIT, N_FINE),
    np.geomspace(R_SPLIT, R_MAX, N_COARSE),
]))

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

ANGLE = -35.0       # deformation: rotation angle (deg). Change as needed.


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


""" Plot a quantity vs R, one curve per order. xlog sets log x-axis. """
def plot_vs_R(rows, ykey, ylabel, title, path, logy=False, xlog=True):
    colours = {"C0": "#1f77b4", "C2": "#ff7f0e", "C4": "#2ca02c"}
    markers = {"C0": "o", "C2": "s", "C4": "^"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for order in ORDERS:
        sub = [r for r in rows if r["order"] == order]
        sub.sort(key=lambda r: r["R"])
        Rs = [r["R"] for r in sub]
        ys = [r[ykey] for r in sub]
        ax.plot(Rs, ys, marker=markers[order], color=colours[order],
                ms=6, lw=2, label=f"Wendland {order}", markeredgecolor="white",
                markeredgewidth=0.5)
    if xlog:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Support radius, R (chords)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.95, edgecolor="0.7")
    ax.grid(True, which="major", alpha=0.4, lw=0.8)
    ax.grid(True, which="minor", alpha=0.15, lw=0.5)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {path}")


""" Combined dual-axis plot: mean orthogonality change (left) and condition
    number (right) vs R, one pair of curves per order. Shows the quality vs
    numerical-cost trade-off in a single figure. """
def plot_combined(rows, title, path, xlog=False):
    colours = {"C0": "#1f77b4", "C2": "#ff7f0e", "C4": "#2ca02c"}
    markers = {"C0": "o", "C2": "s", "C4": "^"}

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()   # second y-axis sharing the same x

    for order in ORDERS:
        sub = sorted([r for r in rows if r["order"] == order], key=lambda r: r["R"])
        Rs = [r["R"] for r in sub]
        oc = [r["Q_change_mean"] for r in sub]
        cond = [r["cond"] for r in sub]
        # solid line = orthogonality change (left axis)
        ax1.plot(Rs, oc, marker=markers[order], color=colours[order], ms=6, lw=2,
                 ls="-", label=f"{order} ortho", markeredgecolor="white",
                 markeredgewidth=0.5)
        # dashed line = condition number (right axis)
        ax2.plot(Rs, cond, marker=markers[order], color=colours[order], ms=5, lw=1.5,
                 ls="--", alpha=0.6)

    if xlog:
        ax1.set_xscale("log")
    ax2.set_yscale("log")   # condition number always log

    ax1.set_xlabel("Support radius, R (chords)", fontsize=13)
    ax1.set_ylabel("Mean orthogonality change  (solid)", fontsize=13)
    ax2.set_ylabel("Condition number  (dashed)", fontsize=13)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, framealpha=0.95, edgecolor="0.7", loc="lower right")
    ax1.grid(True, which="major", alpha=0.4, lw=0.8)
    ax1.tick_params(labelsize=11)
    ax2.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {path}")


""" Pad an interior-node field (nj-2, ni) up to the full grid (nj, ni) by
    copying the nearest interior layer onto the two boundary rings, so every
    node has a value and the grid renders intact in Tecplot. Returns it flat. """
def pad_to_full_grid(field_interior, nj, ni):
    full = np.empty((nj, ni))
    full[1:nj - 1, :] = field_interior
    full[0, :] = field_interior[0, :]        # surface ring = first interior layer
    full[nj - 1, :] = field_interior[-1, :]  # farfield ring = last interior layer
    return full.reshape(-1)


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

    """ --- 2. Plots (linear-spaced R) --- """
    print("Plots:")
    plot_vs_R(rows_lin, "cond", "Condition number",
              f"Condition number vs support radius ({abs(ANGLE):.0f}\u00b0 rotation)",
              OUT_DIR / f"plot_{tag}_cond.png", logy=True, xlog=False)
    plot_vs_R(rows_lin, "Q_change_mean", "Mean orthogonality change",
              f"Mean orthogonality change vs support radius ({abs(ANGLE):.0f}\u00b0)",
              OUT_DIR / f"plot_{tag}_orthochange.png", xlog=False)
    plot_vs_R(rows_lin, "Q_change_max_abs", "Max |orthogonality change|",
              f"Worst-node orthogonality change vs support radius ({abs(ANGLE):.0f}\u00b0)",
              OUT_DIR / f"plot_{tag}_maxchange.png", logy=True, xlog=False)

    """ Combined trade-off plot: orthogonality change + condition number """
    plot_combined(rows_lin,
                  f"Orthogonality vs conditioning trade-off ({abs(ANGLE):.0f}\u00b0)",
                  OUT_DIR / f"plot_{tag}_combined.png", xlog=False)

    """ --- 3. Tecplot contours of orthogonality CHANGE for chosen points ---
        We contour the change dq = q_deformed - q_original, which shows WHERE the
        deformation degraded the mesh (near zero everywhere except distorted
        regions). Negative dq = orthogonality got worse. The absolute deformed
        orthogonality is written too, for reference. """
    from rbf_deform import orthogonality_field
    print("Tecplot contour dumps (orthogonality change):")
    out_xyz = np.zeros((vol.shape[0], 3))
    q_orig_int = orthogonality_field(orig_grid)            # interior field, undeformed
    for order, R in CONTOUR_POINTS:
        deformed, info = deform(vol, ctrl, disp, R, order=order)
        def_grid = deformed.reshape(mesh.nj, mesh.ni, 2)

        q_def_int = orthogonality_field(def_grid)
        dq_int = q_def_int - q_orig_int                    # the change (interior)

        dq_full = pad_to_full_grid(dq_int, mesh.nj, mesh.ni)
        q_full = pad_to_full_grid(q_def_int, mesh.nj, mesh.ni)

        out_xyz[:, 0] = deformed[:, 0]
        out_xyz[:, 2] = deformed[:, 1]
        fname = OUT_DIR / f"ortho_{tag}_{order}_R{R:g}.plt"
        write_plt(fname, out_xyz, mesh.ni, mesh.nj, mesh.nk,
                  extra_vars={"OrthoChange": dq_full, "Orthogonality": q_full},
                  zone_title=f"{order}_R{R:g}")
        print(f"  {order} R={R:<5g} cond={info['cond']:.2e}  "
              f"min dq={dq_int.min():.4f} -> {fname.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()