'''
Created by Ewan Courts on 02/06/2026
Initial testing for the RBF function by testing roatation 
'''

from pathlib import Path
import numpy as np

from rbf_deform import load_plt, load_xyz, to_2d, deform, write_plt

MESH_DIR = Path(__file__).resolve().parents[1] / "Mesh files"
VOLUME_FILE = MESH_DIR / "NACA0012257x129.plt"
SURFACE_FILE = MESH_DIR / "surface257.xyz" 
OUT = MESH_DIR / "5deg.plt"

ANGLE_DEG = -5                 
CENTRE = np.array([0.0, 0.0])    
R = 0.2                          


""" Rotate the initial mesh """
def rotate(pts, angle_deg, centre):
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    shifted = pts - centre
    rot = np.empty_like(shifted)
    rot[:, 0] = shifted[:, 0] * c - shifted[:, 1] * s
    rot[:, 1] = shifted[:, 0] * s + shifted[:, 1] * c
    return rot + centre


""" Full RBF mesh deformation """
def main():
    """ Control points from the surface file, volume points from the mesh """
    ctrl = to_2d(load_xyz(SURFACE_FILE))
    mesh = load_plt(VOLUME_FILE)
    vol = to_2d(mesh.points)

    """ Prescribe the rotation as a surface displacement """
    rotated = rotate(ctrl, ANGLE_DEG, CENTRE)
    disp = rotated - ctrl

    """ RBF deform the whole mesh """
    deformed, info = deform(vol, ctrl, disp, R)
    print(f"Angle = {ANGLE_DEG} deg, R = {R}")
    print(f"Control points N = {info['N']}, condition number = {info['cond']:.3e}")
    print(f"Max surface displacement = {np.abs(disp).max():.4f}")

    """ Tecplot layout """
    out_xyz = np.zeros((deformed.shape[0], 3))
    out_xyz[:, 0] = deformed[:, 0]
    out_xyz[:, 2] = deformed[:, 1]

    """ Displacement contour to see radius influence / if it even worked  """
    vol_disp = deformed - vol
    dmag = np.sqrt(vol_disp[:, 0] ** 2 + vol_disp[:, 1] ** 2)

    write_plt(OUT, out_xyz, mesh.ni, mesh.nj, mesh.nk,
              extra_vars={"DispMag": dmag}, zone_title="5deg_def")
    print(f"Written: {OUT}")


if __name__ == "__main__":
    main()