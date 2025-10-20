import numpy as np
from eos_pr_rdy import lnphii_dP, lnphi_Z, lnphii_dnj
from stability_rdy import stability_succesive_substitution, stability_newton, stability_ss_newton, init_ki
from R_R_rdy import R_R_2p, R_R_mp
from Psat_rdy import Psat_newton
from equil_rdy import equil_max2p_newton
c=1
R = 8.3144598

# P = np.array([137.4, 122.4, 107.2, 91.8, 76.4, 61.2, 46.5, 31.1, 16.8, 1]) * 1e5
P = np.array([42.3, 50.1, 59.8, 80.1, 100.2, 117.6, 125.7, 131.00, 137.4, 152.8, 168.6, 190.7]) * 1e5


T = 13.8 + 273.15
yi = np.array([0.078407, 0.862017, 0.033332, 0.013233, 0.006012, 0.004636, 0.001919, 0.000432, 1.2e-05])
Pci = np.array([33.2235, 46.00155, 48.83865, 42.455175, 37.51241281725, 33.44538741075, 21.0695330643, 16.948980861572338, 11.85307138676045]) * 1e5
Tci = np.array([122.14146, 190.6, 305.4, 369.8, 419.7463, 482.78094, 547.033131025227, 674.0114935582545, 833.5798537147791])
mwi = np.array([27.384656, 16.043, 30.07, 44.097, 58.124, 77.671481, 103.41845, 185.1765, 298.0208375]) * 1e-3
wi = np.array([0.029461732, 0.008, 0.098, 0.152, 0.18767265, 0.25456211, 0.3193064455028616, 0.3284096048633782, 0.43004714480647344])
vsi = np.array([-0.156705, -0.1863326246265798, -0.15018346207032843, -0.11403430435827364, -0.07788565407580418, -0.02654852577140842, 0.0154058838, 0.104404998, 0.2265338])
dij = np.array([[0, 0.000298294, 0.000348482, 0.000431578, 0.001206935, 0.00150003, 0.001760248, 0.004181806, 0.006300359],
[0.000298294, 0, 0.005658024, 0.018839823, 0.028842326, 0.042754906, 0.060614736, 0.159561576, 0.197051635],
[0.000348482, 0.005658024, 0, 0.00295732, 0.008744185, 0.018124696, 0.031126588, 0.107872142, 0.135493525],
[0.000431578, 0.018839823, 0.00295732, 0, 0.000865362, 0.00824275, 0.027580652, 0.074261372, 0.084064681],
[0.001206935, 0.028842326, 0.008744185, 0.000865362, 0, 0.002105814, 0.013288913, 0.04277189, 0.051036361],
[0.00150003, 0.042754906, 0.018124696, 0.00824275, 0.002105814, 0, 0.005641292, 0.023510033, 0.026270292],
[0.001760248, 0.060614736, 0.031126588, 0.027580652, 0.013288913, 0.005641292, 0, -0.001263111, -0.01283853],
[0.004181806, 0.159561576, 0.107872142, 0.074261372, 0.04277189, 0.023510033, -0.001263111, 0, -0.01086615],
[0.006300359, 0.197051635, 0.135493525, 0.084064681, 0.051036361, 0.026270292, -0.01283853, -0.01086615, 0]])

n = 1.

def CVD(P, T, yi, n, Pci, Tci, wi, mwi, dij, vsi, c=1,
        upper=True, Pmin=1e3, Pmax=1e8, N_nodes=100,
        eps1_stab=1e-10, eps2_stab=1e-8, maxiter_stab=50, level=0, pure_ind=0, pure_eps=1e-3,
        eps_newton=1e-8, maxiter_newton=250,
        negative_flash=False, maxiter_flash=500, eps_flash=1e-8, method_stab='newton',
        eps_rr=1e-10, maxiter_rr=50):

  nt = n
  Psat, yji, ki, Zj, fj = Psat_newton(T, yi, Pci, Tci, wi, mwi, dij, vsi, c,
                                      upper, Pmin, Pmax, N_nodes,
                                      eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps,
                                      eps_newton, maxiter_newton)
  Vt = Zj.dot(fj) * nt * R * T / Psat
  where = P < Psat
  P = P[where]
  print(f'Pressure = {Psat/1e6:.3f} MPa')
  print(f'phase compositions:\n{yji}')
  for p in P:
    print(f'Pressure = {p/1e6:.3f} MPa')
    yji, fj, Zj = equil_max2p_newton(p, T, yi, Pci, Tci, wi, dij, vsi, mwi, c,
                                     negative_flash, maxiter_flash, eps_flash, method_stab,
                                     eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps,
                                     eps_rr, maxiter_rr)
    nj = fj * nt
    vj = Zj * R * T / p
    dn = (vj.dot(nj) - Vt) / vj[0]
    nt -=dn
    nj[0] -= dn
    nji = yji * nj[:, None]
    ni = nji.sum(axis=0)
    yi = ni / nt

# CVD(P, T, yi, n, Pci, Tci, wi, mwi, dij, vsi)


def CCE(P, T, yi, n, Pci, Tci, wi, mwi, dij, vsi, c=1,
        upper=True, Pmin=1e3, Pmax=1e8, N_nodes=100,
        eps1_stab=1e-10, eps2_stab=1e-8, maxiter_stab=50, level=0, pure_ind=0, pure_eps=1e-3,
        eps_newton=1e-8, maxiter_newton=250,
        negative_flash=False, maxiter_flash=500, eps_flash=1e-8, method_stab='newton',
        eps_rr=1e-10, maxiter_rr=50):
  nt = n
  for p in P:
    print(f'Pressure = {p/1e6:.3f} MPa')
    yji, fj, Zj = equil_max2p_newton(p, T, yi, Pci, Tci, wi, dij, vsi, mwi, c,
                                     negative_flash, maxiter_flash, eps_flash, method_stab,
                                     eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps,
                                     eps_rr, maxiter_rr)
CCE(P, T, yi, n, Pci, Tci, wi, mwi, dij, vsi)

import numpy as np
from eos_pr_rdy import lnphii_dP, lnphi_Z, lnphii_dnj
from stability_rdy import stability_succesive_substitution, stability_newton, stability_ss_newton, init_ki
from R_R_rdy import R_R_2p, R_R_mp
from Psat_rdy import Psat_newton
from equil_rdy import equil_max2p_newton
c=1
R = 8.3144598

# P = np.array([137.4, 122.4, 107.2, 91.8, 76.4, 61.2, 46.5, 31.1, 16.8, 1]) * 1e5
P = np.array([42.3, 50.1, 59.8, 80.1, 100.2, 117.6, 125.7, 131.00, 137.4, 152.8, 168.6, 190.7]) * 1e5


T = 13.8 + 273.15
yi = np.array([0.078407, 0.862017, 0.033332, 0.013233, 0.006012, 0.004636, 0.001919, 0.000432, 1.2e-05])
Pci = np.array([33.2235, 46.00155, 48.83865, 42.455175, 37.51241281725, 33.44538741075, 21.0695330643, 16.948980861572338, 11.85307138676045]) * 1e5
Tci = np.array([122.14146, 190.6, 305.4, 369.8, 419.7463, 482.78094, 547.033131025227, 674.0114935582545, 833.5798537147791])
mwi = np.array([27.384656, 16.043, 30.07, 44.097, 58.124, 77.671481, 103.41845, 185.1765, 298.0208375]) * 1e-3
wi = np.array([0.029461732, 0.008, 0.098, 0.152, 0.18767265, 0.25456211, 0.3193064455028616, 0.3284096048633782, 0.43004714480647344])
vsi = np.array([-0.156705, -0.1863326246265798, -0.15018346207032843, -0.11403430435827364, -0.07788565407580418, -0.02654852577140842, 0.0154058838, 0.104404998, 0.2265338])
dij = np.array([[0, 0.000298294, 0.000348482, 0.000431578, 0.001206935, 0.00150003, 0.001760248, 0.004181806, 0.006300359],
[0.000298294, 0, 0.005658024, 0.018839823, 0.028842326, 0.042754906, 0.060614736, 0.159561576, 0.197051635],
[0.000348482, 0.005658024, 0, 0.00295732, 0.008744185, 0.018124696, 0.031126588, 0.107872142, 0.135493525],
[0.000431578, 0.018839823, 0.00295732, 0, 0.000865362, 0.00824275, 0.027580652, 0.074261372, 0.084064681],
[0.001206935, 0.028842326, 0.008744185, 0.000865362, 0, 0.002105814, 0.013288913, 0.04277189, 0.051036361],
[0.00150003, 0.042754906, 0.018124696, 0.00824275, 0.002105814, 0, 0.005641292, 0.023510033, 0.026270292],
[0.001760248, 0.060614736, 0.031126588, 0.027580652, 0.013288913, 0.005641292, 0, -0.001263111, -0.01283853],
[0.004181806, 0.159561576, 0.107872142, 0.074261372, 0.04277189, 0.023510033, -0.001263111, 0, -0.01086615],
[0.006300359, 0.197051635, 0.135493525, 0.084064681, 0.051036361, 0.026270292, -0.01283853, -0.01086615, 0]])

n = 1.

def CVD(P, T, yi, n, Pci, Tci, wi, mwi, dij, vsi, c=1,
        upper=True, Pmin=1e3, Pmax=1e8, N_nodes=100,
        eps1_stab=1e-10, eps2_stab=1e-8, maxiter_stab=50, level=0, pure_ind=0, pure_eps=1e-3,
        eps_newton=1e-8, maxiter_newton=250,
        negative_flash=False, maxiter_flash=500, eps_flash=1e-8, method_stab='newton',
        eps_rr=1e-10, maxiter_rr=50):

  nt = n
  Psat, yji, ki, Zj, fj = Psat_newton(T, yi, Pci, Tci, wi, mwi, dij, vsi, c,
                                      upper, Pmin, Pmax, N_nodes,
                                      eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps,
                                      eps_newton, maxiter_newton)
  Vt = Zj.dot(fj) * nt * R * T / Psat
  where = P < Psat
  P = P[where]
  print(f'Pressure = {Psat/1e6:.3f} MPa')
  print(f'phase compositions:\n{yji}')
  for p in P:
    print(f'Pressure = {p/1e6:.3f} MPa')
    yji, fj, Zj = equil_max2p_newton(p, T, yi, Pci, Tci, wi, dij, vsi, mwi, c,
                                     negative_flash, maxiter_flash, eps_flash, method_stab,
                                     eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps,
                                     eps_rr, maxiter_rr)
    nj = fj * nt
    vj = Zj * R * T / p
    dn = (vj.dot(nj) - Vt) / vj[0]
    nt -=dn
    nj[0] -= dn
    nji = yji * nj[:, None]
    ni = nji.sum(axis=0)
    yi = ni / nt

CVD(P, T, yi, n, Pci, Tci, wi, mwi, dij, vsi)


def CCE(P, T, yi, n, Pci, Tci, wi, mwi, dij, vsi, c=1,
        upper=True, Pmin=1e3, Pmax=1e8, N_nodes=100,
        eps1_stab=1e-10, eps2_stab=1e-8, maxiter_stab=50, level=0, pure_ind=0, pure_eps=1e-3,
        eps_newton=1e-8, maxiter_newton=250,
        negative_flash=False, maxiter_flash=500, eps_flash=1e-8, method_stab='newton',
        eps_rr=1e-10, maxiter_rr=50):
  nt = n
  for p in P:
    print(f'Pressure = {p/1e6:.3f} MPa')
    yji, fj, Zj = equil_max2p_newton(p, T, yi, Pci, Tci, wi, dij, vsi, mwi, c,
                                     negative_flash, maxiter_flash, eps_flash, method_stab,
                                     eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps,
                                     eps_rr, maxiter_rr)
CCE(P, T, yi, n, Pci, Tci, wi, mwi, dij, vsi)
