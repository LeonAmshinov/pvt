import numpy as np
from equil_rdy import equil_max2p_ss, equil_ss, equil_max2p_newton, equil_max2p_ss_newton
from stability_rdy import stability_newton, stability_succesive_substitution, stability_ss_newton
from eos_pr_rdy import lnphi_Z
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=np.inf)

#Two phase equil
#Example 1
# T = 273.15 + 10
# P = 6e6
# yi = np.array([0.9, 0.1])
# Pci = np.array([7.37646, 4.600155]) * 1e6
# Tci = np.array([304.2, 190.6])
# wi = np.array([.225, .008])
# mwi = np.array([0.04401, 0.016043])
# vsi = np.array([0., 0.])
# dij = np.array([[0., .025], [.025, 0.]])
# print(equil_max2p_newton(P, T, yi, Pci, Tci, wi, dij, vsi, mwi))
# yji, fj, Zj = equil_max2p_ss_newton(P, T, yi, Pci, Tci, wi, dij, vsi, mwi)
# print(equil_max2p_ss_newton(P, T, yi, Pci, Tci, wi, dij, vsi, mwi))
# print(stability_newton(P, T, yji[0], Pci, Tci, wi, dij, vsi))
# print(yji, fj, Zj)


# xj1 = np.linspace(1e-4, 0.9999, 1000, endpoint=True)
# xji = np.vstack([xj1, 1 - xj1]).T
# Gj = []
# for j, xi in enumerate(xji):
#     lnphi, Z = lnphi_Z(xi, P, T, Pci, Tci, wi, dij, vsi, 1)
#     lnfi = lnphi + np.log(xi * P)
#     Gj.append(xi.dot(lnfi))
# lnfi = lnphi_Z(yji[0], P, T, Pci, Tci, wi, dij, vsi, 1)[0] + np.log(yji[0] * P) #Логарифм летучести от компонентного состава первой фазы
# Lj = xji.dot(lnfi)

# fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
# ax1.plot(xj1, Gj, lw=2, c = 'teal', zorder=2, label='Приведенная добавочная энергия Гиббса')
# ax1.plot(xj1, Lj, lw=2, c='orchid', zorder=2, label='Касательная')
# ax1.set_xlim(0, 1)
# ax1.set_xlabel('Количество вещества диоксида углерода в первой фазе, моль')
# ax1.set_ylabel('Приведенная добавочная энергия Гиббса')
# ax1.grid(zorder=1)
# plt.show()

# Dj = Gj - Lj
# fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
# ax2.plot(xj1, Dj, lw=2, c='lime', zorder=2)
# ax2.set_xlim(0, 1)
# ax2.set_ylim(0, 1)
# ax2.set_xlabel('Количества вещества диоксида углерода в первой фазе, моль')
# ax2.set_ylabel('Tangent plane distance (TPD)')
# plt.show()


#Three phase equil
#Example 2
# P = 101325
# T = 273.15 + 20
# yi = np.array([0.1, 0.6, 0.3])
# Pci = np.array([4.600155, 3.2890095, 22.04832]) * 1e6 # Critical pressures [Pa]
# Tci = np.array([190.6, 507.5, 647.3]) # Critical temperatures [K]
# wi = np.array([0.008, 0.27504, 0.344]) # Acentric factors
# mwi = np.array([0.016043, 0.086, 0.018015]) # Molar mass [kg/gmole]
# vsi = np.array([0., 0., 0.]) # Volume shift parameters
# dij = np.array([[0., .0253, .4907], [.0253, 0, 0.48], [0.4907, 0.48, 0]]) # Binary interaction parameters
# yji = equil_max2p_ss(P, T, yi, Pci, Tci, wi, dij, vsi, mwi, level=1)[0]
# print(stability_newton(P, T, yji[0], Pci, Tci, wi, dij, vsi, 1, level=1, pure_eps=1e-21, pure_ind=1))
# print(equil_ss(P, T, yi, Pci, Tci, wi, dij, vsi, mwi, level=1, pure_eps=1e-21, pure_ind=1, eps_flash=1e-8, eps2_stab=1e-8))
# print(equil_max2p_ss_newton(P, T, yi, Pci, Tci, wi, dij, vsi, mwi, level=1, pure_eps=1e-21, pure_ind=1))

#Example 3
P = 17e6
T = 273.15 + 68
yi = np.array([.7167, .0895, .0917, .0448, .0573])
Pci = np.array([4.599, 4.872, 4.248, 3.796, 2.398]) * 1e6 # Critical pressures [Pa]
Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02]) # Critical temperatures [K]
wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414]) # Acentric factors
mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120]) # Molar mass [kg/gmole]
vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661]) # Volume shift parameters
dij = np.array([
    [0.000000, 0.002689, 0.008537, 0.014748, 0.039265],
    [0.002689, 0.000000, 0.001662, 0.004914, 0.021924],
    [0.008537, 0.001662, 0.000000, 0.000866, 0.011676],
    [0.014748, 0.004914, 0.000866, 0.000000, 0.006228],
    [0.039265, 0.021924, 0.011676, 0.006228, 0.000000]]) # Binary interaction parameters

P = 7e6
T = 223.15
yi = np.array([0.0392035, 0.4310085, 0.016666, 0.0066165, 0.003006,
               0.002318, 0.0009595, 0.000216, 6e-6, 0.5])
Pci = np.array([32.789, 45.4, 48.2, 41.9, 37.022, 33.008, 20.794, 16.727,
                11.698, 217.6]) * 101325.
Tci = np.array([122.141, 190.6, 305.4, 369.8, 419.746, 482.781, 547.033,
                674.011, 833.580, 647.3])
wi = np.array([0.02946, 0.008, 0.098, 0.152, 0.18767, 0.25456, 0.31931,
               0.32841, 0.43005, 0.344])
vsi = np.array([-0.15670, -0.18633, -0.15018, -0.11403, -0.07789,
                -0.02655, 0.01541, 0.10440, 0.22653, 0.232])
mwi = np.array([27.385, 16.043, 30.070, 44.097, 58.124, 77.671, 103.418,
                185.177, 298.021, 18.015]) / 1e3
dij = np.array([[ 0.0000e+00,  3.0000e-04,  3.5000e-04,  4.3000e-04,  1.2100e-03,  1.5000e-03,  1.7600e-03,  4.1800e-03,  6.3000e-03,  1.1000e-01,],
                    [ 3.0000e-04,  0.0000e+00,  5.6600e-03,  1.8840e-02,  2.8840e-02,  4.2750e-02,  6.0610e-02,  1.5956e-01,  1.9705e-01,  4.9070e-01,],
                    [ 3.5000e-04,  5.6600e-03,  0.0000e+00,  2.9600e-03,  8.7400e-03,  1.8120e-02,  3.1130e-02,  1.0787e-01,  1.3549e-01,  5.4690e-01,],
                    [ 4.3000e-04,  1.8840e-02,  2.9600e-03,  0.0000e+00,  8.7000e-04,  8.2400e-03,  2.7580e-02,  7.4260e-02,  8.4060e-02,  4.8000e-01,],
                    [ 1.2100e-03,  2.8840e-02,  8.7400e-03,  8.7000e-04,  0.0000e+00,  2.1100e-03,  1.3290e-02,  4.2770e-02,  5.1040e-02,  4.8000e-01,],
                    [ 1.5000e-03,  4.2750e-02,  1.8120e-02,  8.2400e-03,  2.1100e-03,  0.0000e+00,  5.6400e-03,  2.3510e-02,  2.6270e-02,  4.8000e-01,],
                    [ 1.7600e-03,  6.0610e-02,  3.1130e-02,  2.7580e-02,  1.3290e-02,  5.6400e-03,  0.0000e+00, -1.2600e-03, -1.2840e-02,  4.8000e-01,],
                    [ 4.1800e-03,  1.5956e-01,  1.0787e-01,  7.4260e-02,  4.2770e-02,  2.3510e-02, -1.2600e-03,  0.0000e+00, -1.0870e-02,  4.8000e-01,],
                    [ 6.3000e-03,  1.9705e-01,  1.3549e-01,  8.4060e-02,  5.1040e-02,  2.6270e-02, -1.2840e-02, -1.0870e-02,  0.0000e+00,  4.8000e-01,],
                    [ 1.1000e-01,  4.9070e-01,  5.4690e-01,  4.8000e-01,  4.8000e-01,  4.8000e-01,  4.8000e-01,  4.8000e-01,  4.8000e-01,  0.0000e+00,]])
# print(stability_newton(P, T, yi, Pci, Tci, wi, dij, vsi, 1))
# print(equil_ss(P, T, yi, Pci, Tci, wi, dij, vsi, mwi, eps_flash=1e-6))
# print(stability_ss_newton(P, T, yi, Pci, Tci, wi, dij, vsi, 1))
# print(equil_max2p_newton(P, T, yi, Pci, Tci, wi, dij, vsi, mwi))
print(equil_max2p_ss_newton(P, T, yi, Pci, Tci, wi, dij, vsi, mwi))
# print(equil_ss(P, T, yi, Pci, Tci, wi, dij, vsi, mwi))