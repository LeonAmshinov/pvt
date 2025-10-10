import numpy as np
from eos_pr_rdy import lnphii_dP, lnphi_Z, lnphii_dnj
from stability_rdy import stability_succesive_substitution, stability_newton, stability_ss_newton, init_ki
from R_R_rdy import R_R_2p, R_R_mp

def bounds_Psat(T, zi, Pci, Tci, wi, dij, vsi, c=1, upper=True, Pmin=1e3, Pmax=1e8, N_nodes=100, 
                eps1=1e-10, eps2=1e-8, maxiter=50, level=0, pure_ind=0, pure_eps=1e-3):
    P_step = (Pmax - Pmin) / (N_nodes - 1)
    if upper:
        P = Pmax
        is_stable, kji, Z = stability_ss_newton(Pmax, T, zi, Pci, Tci, wi, dij, vsi, c)
        while is_stable and P >= Pmin:
            P -= P_step
            is_stable, kji, Z = stability_ss_newton(P, T, zi, Pci, Tci, wi, dij, vsi, c)
        Pmin = P
        Pmax = P + P_step
    else:
        P = Pmin
        is_stable, kji, Z = stability_ss_newton(Pmin, T, zi, Pci, Tci, wi, dij, vsi, c)
        while is_stable and P <= Pmax:
            P += P_step
            is_stable, kji, Z = stability_ss_newton(P, T, zi, Pci, Tci, wi, dij, vsi, c)
        Pmin = P - P_step
        Pmax = P
        # if Pmin < 0:
        #     Pmin = 1e-4
    print(Pmin, Pmax)
    return Pmin, Pmax, kji[0]

np.set_printoptions(linewidth=np.inf)

def ki_Psat_ss(Pmin, Pmax, T, ki, zi, Pci, Tci, wi, dij, vsi, c=1, upper=True, eps_ki=1e-8, maxiter_ki=250, eps_Psat=1e-8, maxiter_Psat=10):
    Yi = zi * ki
    yi = Yi / np.sum(Yi)
    if upper:
        Psat = Pmin
    else:
        Psat = Pmax
    Psat, lnphiyi, lnphizi, Zy, Zz, TPD = Psat_newton_in_ss(Psat, Pmin, Pmax, T, ki, zi, yi, Pci, Tci, wi, dij, vsi, c, upper, eps_Psat, maxiter_Psat)
    gi = np.log(ki) + lnphiyi - lnphizi
    gnorm = np.linalg.norm(gi)
    k = 0
    print(k, Psat / 1e6, gnorm, TPD)
    while (gnorm > eps_ki or np.abs(TPD) > eps_Psat) and k < maxiter_ki:
        ki = ki * np.exp(-gi)
        Yi = zi * ki
        yi = Yi / np.sum(Yi)
        Psat, lnphiyi, lnphizi, Zy, Zz, TPD = Psat_newton_in_ss(Psat, Pmin, Pmax, T, ki, zi, yi, Pci, Tci, wi, dij, vsi, c, upper, eps_Psat, maxiter_Psat)
        gi = np.log(ki) + lnphiyi - lnphizi
        gnorm = np.linalg.norm(gi)
        k += 1
        print(k, Psat / 1e6, gnorm, TPD)
    if gnorm < eps_ki and np.abs(TPD) < eps_Psat:
        rhoy = yi.dot(mwi) / Zy
        rhoz = zi.dot(mwi) / Zz
        if rhoy < rhoz:
            return Psat, np.vstack([yi, zi]), ki, np.array([Zy, Zz])
        else:
            return Psat, np.vstack([zi, yi]), ki, np.array([Zz, Zy])
    raise ValueError('Solution Psat not found')



def Psat_newton_in_ss(Psat, Pmin, Pmax, T, ki, zi, yi, Pci, Tci, wi, dij, vsi, c=1, upper=True, eps_Psat=1e-8, maxiter_Psat=10):
    lnphiyi, dlnphiiyidP, Zy = lnphii_dP(yi, Psat, T, Pci, Tci, wi, dij, vsi, c)
    lnphizi, dlnphiizidP, Zz = lnphii_dP(zi, Psat, T, Pci, Tci, wi, dij, vsi, c)
    TPD = (np.log(yi) + lnphiyi - lnphizi - np.log(zi)).dot(yi)
    n = 0
    # Psatp1 = Psat
    while np.abs(TPD) > eps_Psat and n < maxiter_Psat:
        dTPDdP = (dlnphiiyidP - dlnphiizidP).dot(yi)
        Psatp1 = Psat - TPD / dTPDdP
        if Psatp1 > Pmax or Psatp1 < Pmin:
            if upper:
                if TPD > 0:
                    Psat = (Psat + Pmin) * .5
                else:
                    Psat = (Psat + Pmax) * .5
            else:
                if TPD < 0:
                    Psat = (Psat + Pmin) * .5
                else:
                    Psat = (Psat + Pmax) * .5
        else:
            Psat = Psatp1
        lnphiyi, dlnphiiyidP, Zy = lnphii_dP(yi, Psat, T, Pci, Tci, wi, dij, vsi, c)
        lnphizi, dlnphiizidP, Zz = lnphii_dP(zi, Psat, T, Pci, Tci, wi, dij, vsi, c)
        TPD = (np.log(yi) + lnphiyi - lnphizi - np.log(zi)).dot(yi)
        n += 1
    return Psat, lnphiyi, lnphizi, Zy, Zz, TPD

def Psat_newton_in_newton(Psat, Pmin, Pmax, T, ki, zi, yi, Pci, Tci, wi, dij, vsi, c=1, upper=True, eps_Psat=1e-8, maxiter_Psat=10):
    lnphiyi, dlnphiiyidP, Zy = lnphii_dP(yi, Psat, T, Pci, Tci, wi, dij, vsi, c)
    lnphizi, dlnphiizidP, Zz = lnphii_dP(zi, Psat, T, Pci, Tci, wi, dij, vsi, c)
    TPD = (np.log(yi) + lnphiyi - lnphizi - np.log(zi)).dot(yi)
    n = 0
    # Psatp1 = Psat
    while np.abs(TPD) > eps_Psat and n < maxiter_Psat:
        dTPDdP = (dlnphiiyidP - dlnphiizidP).dot(yi)
        Psatp1 = Psat - TPD / dTPDdP
        if Psatp1 > Pmax or Psatp1 < Pmin:
            if upper:
                if TPD > 0:
                    Psat = (Psat + Pmin) * .5
                else:
                    Psat = (Psat + Pmax) * .5
            else:
                if TPD < 0:
                    Psat = (Psat + Pmin) * .5
                else:
                    Psat = (Psat + Pmax) * .5
        else:
            Psat = Psatp1
        lnphiyi, dlnphiiyidP, Zy = lnphii_dP(yi, Psat, T, Pci, Tci, wi, dij, vsi, c)
        lnphizi, dlnphiizidP, Zz = lnphii_dP(zi, Psat, T, Pci, Tci, wi, dij, vsi, c)
        TPD = (np.log(yi) + lnphiyi - lnphizi - np.log(zi)).dot(yi)
        n += 1
    return Psat, lnphiyi, lnphizi, dlnphiiyidP, dlnphiizidP, Zy, Zz, TPD

def Psat_ss(T, zi, Pci, Tci, wi, dij, vsi, c=1, upper=True, Pmin=1e3, Pmax=1e8, N_nodes=100, eps1=1e-10, eps2=1e-8, maxiter=50, level=0, pure_ind=0, pure_eps=1e-3,
         eps_ki=1e-8, maxiter_ki=250, eps_Psat=1e-8, maxiter_Psat=10):
    Pmin, Pmax, ki = bounds_Psat(T, zi, Pci, Tci, wi, dij, vsi, c, upper, Pmin, Pmax, N_nodes, eps1, eps2, maxiter, level, pure_ind, pure_eps)
    Psat, yj, ki, Zj = ki_Psat_ss(Pmin, Pmax, T, ki, zi, Pci, Tci, wi, dij, vsi, c, upper, eps_ki, maxiter_ki, eps_Psat, maxiter_Psat)
    return Psat, ki, yj, Zj

def Psat_newton(T, zi, Pci, Tci, wi, dij, vsi, c=1, upper=True, Pmin=1e3, Pmax=1e8, N_nodes=100, eps1=1e-10, eps2=1e-8, maxiter=50, level=0, pure_ind=0, pure_eps=1e-3,
                eps_newton=1e-8, maxiter_newton=250):
    Pmin, Pmax, ki = bounds_Psat(T, zi, Pci, Tci, wi, dij, vsi, c, upper, Pmin, Pmax, N_nodes, eps1, eps2, maxiter, level, pure_ind, pure_eps)
    Yi = zi * ki
    n = np.sum(Yi)
    yi = Yi / n
    lnzi = np.log(zi)
    if upper:
        Psat = Pmin
    else:
        Psat = Pmax
    Psat, lnphiyi, lnphizi, dlnphiiyidP, dlnphiizidP, Zy, Zz, TPD = Psat_newton_in_newton(Psat, Pmin, Pmax, T, ki, zi, yi, Pci, Tci, wi, dij, vsi, c, upper, eps_newton, maxiter_newton)
    print(Psat)
    dlnphiydnk = lnphii_dnj(yi, n, Psat, T, Pci, Tci, wi, dij, vsi, c)[1]
    Nc = yi.shape[0]
    gi = np.empty(Nc + 1)
    hi = np.log(yi) + lnphiyi - lnphizi - lnzi
    # gi[:Nc] = np.log(ki) + lnphiyi - lnphizi
    gi[:Nc] = hi + np.log(n)
    TPD = hi.dot(yi)
    gi[-1] = TPD
    J = np.empty((Nc + 1, Nc + 1))
    deltaij = np.identity(Nc)
    gnorm = np.linalg.norm(gi)
    k = 0
    while gnorm > eps_newton and k < maxiter_newton:
        k += 1
        dgidlnkj = deltaij + Yi * dlnphiydnk
        dTPDdlnkj = yi * (hi - TPD)
        dgidP = (dlnphiiyidP - dlnphiizidP)
        dgidlnP = Psat * dgidP
        dTPDdlnP = yi.dot(dgidlnP)
        J[:Nc, :Nc] = dgidlnkj
        J[-1, :Nc] = dTPDdlnkj
        J[:Nc, -1] = dgidlnP
        J[Nc, Nc] = dTPDdlnP
        X = -np.linalg.solve(J, gi)
        ki = ki * np.exp(X[:-1])
        Psat = Psat * np.exp(X[-1])
        print(f'{k = }')
        print(f'{ki = }')
        print(f'{Psat = }')
        print(f'{gnorm = }')
        print(f'{TPD = }')
        # quit()
        Yi = zi * ki
        n = np.sum(Yi)
        yi = Yi / n
        lnphiyi, dlnphiydnk, Zy = lnphii_dnj(yi, n, Psat, T, Pci, Tci, wi, dij, vsi, c)
        dlnphiiyidP = lnphii_dP(yi, Psat, T, Pci, Tci, wi, dij, vsi, c)[1]
        lnphizi, dlnphiizidP, Zz = lnphii_dP(zi, Psat, T, Pci, Tci, wi, dij, vsi, c)
        hi = np.log(yi) + lnphiyi - lnphizi - lnzi
        gi[:Nc] = hi + np.log(n)
        TPD = hi.dot(yi)
        gi[-1] = TPD
        gnorm = np.linalg.norm(gi)
    if gnorm < eps_newton:
        rhoy = yi.dot(mwi) / Zy
        rhoz = zi.dot(mwi) / Zz
        if rhoy < rhoz:
            return Psat, np.vstack([yi, zi]), ki, np.array([Zy, Zz])
        else:
            return Psat, np.vstack([zi, yi]), ki, np.array([Zz, Zy])
    raise ValueError('Solution Psat not found')
        # print(Psat)


#Test 21
# P = 60e6
# T = 373.15
# zi = np.array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
# Pci = np.array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
# Tci = np.array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
# mwi = np.array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
# wi = np.array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
# vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
# dij = np.array([
#     [0.00, 0.1200, 0.1200, 0.1200, 0.1200, 0.1200, 0.1200],
#     [0.12, 0.0000, 0.0051, 0.0207, 0.0405, 0.0611, 0.0693],
#     [0.12, 0.0051, 0.0000, 0.0053, 0.0174, 0.0321, 0.0384],
#     [0.12, 0.0207, 0.0053, 0.0000, 0.0035, 0.0117, 0.0156],
#     [0.12, 0.0405, 0.0174, 0.0035, 0.0000, 0.0024, 0.0044],
#     [0.12, 0.0611, 0.0321, 0.0117, 0.0024, 0.0000, 0.0003],
#     [0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003, 0.0000]])



#Test 01
# P = 1e6
T = 68. + 273.15
zi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
dij = np.array([
  [0.0000,  0.0027, 0.0085, 0.0147, 0.0393],
  [0.0027, 0.0000, 0.0017, 0.0049, 0.0219],
  [0.0085, 0.0017, 0.0000, 0.0009, 0.0117],
  [0.0147, 0.0049, 0.0009, 0.0000, 0.0062],
  [0.0393, 0.0219, 0.0117, 0.0062, 0.0000]
])

# Test 02
# P = 15e6
# T = 104.4 + 273.15
# zi = np.array([0.0091, 0.0016, 0.3647, 0.0967, 0.0695, 0.0144, 0.0393, 0.0144,
#                0.0141, 0.0433, 0.1320, 0.0757, 0.0510, 0.0315, 0.0427])
# Pci = np.array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
#                 32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
# Tci = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
#                 469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
# wi = np.array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
#                0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
# vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# mwi = np.array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
#                 72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
# dij = np.array([[ 0. ,   -0.02,   0.105,  0.13,   0.125,  0.12,   0.115,  0.115,  0.115,  0.115,  0.115,  0.115,  0.115,  0.115,  0.115],
# [-0.02 ,  0.    , 0.025  ,0.01  , 0.09   ,0.095 , 0.095  ,0.1    ,0.11   ,0.11   ,0.11   ,0.11   ,0.11   ,0.11   ,0.11 ],
# [ 0.105,  0.025 , 0.     ,0.003 , 0.009  ,0.016 , 0.015  ,0.021  ,0.021  ,0.025  ,0.039  ,0.067  ,0.072  ,0.107  ,0.133],
# [ 0.13 ,  0.01  , 0.003  ,0.    , 0.002  ,0.005 , 0.005  ,0.009  ,0.009  ,0.012  ,0.022  ,0.044  ,0.048  ,0.079  ,0.102],
# [ 0.125,  0.09  , 0.009  ,0.002 , 0.     ,0.001 , 0.001  ,0.003  ,0.003  ,0.005  ,0.012  ,0.029  ,0.032  ,0.059  ,0.08 ],
# [ 0.12 ,  0.095 , 0.016  ,0.005 , 0.001  ,0.    , 0.     ,0.     ,0.     ,0.001  ,0.006  ,0.019  ,0.022  ,0.045  ,0.064],
# [ 0.115,  0.095 , 0.015  ,0.005 , 0.001  ,0.    , 0.     ,0.001  ,0.001  ,0.001  ,0.006  ,0.02   ,0.023  ,0.047  ,0.066],
# [ 0.115,  0.1   , 0.021  ,0.009 , 0.003  ,0.    , 0.001  ,0.     ,0.     ,0.     ,0.003  ,0.014  ,0.017  ,0.038  ,0.055],
# [ 0.115,  0.11  , 0.021  ,0.009 , 0.003  ,0.    , 0.001  ,0.     ,0.     ,0.     ,0.003  ,0.015  ,0.017  ,0.038  ,0.055],
# [ 0.115,  0.11  , 0.025  ,0.012 , 0.005  ,0.001 , 0.001  ,0.     ,0.     ,0.     ,0.002  ,0.011  ,0.013  ,0.032  ,0.048],
# [ 0.115,  0.11  , 0.039  ,0.022 , 0.012  ,0.006 , 0.006  ,0.003  ,0.003  ,0.002  ,0.     ,0.004  ,0.005  ,0.02   ,0.033],
# [ 0.115,  0.11  , 0.067  ,0.044 , 0.029  ,0.019 , 0.02   ,0.014  ,0.015  ,0.011  ,0.004  ,0.     ,0.     ,0.006  ,0.014],
# [ 0.115,  0.11  , 0.072  ,0.048 , 0.032  ,0.022 , 0.023  ,0.017  ,0.017  ,0.013  ,0.005  ,0.     ,0.     ,0.004  ,0.012],
# [ 0.115,  0.11  , 0.107  ,0.079 , 0.059  ,0.045 , 0.047  ,0.038  ,0.038  ,0.032  ,0.02   ,0.006  ,0.004  ,0.     ,0.002],
# [ 0.115,  0.11  , 0.133  ,0.102 , 0.08   ,0.064 , 0.066  ,0.055  ,0.055  ,0.048  ,0.033  ,0.014  ,0.012  ,0.002  ,0.   ]])



# P = 17172545.454545613
# T = 273.15 + 68
# zi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
# Pci = np.array([4.599, 4.872, 4.248, 3.796, 2.398]) * 1e6 # Critical pressures [Pa]
# Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02]) # Critical temperatures [K]
# wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414]) # Acentric factors
# mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120]) # Molar mass [kg/gmole]
# vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661]) # Volume shift parameters
# dij = np.array([
#     [0.000000, 0.002689, 0.008537, 0.014748, 0.039265],
#     [0.002689, 0.000000, 0.001662, 0.004914, 0.021924],
#     [0.008537, 0.001662, 0.000000, 0.000866, 0.011676],
#     [0.014748, 0.004914, 0.000866, 0.000000, 0.006228],
#     [0.039265, 0.021924, 0.011676, 0.006228, 0.000000]
#     ])



# T = 13.8 + 273.15
# zi = np.array([0.078407, 0.862017, 0.033332, 0.013233, 0.006012, 0.004636,
#                0.001919, 0.000432, 1.2e-5])
# Pci = np.array([32.789, 45.4, 48.2, 41.9, 37.022, 33.008, 20.794, 16.727,
#                 11.698]) * 101325.
# Tci = np.array([122.141, 190.6, 305.4, 369.8, 419.746, 482.781, 547.033,
#                 674.011, 833.580])
# wi = np.array([0.02946, 0.008, 0.098, 0.152, 0.18767, 0.25456, 0.31931,
#                0.32841, 0.43005])
# vsi = np.array([-0.15670, -0.18633, -0.15018, -0.11403, -0.07789,
#                 -0.02655, 0.01541, 0.10440, 0.22653])
# mwi = np.array([27.385, 16.043, 30.070, 44.097, 58.124, 77.671, 103.418,
#                 185.177, 298.021]) / 1e3
# dij = np.array(
#     [[ 0.,       0.0003,   0.00035,  0.00043,  0.00121,  0.0015,   0.00176,  0.00418,  0.0063 ],
#  [ 0.0003 ,  0.     ,  0.00566,  0.01884,  0.02884,  0.04275,  0.06061,  0.15956,  0.19705],
#  [ 0.00035,  0.00566,  0.     ,  0.00296,  0.00874,  0.01812,  0.03113,  0.10787,  0.13549],
#  [ 0.00043,  0.01884,  0.00296,  0.     ,  0.00087,  0.00824,  0.02758,  0.07426,  0.08406],
#  [ 0.00121,  0.02884,  0.00874,  0.00087,  0.     ,  0.00211,  0.01329,  0.04277,  0.05104],
#  [ 0.0015 ,  0.04275,  0.01812,  0.00824,  0.00211,  0.     ,  0.00564,  0.02351,  0.02627],
#  [ 0.00176,  0.06061,  0.03113,  0.02758,  0.01329,  0.00564,  0.     , -0.00126, -0.01284],
#  [ 0.00418,  0.15956,  0.10787,  0.07426,  0.04277,  0.02351, -0.00126,  0.     , -0.01087],
#  [ 0.0063 ,  0.19705,  0.13549,  0.08406,  0.05104,  0.02627, -0.01284, -0.01087,  0.     ]
# ])

T = 92.2 + 273.15
zi = np.array([0.0001, 0.4227, 0.1166, 0.1006, 0.0266, 0.1909, 0.1025, 0.0400])
yi = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
fj = np.array([0., 0.1, 0.18, 0.31, 0.5, 0.7])
Pci = np.array([72.8, 45.31, 52.85, 39.81, 33.35, 27.27, 17.63, 10.34]) * 101325.
Tci = np.array([304.2, 190.1, 305.1, 391.3, 465.4, 594.5, 739.7, 886.2])
wi = np.array([0.225, 0.0082, 0.13, 0.1666, 0.2401, 0.3708, 0.6151, 1.0501])
vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
mwi = np.array([44.01, 16.136, 33.585, 49.87, 72.151, 123.16, 225.88, 515.65]) / 1e3
dij = np.array([
[0.    , 0.103   , 0.13    ,0.135   ,0.125   ,0.15    ,0.15    ,0.103  ],
 [0.103,  0.     ,  0.00153, 0.01114, 0.02076, 0.03847, 0.0704 , 0.108  ],
 [0.13 ,  0.00153,  0.     , 0.00446, 0.01118, 0.02512, 0.05245, 0.0864 ],
 [0.135,  0.01114,  0.00446, 0.     , 0.00154, 0.00862, 0.02728, 0.05402],
 [0.125,  0.02076,  0.01118, 0.00154, 0.     , 0.0029 , 0.0161 , 0.03812],
 [0.15 ,  0.03847,  0.02512, 0.00862, 0.0029 , 0.     , 0.00542, 0.02049],
 [0.15 ,  0.0704 ,  0.05245, 0.02728, 0.0161 , 0.00542, 0.     , 0.00495],
 [0.103,  0.108  ,  0.0864 , 0.05402, 0.03812, 0.02049, 0.00495, 0.     ]
 ])

for i, f in enumerate(fj):
  xi = (1. - f) * zi + f * yi
  print(f)

# print(Psat_ss(T, zi, Pci, Tci, wi, dij, vsi, c=1, upper=True, N_nodes=20))
  print(Psat_newton(T, xi, Pci, Tci, wi, dij, vsi, c=1, upper=True))
import sys
print("Current Python folder:", sys.prefix)