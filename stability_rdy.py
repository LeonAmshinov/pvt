import numpy as np
from eos_pr_rdy import lnphi_Z, lnphii_dnj, R

def init_ki(P, T, zi, Pci, Tci, wi, level=0, pure_ind=0, pure_eps=1e-3):
    ki = Pci / P * np.exp(5.3727 * (1 + wi) * (1 - Tci / T))
    if level == 0:
        return (ki, 1/ki)
    elif level == 1:
        ki_pure = pure_eps / ((zi.shape[0] - 1) * zi)
        ki_pure[pure_ind] = (1 - pure_eps) / yi[pure_ind]
        return ki, 1/ki, ki_pure
    elif level == 2:
        ki_pure = pure_eps / ((zi.shape[0] - 1) * zi)
        ki_pure[pure_ind] = (1 - pure_eps) / yi[pure_ind]
        cbrtki = np.cbrt(ki)
        return ki, 1/ki, ki_pure, cbrtki, 1/cbrtki

def stability_newton(P, T, zi, Pci, Tci, wi, vsi, dij, c=1, eps1=1e-10, eps2=1e-8, maxiter=50, level=0, pure_ind=0, pure_eps=1e-3):
    lnphi, Z = lnphi_Z(zi, P, T, Pci, Tci, wi, dij, vsi, c)
    Kji = init_ki(P, T, zi, Pci, Tci, wi, level, pure_ind, pure_eps)
    hi = lnphi + np.log(zi)
    # for j in range(len(Kji)):
    #     ki = Kji[j]
    for j, ki in enumerate(Kji):
        Yi = ki * zi
        sqrtYi = np.sqrt(Yi)
        alphai = 2 * sqrtYi
        yi = Yi / np.sum(Yi)
        lnphii, dlnphiidYj = lnphii_dnj(yi, np.sum(Yi), P, T, Pci, Tci, wi, dij, vsi, c)
        gpi = np.log(Yi) + lnphii - hi
        gi = sqrtYi * gpi
        gnorm = np.linalg.norm(gi)
        k = 0
        while gnorm > eps1 and k < maxiter:
            Hij = np.diagflat(gpi * .5 + 1) + (sqrtYi * sqrtYi[:, None]) * dlnphiidYj
            dalphai = - np.linalg.solve(Hij, gi)
            alphai += dalphai
            sqrtYi = alphai * .5
            Yi = sqrtYi * sqrtYi
            yi = Yi / np.sum(Yi)
            lnphii, dlnphiidYj = lnphii_dnj(yi, np.sum(Yi), P, T, Pci, Tci, wi, dij, vsi, c)
            gpi = np.log(Yi) + lnphii - hi
            gi = sqrtYi * gpi
            gnorm = np.linalg.norm(gi)
            k += 1
        TPD = - np.log(np.sum(Yi))
        #print(np.linalg.eigvals(Hij))
        print(f'For the initial guess #{j}:\n'
              f'\ttolerance of equations: {gnorm}\n'
              f'\tnumber of iterations: {k+1}\n'
              f'\tTPD: {TPD}\n'
              f'\tki = {Yi/zi}')
        if gnorm < eps1 and TPD < -eps2:
            is_stable = False
            break
    else:
        is_stable = True
    print(f'The one phase state is stable: {is_stable}')
    kvi = yi / zi
    return is_stable, (kvi, 1 / kvi)

def stability_succesive_substitution(P, T, zi, Pci, Tci, wi, vsi, dij, c=1, eps1=1e-10, eps2=1e-8, maxiter=500, level=0, pure_ind=0, pure_eps=1e-3):
    lnphi, Z = lnphi_Z(zi, P, T, Pci, Tci, wi, dij, vsi, c)
    Kji = init_ki(P, T, zi, Pci, Tci, wi, level, pure_ind, pure_eps)
    hi = lnphi + np.log(zi)
    # for j in range(len(Kji)):
    #     ki = Kji[j]
    for j, ki in enumerate(Kji):
        Yi = ki * zi
        yi = Yi / np.sum(Yi)
        gi = np.log(Yi) + lnphi_Z(yi, P, T, Pci, Tci, wi, dij, vsi, c)[0] - hi
        gnorm = np.linalg.norm(gi)
        k = 0
        while gnorm > eps1 and c < maxiter:
            ki = ki * np.exp(-gi)
            Yi = ki * zi
            yi = Yi / np.sum(Yi)
            gi = np.log(Yi) + lnphi_Z(yi, P, T, Pci, Tci, wi, dij, vsi, c)[0] - hi
            gnorm = np.linalg.norm(gi)
            k += 1
        TPD = - np.log(np.sum(Yi))
        print(f'For the initial guess #{j}:\n'
              f'\ttolerance of equations: {gnorm}\n'
              f'\tnumber of iterations: {k+1}\n'
              f'\tTPD: {TPD}\n'
              f'\t{ki = }')
        if gnorm < eps1 and TPD < -eps2:
            is_stable = False
            break
    else:
        is_stable = True
    print(f'The one phase state is stable: {is_stable}')
    kvi = yi / zi
    return is_stable, (kvi, 1 / kvi)