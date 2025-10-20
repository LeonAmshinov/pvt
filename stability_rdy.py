import numpy as np
from eos_pr_rdy import lnphi_Z, lnphii_dnj, R

def init_ki(P, T, zi, Pci, Tci, wi, level=0, pure_ind=0, pure_eps=1e-8):
    ki = Pci / P * np.exp(5.3727 * (1 + wi) * (1 - Tci / T))
    if level == 0:
        return (ki, 1/ki)
    elif level == 1:
        ki_pure = pure_eps / ((zi.shape[0] - 1) * zi)
        ki_pure[pure_ind] = (1 - pure_eps) / zi[pure_ind]
        return ki, 1/ki, ki_pure, 1/ki_pure
    elif level == 2:
        ki_pure = pure_eps / ((zi.shape[0] - 1) * zi)
        ki_pure[pure_ind] = (1 - pure_eps) / zi[pure_ind]
        cbrtki = np.cbrt(ki)
        return ki, 1/ki, ki_pure, 1/ki_pure, cbrtki, 1/cbrtki

def stability_newton(P, T, zi, Pci, Tci, wi, dij, vsi, c=1, eps1=1e-10, eps2=1e-8, maxiter=50, level=0, pure_ind=0, pure_eps=1e-3):
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
        lnphii, dlnphiidYj, _ = lnphii_dnj(yi, np.sum(Yi), P, T, Pci, Tci, wi, dij, vsi, c)
        gpi = np.log(Yi) + lnphii - hi
        gi = sqrtYi * gpi
        gnorm = np.linalg.norm(gpi)
        k = 0
        while gnorm > eps1 and k < maxiter:
            Hij = np.diagflat(gpi * .5 + 1) + (sqrtYi * sqrtYi[:, None]) * dlnphiidYj
            dalphai = - np.linalg.solve(Hij, gi)
            alphai += dalphai
            sqrtYi = alphai * .5
            Yi = sqrtYi * sqrtYi
            yi = Yi / np.sum(Yi)
            lnphii, dlnphiidYj, _ = lnphii_dnj(yi, np.sum(Yi), P, T, Pci, Tci, wi, dij, vsi, c)
            gpi = np.log(Yi) + lnphii - hi
            gi = sqrtYi * gpi
            gnorm = np.linalg.norm(gpi)
            k += 1
        TPD = - np.log(np.sum(Yi))
        # print(np.linalg.eigvals(Hij))
        # print(f'For the initial guess #{j}:\n'
        #       f'\ttolerance of equations: {gnorm}\n'
        #       f'\tnumber of iterations: {k+1}\n'
        #       f'\tTPD: {TPD}\n'
        #       f'\tlnki = {np.log(Yi/zi)}')
        if gnorm < eps1 and TPD < -eps2:
            is_stable = False
            break
    else:
        is_stable = True
    # print(f'The one phase state is stable: {is_stable}')
    kvi = yi / zi
    return is_stable, (kvi,), Z

def stability_succesive_substitution(P, T, zi, Pci, Tci, wi, dij, vsi, c=1, eps1=1e-10, eps2=1e-8, maxiter=500, level=0, pure_ind=0, pure_eps=1e-3):
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
        # print(f'For the initial guess #{j}:\n'
        #       f'\ttolerance of equations: {gnorm}\n'
        #       f'\tnumber of iterations: {k+1}\n'
        #       f'\tTPD: {TPD}\n'
        #       f'\t{ki = }')
        if gnorm < eps1 and TPD < -eps2:
            is_stable = False
            break
    else:
        is_stable = True
    # print(f'The one phase state is stable: {is_stable}')
    kvi = yi / zi
    return is_stable, (kvi,), Z

def stability_newton_after_ss(P, T, zi, ki, Z, hi, Pci, Tci, wi, dij, vsi, k, c=1, eps1=1e-20, eps2=1e-8, maxiter=50):
    # lnphi, Z = lnphi_Z(zi, P, T, Pci, Tci, wi, dij, vsi, c)
    # hi = lnphi + np.log(zi)
    Yi = ki * zi
    sqrtYi = np.sqrt(Yi)
    alphai = 2 * sqrtYi
    yi = Yi / np.sum(Yi)
    lnphii, dlnphiidYj, _ = lnphii_dnj(yi, np.sum(Yi), P, T, Pci, Tci, wi, dij, vsi, c)
    gpi = np.log(Yi) + lnphii - hi
    gi = sqrtYi * gpi
    g2 = gpi.dot(gpi)
    k += 1
    # print(f'\tmethod = Newton, {k = }, {g2 = :.2e}, lnki = {np.log(Yi/zi)}')
    while g2 > eps1 and k < maxiter:
        Hij = np.diagflat(gpi * .5 + 1) + (sqrtYi * sqrtYi[:, None]) * dlnphiidYj
        dalphai = - np.linalg.solve(Hij, gi)
        alphai += dalphai
        sqrtYi = alphai * .5
        Yi = sqrtYi * sqrtYi
        yi = Yi / np.sum(Yi)
        lnphii, dlnphiidYj, _ = lnphii_dnj(yi, np.sum(Yi), P, T, Pci, Tci, wi, dij, vsi, c)
        gpi = np.log(Yi) + lnphii - hi
        gi = sqrtYi * gpi
        g2 = gpi.dot(gpi)
        k += 1
        # print(f'\tmethod = Newton, {k = }, {g2 = :.2e}, lnki = {np.log(Yi / zi)}')
    kvi = yi / zi
    return g2, Yi, (kvi,), Z


def stability_ss_newton(P, T, zi, Pci, Tci, wi, dij, vsi, c=1,
                        eps1=1e-20, eps2=1e-8, maxiter=50,
                        level=0, pure_ind=0, pure_eps=1e-3,
                        eps_r=0.1, eps_u=1e-4, eps_l=1e-12):
    lnphi, Z = lnphi_Z(zi, P, T, Pci, Tci, wi, dij, vsi, c)
    Kji = init_ki(P, T, zi, Pci, Tci, wi, level, pure_ind, pure_eps)
    hi = lnphi + np.log(zi)
    TPD_min = -eps2
    is_stable = True
    ki_min = None
    # for j in range(len(Kji)):
    #     ki = Kji[j]
    for j, ki in enumerate(Kji):
        # print(f'Initial guess #{j}:')
        Yi = ki * zi
        yi = Yi / np.sum(Yi)
        gi = np.log(Yi) + lnphi_Z(yi, P, T, Pci, Tci, wi, dij, vsi, c)[0] - hi
        g2 = gi.dot(gi) # g2 = gnorm * gnorm
        k = 0
        # print(f'\tmethod =     SS, {k = }, {g2 = :.2e}, lnki = {np.log(ki)}')
        while g2 > eps1 and k < maxiter:
            ki = ki * np.exp(-gi)
            Yi = ki * zi
            yi = Yi / np.sum(Yi)
            g2m1 = g2
            gi = np.log(Yi) + lnphi_Z(yi, P, T, Pci, Tci, wi, dij, vsi, c)[0] - hi
            g2 = gi.dot(gi)
            if g2 / g2m1 > eps_r and g2 < eps_u and g2 > eps_l:
                # print(f'\tmethod =     SS, {k = }, {g2 = :.2e}, lnki = {np.log(ki)}')
                g2, Yi, kji, Z = stability_newton_after_ss(P, T, zi, ki, Z, hi, Pci, Tci, wi, dij, vsi, k, c)
                ki = kji[0]
                break
            else:
                # print(f'\tmethod =     SS, {k = }, {g2 = :.2e}, lnki = {np.log(ki)}')
                k += 1
        TPD = - np.log(np.sum(Yi))
        if g2 < eps1 and TPD < TPD_min:
            is_stable = False
            ki_min = ki
            TPD_min = TPD

    # print(f'The one phase state is stable: {is_stable}')

    return is_stable, (ki_min,), Z