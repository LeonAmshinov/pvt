import numpy as np
from eos_pr_rdy import lnphi_Z
from stability_rdy import stability_succesive_substitution, stability_newton, init_ki
from R_R_rdy import R_R_2p, R_R_mp

def equil_max2p(P, T, yi,
                Pci, Tci, wi, dij, vsi, mwi, c=1,
                negative_flash=False, maxiter_flash=500, eps_flash=1e-6, method_stab='ss',
                eps1_stab=1e-10, eps2_stab=1e-8, maxiter_stab=50,
                level=0, pure_ind=0, pure_eps=1e-3,
                eps_rr=1e-10, maxiter_rr=50):

    if negative_flash: #Негативный флэш, чтобы выйти за физичные значения мольных долей фаз, но остаться в NFW
        kvji0 = init_ki(P, T, yi, Pci, Tci, wi, level, pure_ind, pure_eps)
        is_stable = False

    else:
        if method_stab == 'ss':
            is_stable, kvji0 = stability_succesive_substitution(P, T, yi, Pci, Tci, wi, vsi, dij, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
        elif method_stab == 'newton':
            is_stable, kvji0 = stability_newton(P, T, yi, Pci, Tci, wi, vsi, dij, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
        else:
            raise ValueError('Unknown stability method')
    if is_stable:
        return np.atleast_2d(yi), np.array([1., 0.])
    else:
        # Run flash
        for j, kvi in enumerate(kvji0):
            f1 = R_R_2p(yi, kvi, eps_rr, maxiter_rr)
            y2i = yi / (f1 * (kvi - 1) + 1)
            y1i = kvi * y2i
            lnphi1i, Z1 = lnphi_Z(y1i, P, T, Pci, Tci, wi, dij, vsi, c)
            lnphi2i, Z2 = lnphi_Z(y2i, P, T, Pci, Tci, wi, dij, vsi, c)
            gi = lnphi1i - lnphi2i + np.log(kvi)
            k = 0
            gnorm = np.linalg.norm(gi)
            while gnorm > eps_flash and k < maxiter_flash:
                k += 1
                kvi *= np.exp(-gi)
                f1 = R_R_2p(yi, kvi, eps_rr, maxiter_rr)
                y2i = yi / (f1 * (kvi - 1) + 1)
                y1i = kvi * y2i
                lnphi1i, Z1 = lnphi_Z(y1i, P, T, Pci, Tci, wi, dij, vsi, c)
                lnphi2i, Z2 = lnphi_Z(y2i, P, T, Pci, Tci, wi, dij, vsi, c)
                gi = lnphi1i - lnphi2i + np.log(kvi)
                gnorm = np.linalg.norm(gi)
            if gnorm < eps_flash:
                rho1 = y1i.dot(mwi) / Z1
                rho2 = y2i.dot(mwi) / Z2
                if rho1 < rho2:
                    return np.vstack((y1i, y2i)), np.array([f1, 1 - f1])
                else:
                    return np.vstack((y2i, y1i)), np.array([1 - f1, f1])
        raise ValueError('Solution equil_max2p_ss not found')