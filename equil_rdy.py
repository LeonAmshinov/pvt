import numpy as np
from eos_pr_rdy import lnphi_Z, lnphii_dnj
from stability_rdy import stability_succesive_substitution, stability_newton, init_ki
from R_R_rdy import R_R_2p, R_R_mp

def equil_max2p_ss(P, T, yi,
                Pci, Tci, wi, dij, vsi, mwi, c=1,
                negative_flash=False, maxiter_flash=500, eps_flash=1e-8, method_stab='ss',
                eps1_stab=1e-10, eps2_stab=1e-8, maxiter_stab=50,
                level=0, pure_ind=0, pure_eps=1e-3,
                eps_rr=1e-10, maxiter_rr=50):

    if negative_flash: #Негативный флэш, чтобы выйти за физичные значения мольных долей фаз, но остаться в NFW
        kvji0 = init_ki(P, T, yi, Pci, Tci, wi, level, pure_ind, pure_eps)
        is_stable = False

    else:
        if method_stab == 'ss':
            is_stable, kvji0, Z = stability_succesive_substitution(P, T, yi, Pci, Tci, wi, dij, vsi, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
        elif method_stab == 'newton':
            is_stable, kvji0, Z = stability_newton(P, T, yi, Pci, Tci, wi, dij, vsi, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
        else:
            raise ValueError('Unknown stability method')
    if is_stable:
        return np.atleast_2d(yi), np.array([1.]), np.array([Z])
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
                # print(f'For the initial guess #{j}:\n'
                #       f'\ttolerance of equations: {gnorm}\n'
                #       f'\tnumber of iterations: {k}\n'
                #       f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
                #       f'\tphase mole fractions: {f1}, {1 - f1}')
                rho1 = y1i.dot(mwi) / Z1
                rho2 = y2i.dot(mwi) / Z2
                if rho1 < rho2:
                    return np.vstack([y1i, y2i]), np.array([f1, 1 - f1]), np.array([Z1, Z2])
                else:
                    return np.vstack([y2i, y1i]), np.array([1 - f1, f1]), np.array([Z2, Z1])
        raise ValueError('Solution equil_max2p_ss not found')

def equil_ss(P, T, yi,
          Pci, Tci, wi, dij, vsi, mwi, c=1,
          nphase=3, negative_flash=False, maxiter_flash=500, eps_flash=1e-8, method_stab='ss',
          eps1_stab=1e-10, eps2_stab=1e-8, maxiter_stab=50,
          level=0, pure_ind=0, pure_eps=1e-3,
          eps_rr=1e-10, maxiter_rr=50, eps_linesearch=1e-5, maxiter_linesearch=10):
    # if method_stab == 'ss':
    #     fstab = partial(stability_succesive_substitution, **stabkwargs)
    # else:
    #     fstab = partial(stability_newton, **stabkwargs)
    # else:
    #     raise ValueError('Unknown stability method')

    #Проверка стабильности однофазной системы, если нестабильна, то флеш двухфазной
    yji, fj, Zj = equil_max2p_ss(P, T, yi, Pci, Tci, wi, dij, vsi, mwi, c, negative_flash, maxiter_flash, eps_flash, method_stab, eps1_stab, eps2_stab, maxiter_stab, 
                              level, pure_ind, pure_eps, eps_rr, maxiter_rr)
    #Проверка стабильности двухфазной системы на основе компонентного состава второй фазы
    if method_stab == 'ss':
        is_stable, kvji0, _ = stability_succesive_substitution(P, T, yji[1], Pci, Tci, wi, dij, vsi, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
    elif method_stab == 'newton':
        is_stable, kvji0, _ = stability_newton(P, T, yji[1], Pci, Tci, wi, dij, vsi, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
    else:
        raise ValueError('Unknown stability method')
    # Если двухфазная система нестабильна, то делаем флеш трехфазной системы
    if not is_stable:
        fj = fj[0] #Подготовили для начального приближения
        kvji = yji[0] / yji[1] #Подготовили для начального приближения
        for p in range(3, nphase + 1): #Начальные приближения для Np-фазного флеша
            fj = np.hstack([fj, 0.]) #Начальное приближение для РР (добавили фазу), где 0 - мольная доля мнимой фазы
            kvji = np.vstack([kvji, kvji0[0]]) #Константы фазового равновесия для РР и начальное приближение для флеша (добавили фазу)
            k = 0
            fj = R_R_mp(yi, kvji, fj, eps_rr, maxiter_rr, eps_linesearch, maxiter_linesearch) #Решаем РР, находим мольные доли Np-1 фаз
            xi = yi / (fj.dot(kvji - 1) + 1) #Находим компонентный состав референсной фазы (Np фазы)
            yji = kvji * xi #Находим компонентный состав нереференсных Np-1 фаз 
            lnphiyji = []
            Zyj = []
            for j in range(yji.shape[0]): #Находим логарифмы коэффициентов летучестей компонентов в нереференсных Np-1 фазах
                lnphiyi, Zy = lnphi_Z(yji[j], P, T, Pci, Tci, wi, dij, vsi, c)
                lnphiyji.append(lnphiyi)
                Zyj.append(Zy)
            lnphiyji = np.array(lnphiyji)
            Zyj = np.array(Zyj)
            lnphixi, Zx = lnphi_Z(xi, P, T, Pci, Tci, wi, dij, vsi, c) #Находим логарифмы коэффициентов летучестей компонентов референсной фазы
            gji = np.log(kvji) + lnphiyji - lnphixi #Рассчитываем систему уравнений равенства летучестей компонентов (невязка)
            gnorm = np.linalg.norm(gji) #Рассчитываем длину матрицы невязок
            while gnorm > eps_flash and k < maxiter_flash: #Обновление КФР для решения системы уравнений равенства летучестей компонентов
                kvji *= np.exp(-gji) #Обновление КФР методом последовательных подстановок
                k += 1
                fj = R_R_mp(yi, kvji, fj, eps_rr, maxiter_rr, eps_linesearch, maxiter_linesearch) #Решаем РР, находим мольные доли Np-1 фаз
                xi = yi / (fj.dot(kvji - 1) + 1) #Находим компонентный состав референсной фазы (Np фазы)
                yji = kvji * xi #Находим компонентный состав нереференсных Np-1 фаз 
                lnphiyji = []
                Zyj = []
                for j in range(yji.shape[0]): #Находим логарифмы коэффициентов летучестей компонентов в нереференсных Np-1 фазах
                    lnphiyi, Zy = lnphi_Z(yji[j], P, T, Pci, Tci, wi, dij, vsi, c)
                    lnphiyji.append(lnphiyi)
                    Zyj.append(Zy)
                lnphiyji = np.array(lnphiyji)
                Zyj = np.array(Zyj)
                lnphixi, Zx = lnphi_Z(xi, P, T, Pci, Tci, wi, dij, vsi, c) #Находим логарифмы коэффициентов летучестей компонентов референсной фазы
                gji = np.log(kvji) + lnphiyji - lnphixi #Рассчитываем систему уравнений равенства летучестей компонентов (невязка)
                gnorm = np.linalg.norm(gji) #Рассчитываем длину матрицы невязок
            if gnorm < eps_flash:
                print(f'For Np = {p}:\n'
                      f'\ttolerance of equations: {gnorm}\n'
                      f'\tnumber of iterations: {k}\n'
                      f'\tphase compositions:\n{np.vstack([yji, xi])}\n'
                      f'\tphase mole fractions: {np.hstack([fj, 1 - fj.sum()])}')
                #Если нашел решение, то проверка его стабильности на основе компонентного состава референсной фазы
                if method_stab == 'ss':
                    is_stable, kvji0, _ = stability_succesive_substitution(P, T, xi, Pci, Tci, wi, dij, vsi, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
                elif method_stab == 'newton':
                    is_stable, kvji0, _ = stability_newton(P, T, xi, Pci, Tci, wi, dij, vsi, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
                else:
                    raise ValueError('Unknown stability method')
                if is_stable:
                    yji = np.vstack([yji, xi])
                    fj = np.hstack([fj, 1 - fj.sum()])
                    Zj = np.hstack([Zyj, Zx])
                    rhoj = yji.dot(mwi) / Zj
                    idx = np.argsort(rhoj)
                    return yji[idx], fj[idx], Zj[idx]
            else:
                raise ValueError('Solution not found')
        print('WARNING!!! Increase max number of phases')
        yji = np.vstack([yji, xi])
        fj = np.hstack([fj, 1 - fj.sum()])
        Zj = np.hstack([Zyj, Zx])
        rhoj = yji.dot(mwi) / Zj
        idx = np.argsort(rhoj)
        return yji[idx], fj[idx], Zj[idx]
    else:
        return yji, fj, Zj

def equil_max2p_newton(P, T, yi, Pci, Tci, wi, dij, vsi, mwi, c=1,
                       negative_flash=False, maxiter_flash=500, eps_flash=1e-8, method_stab='ss',
                       eps1_stab=1e-10, eps2_stab=1e-8, maxiter_stab=50,
                       level=0, pure_ind=0, pure_eps=1e-3,
                       eps_rr=1e-10, maxiter_rr=50):

    if negative_flash: #Негативный флэш, чтобы выйти за физичные значения мольных долей фаз, но остаться в NFW
        kvji0 = init_ki(P, T, yi, Pci, Tci, wi, level, pure_ind, pure_eps)
        is_stable = False

    else:
        if method_stab == 'ss':
            is_stable, kvji0, Z = stability_succesive_substitution(P, T, yi, Pci, Tci, wi, dij, vsi, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
        elif method_stab == 'newton':
            is_stable, kvji0, Z = stability_newton(P, T, yi, Pci, Tci, wi, dij, vsi, c, eps1_stab, eps2_stab, maxiter_stab, level, pure_ind, pure_eps)
        else:
            raise ValueError('Unknown stability method')
    if is_stable:
        return np.atleast_2d(yi), np.array([1.]), np.array([Z])
    else:
        # Run flash
        for j, kvi in enumerate(kvji0):
            lnkvi = np.log(kvi)
            f1 = R_R_2p(yi, kvi, eps_rr, maxiter_rr)
            f2 = 1 - f1
            y2i = yi / (f1 * (kvi - 1) + 1)
            y1i = kvi * y2i
            n1i = y1i * f1
            n2i = y2i * f2
            lnphi1i, dlnphi1i, Z1 = lnphii_dnj(y1i, f1, P, T, Pci, Tci, wi, dij, vsi, c)
            lnphi2i, dlnphi2i, Z2 = lnphii_dnj(y2i, f2, P, T, Pci, Tci, wi, dij, vsi, c)
            gi = lnphi1i - lnphi2i + lnkvi
            k = 0
            gnorm = np.linalg.norm(gi)
            while gnorm > eps_flash and k < maxiter_flash:
                k += 1
                Fil = dlnphi1i + dlnphi2i
                Uil = np.diagflat(yi / (n1i * n2i)) - 1 / (f1 * f2)
                Hil = Fil + Uil
                dn1i = - np.linalg.solve(Hil, gi)
                dlnkvi = Uil.dot(dn1i)
                lnkvi += dlnkvi
                kvi = np.exp(lnkvi)
                f1 = R_R_2p(yi, kvi, eps_rr, maxiter_rr)
                f2 = 1 - f1
                y2i = yi / (f1 * (kvi - 1) + 1)
                y1i = kvi * y2i
                n1i = y1i * f1
                n2i = y2i * f2
                lnphi1i, dlnphi1i, Z1 = lnphii_dnj(y1i, f1, P, T, Pci, Tci, wi, dij, vsi, c)
                lnphi2i, dlnphi2i, Z2 = lnphii_dnj(y2i, f2, P, T, Pci, Tci, wi, dij, vsi, c)
                gi = lnphi1i - lnphi2i + lnkvi
                gnorm = np.linalg.norm(gi)
            if gnorm < eps_flash:
                rho1 = y1i.dot(mwi) / Z1
                rho2 = y2i.dot(mwi) / Z2
                if rho1 < rho2:
                    print(f'For the initial guess #{j}:\n'
                    f'\ttolerance of equations: {gnorm}\n'
                    f'\tnumber of iterations: {k}\n'
                    f'\tphase compositions:\n\t\t{y1i}\n\t\t{y2i}\n'
                    f'\tphase mole fractions: {f1}, {1 - f1}')
                    return np.vstack([y1i, y2i]), np.array([f1, 1 - f1]), np.array([Z1, Z2])
                else:
                    print(f'For the initial guess #{j}:\n'
                    f'\ttolerance of equations: {gnorm}\n'
                    f'\tnumber of iterations: {k}\n'
                    f'\tphase compositions:\n\t\t{y2i}\n\t\t{y1i}\n'
                    f'\tphase mole fractions: {1 - f1}, {f1}')
                    return np.vstack([y2i, y1i]), np.array([1 - f1, f1]), np.array([Z2, Z1])
        raise ValueError('Solution equil_max2p_ss not found')