import numpy as np

def G_value_deriv(a, yi, di):
    znam_G = 1 / ((1 + di) * a + di)
    return (a + 1) * yi.dot(znam_G), -yi.dot(znam_G ** 2)

def H_value_deriv(a, yi, di):
    G_value, G_deriv = G_value_deriv(a, yi, di)
    return -a * G_value, -(G_value + a * G_deriv)

def D_value_deriv(a, yi, di):
    G_value, G_deriv = G_value_deriv(a, yi, di)
    return a / (a + 1) * G_value, G_deriv * a / (a + 1) + G_value / (a + 1) ** 2

def R_R_2p(yi, kvi, eps=1e-10, maxiter=50):
    head = '%3s %11s %11s %8s'
    tmpl = '%3i %11.2e %11.2e %8s'
    ind = np.argsort(kvi)[::-1]
    kvi = kvi[ind]
    yi = yi[ind]
    ci = 1 / (1 - kvi)
    di = (ci[0] - ci) / (ci[-1] - ci[0])
    k = 0
    ak = yi[0] / yi[-1]
    # print(head % ('Nit', 'a', 'D(a)', 'method'))
    Dk, dDdak = D_value_deriv(ak, yi, di)
    # print(tmpl % (k, ak, Dk, 'D'))
    while np.abs(Dk) > eps and k < maxiter:
        h = Dk / dDdak
        akm1 = ak
        ak -= h
        method = 'D'
        if ak < 0:
            if Dk > 0:
                ak += h ** 2 / (h - akm1 * (akm1 + 1))
                method = 'G'
            else:
                ak += h ** 2 / (h + 1 + akm1)
                method = 'H'
        Dk, dDdak = D_value_deriv(ak, yi, di)
        k += 1
        # print(tmpl % (k, ak, Dk, method))
    f = (ak * ci[-1] + ci[0]) / (1 + ak) #мольная доля нереференсной фазы
    # print('f = %.3f' % f)
    return f

def R_R_mp(yi, Kji, fjk, eps=1e-6, maxiter=30, eps_linesearch=1e-5, maxiter_linesearch=10):
    Nphasem1 = Kji.shape[0]
    headphases = []
    for i in range(Nphasem1):
        headphases.append('f' + str(i))
        # headphases.append('f%i' % i)
    head = ('%3s %5s' + Nphasem1 * '%9s' + '%10s%9s%9s') % ('Nit', 'Nls', *headphases, 'gnorm', 'lmbd', 'dFdlmbd')
    # print(head)
    tmpl = '%3i %5i' + Nphasem1 * '%9.5f' + '%10.2e%9.5f%9.5f'

    sqrtyi = np.sqrt(yi)
    Aji = 1 - Kji
    bi = np.min([1 - yi, np.min(1 - Kji * yi, axis=0)], axis=0)
    ti = 1 - fjk.dot(Aji)
    dFdfj = Aji.dot(yi / ti)
    gnorm = np.linalg.norm(dFdfj)
    k = 0
    n = 0
    lmbdn = 1
    dFdlmbd = 1
    # print(tmpl % (k, n, *fjk, gnorm, lmbdn, dFdlmbd))
    while gnorm > eps and k < maxiter:
        Pji = Aji * (sqrtyi / ti)
        Hjl = Pji.dot(Pji.T)
        dfj = np.linalg.solve(Hjl, -dFdfj)
        tempi = dfj.dot(Aji)
        lmbdi = (bi - fjk.dot(Aji)) / tempi
        where = tempi > 0
        lmbdmax = np.min(lmbdi[where])
        if lmbdmax < 1:
            lmbdn = lmbdmax * .5
            fjkp1 = fjk + lmbdn * dfj
            ti = 1 - fjkp1.dot(Aji)
            dFdfj = Aji.dot(yi / ti)
            gnorm = np.linalg.norm(dFdfj)
            dFdlmbd = dFdfj.dot(dfj)
            poluslmbdi = (1 - fjk.dot(Aji)) / tempi
            poluslmbd = np.min(poluslmbdi[where])
            # print(tmpl % (k, n, *fjkp1, gnorm, lmbdn, dFdlmbd))
            while np.abs(dFdlmbd) > eps_linesearch and n < maxiter_linesearch:
                Pji = Aji * (sqrtyi / ti)
                Hjl = Pji.dot(Pji.T)
                d2Fdlmbd2 = dfj.dot(Hjl).dot(dfj)
                lmbdnp1 = lmbdn - dFdlmbd / d2Fdlmbd2
                if lmbdnp1 > poluslmbd:
                    lmbdn = (poluslmbd + lmbdn) * .5
                    # print('bisection: lmbdn > poluslmbd')
                elif lmbdnp1 < 0:
                    lmbdn *= .5
                    # print('bisection: lmbdn < 0')
                else:
                    lmbdn = lmbdnp1
                n += 1
                fjkp1 = fjk + lmbdn * dfj
                ti = 1 - fjkp1.dot(Aji)
                dFdfj = Aji.dot(yi / ti)
                gnorm = np.linalg.norm(dFdfj)
                dFdlmbd = dFdfj.dot(dfj)
                # print(tmpl % (k, n, *fjkp1, gnorm, lmbdn, dFdlmbd))
            fjk = fjkp1
            n = 0
        else:
            # берем lmbd = 1
            fjk += dfj
            ti = 1 - fjk.dot(Aji)
            dFdfj = Aji.dot(yi / ti)
            gnorm = np.linalg.norm(dFdfj)
            lmbdn = 1
        k += 1
        # print(tmpl % (k, n, *fjk, gnorm, lmbdn, dFdlmbd))
    return fjk