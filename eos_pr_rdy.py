import numpy as np
R = 8.3144598

def cardano(b, c, d):
    p = (3 * c - b ** 2) / 3
    q = (2 * b ** 3 - 9 * b * c + 27 * d) / 27
    D = q ** 2 + 4 * (p / 3) ** 3
    if D >= 0:
        u1 = np.cbrt((-q + np.sqrt(D)) / 2)
        u2 = np.cbrt((-q - np.sqrt(D)) / 2)
        return (u1 + u2 - b / 3, 0, 0), True
    else:
        t0 = 2 * np.sqrt(-p / 3) * np.cos(1 / 3 * np.arccos(3 * q / (2 * p) * np.sqrt(-3 / p)))
        x0 = t0 - b / 3
        r = b + x0
        s = -d / x0
        D = r ** 2 - 4 * s
        return (x0, (-r + np.sqrt(D)) / 2, (-r - np.sqrt(D)) / 2), False

def compressibility_factor(xi, P, T, Pci, Tci, wi, dij, vsi, c):
    if c:
        omegaA = 0.4572355289213823
        sqrtomegaA = 0.6761919320144113
        omegaB = 0.0777960739038885
        ki = np.where(wi <= 0.491, 0.37464 + 1.54226 * wi - 0.26992 * wi ** 2, 0.379642 + 1.48503 * wi - 0.164423 * wi ** 2 + 0.016666 * wi ** 3)
    else:
        omegaA = 0.42748023354034137
        sqrtomegaA = 0.6538197255668732
        omegaB = 0.08664034996495773
        ki = 0.48 + 1.574 * wi - 0.176 * wi ** 2
    bi = omegaB * R * Tci / Pci
    sqrtai = sqrtomegaA * R * Tci / np.sqrt(Pci) * (1. + ki * (1. - np.sqrt(T / Tci)))
    Dij = 1 - dij
    Si = Dij.dot(xi * sqrtai) * sqrtai
    am = xi.dot(Si)
    bm = xi.dot(bi)
    A = am * P / (R * R * T * T )
    B = bm * P / (R * T)
    roots, unique = cardano(-1. + c * B, (A - (c + 1.) * B - (2. * c + 1.) * B * B), -(A * B - c * (B * B + B ** 3)))
    delta1 = (c + 1) * .5 - np.sqrt((c + 1) ** 2 * .25 + c)
    delta2 = -c / delta1
    if unique:
        z_2p = roots[0]
    else:
        fdG = lambda Z2, Z1: (np.log((Z2 - B) / (Z1 - B))
                              + (Z1 - Z2)
                              + A / (B * (delta2 - delta1)) * np.log((Z1 + B * delta1) * (Z2 + B * delta2) / ((Z1 + B * delta2) * (Z2 + B * delta1))))
        Z0, Z1, Z2 = roots
        if Z2 > B:
            if fdG(Z0, Z2) > 0.:
                z_2p = Z0
            else:
                z_2p = Z2
        elif Z1 > B:
            if fdG(Z0, Z1) > 0.:
                z_2p = Z0
            else:
                z_2p = Z1
        else:
            z_2p = Z0
    z_3p = z_2p - P / (R * T) * xi.dot(bi * vsi)
    return z3p

def lnphi_Z(xi, P, T, Pci, Tci, wi, dij, vsi, c):
    if c:
        omegaA = 0.4572355289213823
        sqrtomegaA = 0.6761919320144113
        omegaB = 0.0777960739038885
        ki = np.where(wi <= 0.491, 0.37464 + 1.54226 * wi - 0.26992 * wi ** 2, 0.379642 + 1.48503 * wi - 0.164423 * wi ** 2 + 0.016666 * wi ** 3)
    else:
        omegaA = 0.42748023354034137
        sqrtomegaA = 0.6538197255668732
        omegaB = 0.08664034996495773
        ki = 0.48 + 1.574 * wi - 0.176 * wi ** 2
    bi = omegaB * R * Tci / Pci
    sqrtai = sqrtomegaA * R * Tci / np.sqrt(Pci) * (1. + ki * (1. - np.sqrt(T / Tci)))
    Dij = 1 - dij
    Si = Dij.dot(xi * sqrtai) * sqrtai
    am = xi.dot(Si)
    bm = xi.dot(bi)
    A = am * P / (R * R * T * T )
    B = bm * P / (R * T)
    roots, unique = cardano(-1. + c * B, (A - (c + 1.) * B - (2. * c + 1.) * B * B), -(A * B - c * (B * B + B ** 3)))
    delta1 = (c + 1) * .5 - np.sqrt((c + 1) ** 2 * .25 + c)
    delta2 = -c / delta1
    if unique:
        z_2p = roots[0]
    else:
        fdG = lambda Z2, Z1: (np.log((Z2 - B) / (Z1 - B))
                              + (Z1 - Z2)
                              + A / (B * (delta2 - delta1)) * np.log((Z1 + B * delta1) * (Z2 + B * delta2) / ((Z1 + B * delta2) * (Z2 + B * delta1))))
        Z0, Z1, Z2 = roots
        if Z2 > B:
            if fdG(Z0, Z2) > 0.:
                z_2p = Z0
            else:
                z_2p = Z2
        elif Z1 > B:
            if fdG(Z0, Z1) > 0.:
                z_2p = Z0
            else:
                z_2p = Z1
        else:
            z_2p = Z0
    gphii = A / B * (2 / am * Si - bi / bm)
    fz = np.log((z_2p + B * delta1) / (z_2p + B * delta2))
    lnphi_2p = - np.log(z_2p - B) + (z_2p - 1) / bm * bi + fz / (delta2 - delta1) * gphii
    z_3p = z_2p - P / (R * T) * xi.dot(bi * vsi)
    lnphi_3p = lnphi_2p - P / (R * T) * (bi * vsi)
    return lnphi_3p, z_3p

def lnphii_dnj(xi, n, P, T, Pci, Tci, wi, dij, vsi, c):
    if c:
        omegaA = 0.4572355289213823
        sqrtomegaA = 0.6761919320144113
        omegaB = 0.0777960739038885
        ki = np.where(wi <= 0.491, 0.37464 + 1.54226 * wi - 0.26992 * wi ** 2, 0.379642 + 1.48503 * wi - 0.164423 * wi ** 2 + 0.016666 * wi ** 3)
    else:
        omegaA = 0.42748023354034137
        sqrtomegaA = 0.6538197255668732
        omegaB = 0.08664034996495773
        ki = 0.48 + 1.574 * wi - 0.176 * wi ** 2
    bi = omegaB * R * Tci / Pci
    sqrtai = sqrtomegaA * R * Tci / np.sqrt(Pci) * (1. + ki * (1. - np.sqrt(T / Tci)))
    Dij = 1 - dij
    Si = Dij.dot(xi * sqrtai) * sqrtai
    am = xi.dot(Si)
    bm = xi.dot(bi)
    A = am * P / (R * R * T * T)
    B = bm * P / (R * T)
    roots, unique = cardano(-1. + c * B, (A - (c + 1.) * B - (2. * c + 1.) * B * B), -(A * B - c * (B * B + B ** 3)))
    delta1 = (c + 1) * .5 - np.sqrt((c + 1) ** 2 * .25 + c)
    delta2 = -c / delta1
    if unique:
        z_2p = roots[0]
    else:
        fdG = lambda Z2, Z1: (np.log((Z2 - B) / (Z1 - B))
                              + (Z1 - Z2)
                              + A / (B * (delta2 - delta1)) * np.log((Z1 + B * delta1) * (Z2 + B * delta2) / ((Z1 + B * delta2) * (Z2 + B * delta1))))
        Z0, Z1, Z2 = roots
        if Z2 > B:
            if fdG(Z0, Z2) > 0.:
                z_2p = Z0
            else:
                z_2p = Z2
        elif Z1 > B:
            if fdG(Z0, Z1) > 0.:
                z_2p = Z0
            else:
                z_2p = Z1
        else:
            z_2p = Z0
    gphii = A / B * (2 / am * Si - bi / bm)
    fz = np.log((z_2p + B * delta1) / (z_2p + B * delta2))
    lnphi_2p = - np.log(z_2p - B) + (z_2p - 1) / bm * bi + fz / (delta2 - delta1) * gphii
    z_3p = z_2p - P / (R * T) * xi.dot(bi * vsi)
    lnphi_3p = lnphi_2p - P / (R * T) * (bi * vsi)
    damdnk = 2 / n * (Si - am)
    dbmdnk = (bi - bm) / n
    dqdz = 3 * z_2p ** 2 + 2 * (-1 + B * c) * z_2p + (A - (c + 1) * B - (2 * c + 1) * B ** 2)
    dAdnk = P / (R ** 2 * T ** 2) * damdnk
    dBdnk = P / (R * T) * dbmdnk
    dd2dnk = c * dBdnk
    dd1dnk = dAdnk - ((c + 1) + 2 * (2 * c + 1) * B) * dBdnk
    dd0dnk = - B * dAdnk + (-A + c * (2 * B + 3 * B ** 2)) * dBdnk
    dqdnk = dd2dnk * z_2p ** 2 + dd1dnk * z_2p + dd0dnk
    dzdnk = - dqdnk / dqdz
    dgzdnk = (dzdnk - dBdnk) / (z_2p - B)
    dgphiidnk = (2 / n * (sqrtai[:, None] * sqrtai * Dij - Si[:, None]) - 2 / bm * dbmdnk * (Si - am / bm * bi)[:, None] - bi[:, None] / bm * damdnk) / (R * T * bm)
    dfzdnk = (1 / (z_2p + B * delta1) - 1 / (z_2p + B * delta2)) * dzdnk + (delta1 / (z_2p + B * delta1) - delta2 / (z_2p + B * delta2)) * dBdnk
    dlnphiidnk = - dgzdnk + bi[:, None] * (dzdnk / bm - (z_2p - 1) / bm ** 2 * dbmdnk) + (dgphiidnk * fz + dfzdnk * gphii[:, None]) / (delta2 - delta1)
    return lnphi_3p, dlnphiidnk, z_3p


def lnphii_dP(xi, P, T, Pci, Tci, wi, dij, vsi, c):
    if c:
        omegaA = 0.4572355289213823
        sqrtomegaA = 0.6761919320144113
        omegaB = 0.0777960739038885
        ki = np.where(wi <= 0.491, 0.37464 + 1.54226 * wi - 0.26992 * wi ** 2, 0.379642 + 1.48503 * wi - 0.164423 * wi ** 2 + 0.016666 * wi ** 3)
    else:
        omegaA = 0.42748023354034137
        sqrtomegaA = 0.6538197255668732
        omegaB = 0.08664034996495773
        ki = 0.48 + 1.574 * wi - 0.176 * wi ** 2
    bi = omegaB * R * Tci / Pci
    sqrtai = sqrtomegaA * R * Tci / np.sqrt(Pci) * (1. + ki * (1. - np.sqrt(T / Tci)))
    Dij = 1 - dij
    Si = Dij.dot(xi * sqrtai) * sqrtai
    am = xi.dot(Si)
    bm = xi.dot(bi)
    A = am * P / (R * R * T * T)
    B = bm * P / (R * T)
    roots, unique = cardano(-1. + c * B, (A - (c + 1.) * B - (2. * c + 1.) * B * B), -(A * B - c * (B * B + B ** 3)))
    delta1 = (c + 1) * .5 - np.sqrt((c + 1) ** 2 * .25 + c)
    delta2 = -c / delta1
    if unique:
        z_2p = roots[0]
    else:
        fdG = lambda Z2, Z1: (np.log((Z2 - B) / (Z1 - B))
                              + (Z1 - Z2)
                              + A / (B * (delta2 - delta1)) * np.log((Z1 + B * delta1) * (Z2 + B * delta2) / ((Z1 + B * delta2) * (Z2 + B * delta1))))
        Z0, Z1, Z2 = roots
        if Z2 > B:
            if fdG(Z0, Z2) > 0.:
                z_2p = Z0
            else:
                z_2p = Z2
        elif Z1 > B:
            if fdG(Z0, Z1) > 0.:
                z_2p = Z0
            else:
                z_2p = Z1
        else:
            z_2p = Z0
    gphii = A / B * (2 / am * Si - bi / bm)
    fz = np.log((z_2p + B * delta1) / (z_2p + B * delta2))
    lnphi_2p = - np.log(z_2p - B) + (z_2p - 1) / bm * bi + fz / (delta2 - delta1) * gphii
    z_3p = z_2p - P / (R * T) * xi.dot(bi * vsi)
    lnphi_3p = lnphi_2p - P / (R * T) * (bi * vsi)
    dqdz = 3 * z_2p ** 2 + 2 * (-1 + B * c) * z_2p + (A - (c + 1) * B - (2 * c + 1) * B ** 2)
    dAdP = A / P
    dBdP = B / P
    dd2dP = c * dBdP
    dd1dP = dAdP - ((c + 1) + 2 * (2 * c + 1) * B) * dBdP
    dd0dP = - B * dAdP + (-A + c * (2 * B + 3 * B ** 2)) * dBdP
    dqdP = dd2dP * z_2p ** 2 + dd1dP * z_2p + dd0dP
    dzdP = - dqdP / dqdz
    dgzdP = (dzdP - dBdP) / (z_2p - B)
    dfzdP = (1 / (z_2p + B * delta1) - 1 / (z_2p + B * delta2)) * dzdP + (delta1 / (z_2p + B * delta1) - delta2 / (z_2p + B * delta2)) * dBdP
    dlnphiidP = - dgzdP + bi * (dzdP / bm) + gphii * (dfzdP / (delta2 - delta1))
    return lnphi_3p, dlnphiidP, z_3p
