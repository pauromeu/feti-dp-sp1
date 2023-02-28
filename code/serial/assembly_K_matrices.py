import numpy as np


def assembly_KPP_matrix(Ks, APq, qs, qs_left_bound, mesh):
    KPP = np.zeros([mesh.NP, mesh.NP])

    for APqs in APq:
        if np.shape(APqs)[1] == len(qs):
            Kqqs = Ks[qs][:, qs]  # Obtain local Kqqs from local Ks
        else:
            # Obtain local Kqqs from local Ks
            Kqqs = Ks[qs_left_bound][:, qs_left_bound]
        KPP += APqs @ Kqqs @ APqs.T

    return KPP


def assembly_KRR_matrix(Ks, ARr, rs, mesh):
    KRR = np.zeros([mesh.NR, mesh.NR])
    Kqqs = Ks[rs][:, rs]  # Obtain local KRR from local Ks

    for ARrs in ARr:
        KRR += ARrs @ Kqqs @ ARrs.T

    return KRR


def assembly_KRP_matrix(Ks, APq, ARr, qs, qs_left_bound, rs, mesh):
    KRP = np.zeros([mesh.NR, mesh.NP])

    for APqs, ARrs in zip(APq, ARr):
        if np.shape(APqs)[1] == len(qs):
            Kqrs = Ks[rs][:, qs]  # Obtain local Kqqs from local Ks
        else:
            # Obtain local Kqqs from local Ks
            Kqrs = Ks[rs][:, qs_left_bound]
        KRP += ARrs @ Kqrs @ APqs.T

    return KRP
