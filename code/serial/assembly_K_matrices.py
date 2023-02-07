import numpy as np


def assembly_KPP_matrix(Ks, APq, qs, mesh):
    KPP = np.zeros([mesh.NP, mesh.NP])
    Kqqs = Ks[qs][:, qs]  # Obtain local Kqqs from local Ks

    for APqs in APq:
        KPP += APqs @ Kqqs @ APqs.T

    return KPP


def assembly_KRR_matrix(Ks, ARr, rs, mesh):
    KRR = np.zeros([mesh.NR, mesh.NR])
    Kqqs = Ks[rs][:, rs]  # Obtain local KRR from local Ks

    for ARrs in ARr:
        KRR += ARrs @ Kqqs @ ARrs.T

    return KRR


def assembly_KRP_matrix(Ks, APq, ARr, qs, rs, mesh):
    KRP = np.zeros([mesh.NR, mesh.NP])
    Kqrs = Ks[rs][:, qs]

    for APqs, ARrs in zip(APq, ARr):
        KRP += ARrs @ Kqrs @ APqs.T

    return KRP
