import numpy as np
from scipy.sparse import csr_matrix


def assembly_KPP_matrix(Ks, APq, qs, qs_left_bound, mesh):
    KPP = np.zeros([mesh.NP, mesh.NP])
    # KPP = csr_matrix((mesh.NP, mesh.NP))

    for APqs in APq:
        if np.shape(APqs)[1] == len(qs):
            Kqqs = csr_matrix(Ks[qs][:, qs])  # Obtain local Kqqs from local Ks
        else:
            # Obtain local Kqqs from local Ks
            Kqqs = csr_matrix(Ks[qs_left_bound][:, qs_left_bound])
        KPP += APqs @ Kqqs @ APqs.T

    return KPP


def assembly_KRR_matrix(Ks, ARr, rs, mesh):
    KRR = np.zeros([mesh.NR, mesh.NR])
    # KRR = csr_matrix((mesh.NR, mesh.NR))
    Kqqs = csr_matrix(Ks[rs][:, rs])  # Obtain local KRR from local Ks

    for ARrs in ARr:
        KRR += ARrs @ Kqqs @ ARrs.T

    return KRR


def assembly_KRP_matrix(Ks, APq, ARr, qs, qs_left_bound, rs, mesh):
    KRP = np.zeros([mesh.NR, mesh.NP])
    # KRP = csr_matrix((mesh.NR, mesh.NP))
    Krqs_list = []

    for APqs, ARrs in zip(APq, ARr):
        if np.shape(APqs)[1] == len(qs):
            Kqrs = csr_matrix(Ks[rs][:, qs])  # Obtain local Kqqs from local Ks
        else:
            # Obtain local Kqqs from local Ks
            Kqrs = csr_matrix(Ks[rs][:, qs_left_bound])
        KRP += ARrs @ Kqrs @ APqs.T
        Krqs_list.append(Kqrs)

    return KRP, Krqs_list


def assembly_KPD_matrix(Ks, APq, ADq, qs_left_bound, qs_right_bound, mesh):
    KPD = np.zeros([mesh.NP, mesh.ND])
    # KPD = csr_matrix((mesh.NP, mesh.ND))

    for APqs, ADqs in zip(APq, ADq):
        if type(ADqs) is not np.ndarray:
            continue
        Kqds = csr_matrix(Ks[qs_right_bound][:, qs_left_bound])
        KPD += APqs @ Kqds @ ADqs.T
    return KPD


def assembly_KRD_matrix(Ks, ARr, ADq, qs_left_bound, rs, mesh):
    KRD = np.zeros([mesh.NR, mesh.ND])
    # KRD = csr_matrix((mesh.NR, mesh.ND))

    for ARrs, ADqs in zip(ARr, ADq):
        if type(ADqs) is not np.ndarray:
            continue
        Krds = csr_matrix(Ks[rs][:, qs_left_bound])
        KRD += ARrs @ Krds @ ADqs.T

    return KRD
