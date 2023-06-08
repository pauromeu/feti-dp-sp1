import numpy as np
from scipy.sparse import csr_matrix


def assembly_KPP_matrix(mesh):
    KPP = np.zeros([mesh.NP, mesh.NP])
    # KPP = csr_matrix((mesh.NP, mesh.NP))

    for APqs in mesh.APq_array:
        if np.shape(APqs)[1] == len(mesh.qs):
            # Obtain local Kqqs from local Ks
            Kqqs = csr_matrix(mesh.Ks[mesh.qs][:, mesh.qs])
        else:
            # Obtain local Kqqs from local Ks
            Kqqs = csr_matrix(
                mesh.Ks[mesh.qs_left_bound][:, mesh.qs_left_bound]
            )
        KPP += APqs @ Kqqs @ APqs.T

    return KPP


def assembly_KRR_matrix(mesh):
    KRR = np.zeros([mesh.NR, mesh.NR])
    # KRR = csr_matrix((mesh.NR, mesh.NR))
    # Obtain local KRR from local Ks
    Kqqs = csr_matrix(mesh.Ks[mesh.rs][:, mesh.rs])

    for ARrs in mesh.ARr_array:
        KRR += ARrs @ Kqqs @ ARrs.T

    return KRR


def assembly_KRP_matrix(mesh):
    KRP = np.zeros([mesh.NR, mesh.NP])
    # KRP = csr_matrix((mesh.NR, mesh.NP))
    Krqs_list = []

    for APqs, ARrs in zip(mesh.APq_array, mesh.ARr_array):
        if np.shape(APqs)[1] == len(mesh.qs):
            # Obtain local Kqqs from local Ks
            Kqrs = csr_matrix(mesh.Ks[mesh.rs][:, mesh.qs])
        else:
            # Obtain local Kqqs from local Ks
            Kqrs = csr_matrix(mesh.Ks[mesh.rs][:, mesh.qs_left_bound])
        KRP += ARrs @ Kqrs @ APqs.T
        Krqs_list.append(Kqrs)

    return KRP, Krqs_list


def assembly_KPD_matrix(mesh):
    KPD = np.zeros([mesh.NP, mesh.ND])
    # KPD = csr_matrix((mesh.NP, mesh.ND))

    for APqs, ADqs in zip(mesh.APq_array, mesh.ADq_array):
        if type(ADqs) is not np.ndarray:
            continue
        Kqds = csr_matrix(mesh.Ks[mesh.qs_right][:, mesh.qs_left_bound])
        KPD += APqs @ Kqds @ ADqs.T
    return KPD


def assembly_KRD_matrix(mesh):
    KRD = np.zeros([mesh.NR, mesh.ND])
    # KRD = csr_matrix((mesh.NR, mesh.ND))

    for ARrs, ADqs in zip(mesh.ARr_array, mesh.ADq_array):
        if type(ADqs) is not np.ndarray:
            continue
        Krds = csr_matrix(mesh.Ks[mesh.rs][:, mesh.qs_left_bound])
        KRD += ARrs @ Krds @ ADqs.T

    return KRD
