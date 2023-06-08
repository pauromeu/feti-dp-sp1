import numpy as np


def create_APq_matrices(mesh):
    APq_array = []  # Array of APq matrices for each subdomain

    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            if i == 0:
                P_nodes = [
                    j*mesh.Nsub_x,
                    (j + 1)*mesh.Nsub_x
                ]
            else:
                P_nodes = [
                    j*(mesh.Nsub_x) + i - 1,
                    j*(mesh.Nsub_x) + i,
                    (j + 1)*(mesh.Nsub_x) + i - 1,
                    (j + 1)*(mesh.Nsub_x) + i
                ]
            APqs = np.zeros([mesh.NP, len(P_nodes)])
            for col, node in enumerate(P_nodes):
                APqs[node, col] = 1
            APq_array.append(APqs)

    return APq_array


def create_ADq_matrices(mesh):
    ADq_array = []
    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            if i == 0:
                D_nodes = [
                    j,
                    j + 1
                ]
                ADqs = np.zeros([mesh.ND, len(D_nodes)])
                for col, node in enumerate(D_nodes):
                    ADqs[int(node), col] = 1
            else:
                ADqs = []
            ADq_array.append(ADqs)
    return ADq_array


def create_ARr_matrices_big(mesh):
    ARr_array = []
    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            ARrs = np.zeros((mesh.NR, mesh.Nr))
            R0 = (i + j*mesh.Nsub_x)*mesh.Nr
            for r in range(mesh.Nr):
                ARrs[R0 + r, r] = 1
            ARr_array.append(ARrs)
    return ARr_array
