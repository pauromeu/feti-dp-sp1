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


def create_ARr_matrices(mesh):
    ARr_array = []  # Array of ARr matrices for each subdomain

    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            R_nodes = []
            for k in range(mesh.Nr_y):
                if (k == 0):
                    R_nodes = np.concatenate(
                        [R_nodes, np.arange(mesh.Nr_x - 2) + mesh.Nr*mesh.Nsub_x*j + (mesh.Nr_x - 2)*i])
                elif (k == mesh.Nr_y - 1):
                    R_nodes = np.concatenate([R_nodes, np.arange(
                        mesh.Nr_x - 2) + mesh.Nr*mesh.Nsub_x*j + (mesh.Nr_x - 2)*i + (mesh.Nr_x*k - 2)*mesh.Nsub_x])
                else:
                    R_nodes = np.concatenate(
                        [R_nodes, np.arange(mesh.Nr_x) + mesh.Nr*mesh.Nsub_x*j + (mesh.Nr_x)*i + (mesh.Nr_x*k - 2)*mesh.Nsub_x])

            ARrs = np.zeros([mesh.NR, mesh.Nr])
            for col, node in enumerate(R_nodes):
                ARrs[int(node), col] = 1
            ARr_array.append(ARrs)

    return ARr_array
