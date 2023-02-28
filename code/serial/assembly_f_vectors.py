import numpy as np


def assembly_fR_vector(mesh, fr_dat):
    fR = np.zeros(mesh.NR)
    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            Rs = []
            for sj in range(mesh.Nr_y):
                if sj == 0:
                    Rs = np.concatenate([Rs, np.arange(
                        j*(mesh.Nsub_x*mesh.Nr) + i*(mesh.Nr_x - 2), j*(mesh.Nsub_x*mesh.Nr) + i*(mesh.Nr_x - 2) + mesh.Nr_x - 2)])
                elif sj == mesh.Nr_y - 1:
                    Rs = np.concatenate([Rs, np.arange(j*(mesh.Nsub_x*mesh.Nr) + i*(mesh.Nr_x - 2) + mesh.Nsub_x*mesh.Nr_x*(mesh.Nr_y - 2) + mesh.Nsub_x*(mesh.Nr_x - 2),
                                                       j*(mesh.Nsub_x*mesh.Nr) + i*(mesh.Nr_x - 2) + mesh.Nsub_x*mesh.Nr_x*(mesh.Nr_y - 2) + mesh.Nsub_x*(mesh.Nr_x - 2) + mesh.Nr_x - 2)])
                else:
                    Rs = np.concatenate([Rs, np.arange(j*(mesh.Nsub_x*mesh.Nr) + i*(mesh.Nr_x) + mesh.Nsub_x*mesh.Nr_x*(sj - 1) + mesh.Nsub_x*(mesh.Nr_x - 2),
                                                       j*(mesh.Nsub_x*mesh.Nr) + i*(mesh.Nr_x) + mesh.Nsub_x*mesh.Nr_x*(sj - 1) + mesh.Nsub_x*(mesh.Nr_x - 2) + mesh.Nr_x)])
            fR[Rs.astype(int)] = fr_dat
    return fR


def assembly_fP_vector(mesh, fP_dat):
    indices = np.arange((mesh.Nsub_x + 1) * (mesh.Nsub_y + 1)) % (
        mesh.Nsub_x + 1) != 0  # Remove Dirichlet nodes
    fP = fP_dat[indices]
    return fP
