import numpy as np


def assembly_u_solution(mesh, uD, uP, uR, BlambdaR):
    u = np.zeros(mesh.Ntot)
    nodesD = np.arange(0, (mesh.Nsub_y + 1)*2*mesh.Ntot_x, 2*mesh.Ntot_x)
    nodesP = np.array([])
    for j in range(mesh.Nsub_y + 1):
        nodesP = np.concatenate((nodesP, np.arange(mesh.Nr_x - 1 + j*2*mesh.Ntot_x, mesh.Nr_x -
                                1 + j*2*mesh.Ntot_x + mesh.Nsub_x*(mesh.Nr_x - 1), step=mesh.Nr_x - 1))).astype(int)
    repeatedR = np.where((BlambdaR == -1).any(axis=0))[0]
    localR = np.setdiff1d(np.arange(0, mesh.NR), repeatedR)
    nodesR = np.setdiff1d(np.arange(0, mesh.Ntot),
                          np.concatenate((nodesP, nodesD)))
    u[nodesD] = uD
    u[nodesP] = uP
    u[nodesR] = uR[localR]
    return u
