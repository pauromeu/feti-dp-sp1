import numpy as np


def assembly_BR_matrix(mesh, ARr):
    BlambdaR = np.zeros([mesh.Nlambda, mesh.NR])

    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            Brs = np.zeros([mesh.Nlambda, mesh.Nr])

            # Local rs nodes in boundaries
            rs_bot = np.arange(mesh.Nr_x - 2)
            rs_top = np.arange(mesh.Nr_x - 2) + \
                (mesh.Nr_y - 2)*mesh.Nr_x + (mesh.Nr_x - 2)
            rs_left = np.arange(mesh.Nr_x - 2, (mesh.Nr_y - 2)
                                * (mesh.Nr_x) + mesh.Nr_x - 2, mesh.Nr_x)
            rs_right = rs_left + mesh.Nr_x - 1

            Nrs_ver = mesh.Nr_y - 2  # Number of vertical nodes

            # Left
            if i > 0:
                lambda_left = np.arange((i-1) + j*(mesh.Nr_y - 2)*(mesh.Nsub_x - 1), Nrs_ver*(
                    mesh.Nsub_x - 1) + (i-1) + j*(mesh.Nr_y - 2)*(mesh.Nsub_x - 1), step=mesh.Nsub_x - 1)
                for lambda_, rs in zip(lambda_left, rs_left):
                    Brs[int(lambda_), int(rs)] = -1

            # Right
            if i < mesh.Nsub_x - 1:
                lambda_right = np.arange((i) + j*(mesh.Nr_y - 2)*(mesh.Nsub_x - 1), Nrs_ver*(
                    mesh.Nsub_x - 1) + (i) + j*(mesh.Nr_y - 2)*(mesh.Nsub_x - 1), step=mesh.Nsub_x - 1)
                for lambda_, rs in zip(lambda_right, rs_right):
                    Brs[int(lambda_), int(rs)] = 1

            # Bottom
            NlambdaR_hor = (mesh.Nr_y - 2)*(mesh.Nsub_y)*(mesh.Nsub_x - 1)
            if j > 0:
                lambda_bot = np.arange(NlambdaR_hor + (j - 1)*(mesh.Nr_x - 2)*mesh.Nsub_x + i*(
                    mesh.Nr_x - 2), NlambdaR_hor + (j - 1)*(mesh.Nr_x - 2)*mesh.Nsub_x + i*(mesh.Nr_x - 2) + (mesh.Nr_x - 2))
                for lambda_, rs in zip(lambda_bot, rs_bot):
                    Brs[int(lambda_), int(rs)] = 1

            # Top
            if j < mesh.Nsub_y - 1:
                lambda_top = np.arange(NlambdaR_hor + (j)*(mesh.Nr_x - 2)*mesh.Nsub_x + i*(
                    mesh.Nr_x - 2), NlambdaR_hor + (j)*(mesh.Nr_x - 2)*mesh.Nsub_x + i*(mesh.Nr_x - 2) + (mesh.Nr_x - 2))
                for lambda_, rs in zip(lambda_top, rs_top):
                    Brs[int(lambda_), int(rs)] = -1

            BlambdaR += Brs @ ARr[i + j*mesh.Nsub_x].T
    return BlambdaR


def assembly_Dirichlet_BR_matrix(mesh, ARr, BlambdaR):
    BlambdaR_aux = np.zeros([mesh.Nlambda, mesh.NR])
    for j in range(mesh.Nsub_y):
        Brs = np.zeros([mesh.Nlambda, mesh.Nr])
        rs_left = np.arange(mesh.Nr_x - 2, (mesh.Nr_y - 2)
                            * (mesh.Nr_x) + mesh.Nr_x - 2, mesh.Nr_x)
        lambda_left = np.arange(mesh.NlambdaR + (mesh.Nr_y - 1)
                                * j + 1, mesh.NlambdaR + (mesh.Nr_y - 1)*j + 1 + mesh.Nr_y - 2)
        # d_left = np.arange(1 + j*(Nr_y - 1), 1+j*(Nr_y - 1) + Nr_y - 2)
        for rs, lambda_ in zip(rs_left, lambda_left):
            Brs[lambda_, rs] = 1
        BlambdaR_aux += Brs @ ARr[j*mesh.Nsub_x].T

    BlambdaR += BlambdaR_aux
    return BlambdaR


def assembly_Dirichlet_BP_matrix(mesh, ADd):
    BlambdaP = np.zeros([mesh.Nlambda, mesh.NP])
    for j in range(mesh.Nsub_y):
        Bqs = np.zeros([mesh.Nlambda, mesh.Nd])
        if j == 0:
            Bqs[mesh.NlambdaR, 0] = 1
        Bqs[mesh.NlambdaR + j*(mesh.Nr_y - 1) + mesh.Nr_y - 1, 1] = 1
        BlambdaP += Bqs @ ADd[j*mesh.Nsub_x].T
    return BlambdaP
