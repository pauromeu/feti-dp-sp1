import numpy as np


def assembly_BR_matrix_big(mesh):
    BlambdaR = np.zeros((mesh.NLambda, mesh.NR))
    Brs_list = []

    # Local rs nodes in boundaries
    rs_bot = mesh.get_remaining_numeration(mesh.bottom_r)
    rs_left = mesh.get_remaining_numeration(mesh.left_r)
    rs_right = mesh.get_remaining_numeration(mesh.right_r)
    rs_top = mesh.get_remaining_numeration(mesh.top_r)

    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            Brs = np.zeros([mesh.NLambda, mesh.Nr])

            # Left
            if i > 0:
                lambda_left = np.arange(
                    (i-1) + j*(mesh.Nr_y)*(mesh.Nsub_x - 1),
                    len(rs_left)*(mesh.Nsub_x - 1) + (i-1) +
                    j*(mesh.Nr_y)*(mesh.Nsub_x - 1),
                    step=mesh.Nsub_x - 1
                )
                for lambda_, rs in zip(lambda_left, rs_left):
                    # print("(",int(lambda_), int(rs),")", end="")
                    Brs[int(lambda_), int(rs)] = -1

            # Right
            if i < mesh.Nsub_x - 1:
                lambda_right = np.arange(
                    (i) + j*(mesh.Nr_y)*(mesh.Nsub_x - 1),
                    len(rs_right)*(mesh.Nsub_x - 1) + (i) +
                    j*(mesh.Nr_y)*(mesh.Nsub_x - 1),
                    step=mesh.Nsub_x - 1
                )
                for lambda_, rs in zip(lambda_right, rs_right):
                    # print("(",int(lambda_), int(rs),")", end="")
                    Brs[int(lambda_), int(rs)] = 1

            # Bottom
            NlambdaR_hor = (mesh.Nr_y)*(mesh.Nsub_y)*(mesh.Nsub_x - 1)
            if j > 0:
                lambda_bot = np.arange(
                    NlambdaR_hor + (j - 1)*(mesh.Nr_x) *
                    mesh.Nsub_x + i*(mesh.Nr_x),
                    NlambdaR_hor + (j - 1)*(mesh.Nr_x) *
                    mesh.Nsub_x + i*(mesh.Nr_x) + (mesh.Nr_x)
                )
                for lambda_, rs in zip(lambda_bot, rs_bot):
                    # print("(",int(lambda_), int(rs),")", end="")
                    Brs[int(lambda_), int(rs)] = 1

            # Top
            if j < mesh.Nsub_y - 1:
                lambda_top = np.arange(
                    NlambdaR_hor + (j)*(mesh.Nr_x)*mesh.Nsub_x + i*(mesh.Nr_x),
                    NlambdaR_hor + (j)*(mesh.Nr_x)*mesh.Nsub_x +
                    i*(mesh.Nr_x) + (mesh.Nr_x)
                )
                for lambda_, rs in zip(lambda_top, rs_top):
                    # print("(",int(lambda_), int(rs),")", end="")
                    Brs[int(lambda_), int(rs)] = -1

            Brs_list.append(Brs)
            BlambdaR += Brs @ mesh.ARr_array[i + j*mesh.Nsub_x].T

    return BlambdaR, Brs_list


def assembly_Dirichlet_BR_matrix_big(mesh):
    BlambdaR = mesh.BlambdaR
    BRs_list = mesh.Brs_list
    BlambdaR_aux = np.zeros([mesh.NLambda, mesh.NR])

    for j in range(mesh.Nsub_y):
        Brs = np.zeros([mesh.NLambda, mesh.Nr])
        rs_left = mesh.get_remaining_numeration(mesh.left_r)
        lambda_left = np.arange(
            mesh.NLambdaR + mesh.Nr_y*j, mesh.NLambdaR + mesh.Nr_y*j + mesh.Nr_y)
        for rs, lambda_ in zip(rs_left, lambda_left):
            Brs[lambda_, rs] = 1
        BlambdaR_aux += Brs @ mesh.ARr_array[j*mesh.Nsub_x].T
        BRs_list[j*mesh.Nsub_x] += Brs

    BlambdaR += BlambdaR_aux
    return BlambdaR, BRs_list
