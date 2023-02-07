import numpy as np


def assembly_BR_matrix(Nr_x, Nr_y, Nsub_x, Nsub_y, Nlambda, ARr):
    Nr = Nr_x*Nr_y - 4  # Number of local remaining nodes r
    NR = Nr*(Nsub_x*Nsub_y)  # Number of total remaining nodes R
    BlambdaR = np.zeros([Nlambda, NR])

    for j in range(Nsub_y):
        for i in range(Nsub_x):
            Brs = np.zeros([Nlambda, Nr])
            # Local rs nodes in boundaries
            rs_bot = np.arange(Nr_x - 2)
            rs_top = np.arange(Nr_x - 2) + (Nr_y - 2)*Nr_x + (Nr_x - 2)
            rs_left = np.arange(Nr_x - 2, (Nr_y - 2)*(Nr_x) + Nr_x - 2, Nr_x)
            rs_right = rs_left + Nr_x - 1

            Nrs_hor = Nr_x - 2
            Nrs_ver = Nr_y - 2

            # left
            if i > 0:
                lambda_left = np.arange((i-1) + j*(Nr_y - 2)*(Nsub_x - 1), Nrs_ver*(
                    Nsub_x - 1) + (i-1) + j*(Nr_y - 2)*(Nsub_x - 1), step=Nsub_x - 1)
                # print('left', lambda_left)
                for lambda_, rs in zip(lambda_left, rs_left):
                    Brs[int(lambda_), int(rs)] = -1
            # right
            if i < Nsub_x - 1:
                lambda_right = np.arange((i) + j*(Nr_y - 2)*(Nsub_x - 1), Nrs_ver*(
                    Nsub_x - 1) + (i) + j*(Nr_y - 2)*(Nsub_x - 1), step=Nsub_x - 1)
                # print('right', lambda_right)
                for lambda_, rs in zip(lambda_right, rs_right):
                    Brs[int(lambda_), int(rs)] = 1
            # bottom
            NlambdaR_hor = (Nr_y - 2)*(Nsub_y)*(Nsub_x - 1)
            if j > 0:
                lambda_bot = np.arange(NlambdaR_hor + (j - 1)*(Nr_x - 2)*Nsub_x + i*(
                    Nr_x - 2), NlambdaR_hor + (j - 1)*(Nr_x - 2)*Nsub_x + i*(Nr_x - 2) + (Nr_x - 2))
                # print('bot', lambda_bot)
                for lambda_, rs in zip(lambda_bot, rs_bot):
                    Brs[int(lambda_), int(rs)] = 1
            # top
            if j < Nsub_y - 1:
                lambda_top = np.arange(NlambdaR_hor + (j)*(Nr_x - 2)*Nsub_x + i*(
                    Nr_x - 2), NlambdaR_hor + (j)*(Nr_x - 2)*Nsub_x + i*(Nr_x - 2) + (Nr_x - 2))
                # print('top', lambda_top)
                for lambda_, rs in zip(lambda_top, rs_top):
                    Brs[int(lambda_), int(rs)] = -1

            BlambdaR += Brs @ ARr[i + j*Nsub_x].T
    return BlambdaR


def assembly_Dirichlet_BR_matrix(Nlambda, NR, Nsub_y, Nr, Nr_x, Nr_y, NlambdaR, Nsub_x, ARr, BlambdaR):
    BlambdaR_aux = np.zeros([Nlambda, NR])
    for j in range(Nsub_y):
        Brs = np.zeros([Nlambda, Nr])
        rs_left = np.arange(Nr_x - 2, (Nr_y - 2)*(Nr_x) + Nr_x - 2, Nr_x)
        lambda_left = np.arange(NlambdaR + (Nr_y - 1)
                                * j + 1, NlambdaR + (Nr_y - 1)*j + 1 + Nr_y - 2)
        # d_left = np.arange(1 + j*(Nr_y - 1), 1+j*(Nr_y - 1) + Nr_y - 2)
        for rs, lambda_ in zip(rs_left, lambda_left):
            Brs[lambda_, rs] = 1
        BlambdaR_aux += Brs @ ARr[j*Nsub_x].T

    BlambdaR += BlambdaR_aux
    return BlambdaR


def assembly_Dirichlet_BP_matrix(Nsub_x, Nsub_y, Nr_y, Nlambda, NP, Nq, NlambdaR, APq):
    BlambdaP = np.zeros([Nlambda, NP])
    for j in range(Nsub_y):
        Bqs = np.zeros([Nlambda, Nq])
        if j == 0:
            Bqs[NlambdaR, 0] = 1
        Bqs[NlambdaR + j*(Nr_y - 1) + Nr_y - 1, 2] = 1
        BlambdaP += Bqs @ APq[j*Nsub_x].T
    return BlambdaP
