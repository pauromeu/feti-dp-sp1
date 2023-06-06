import numpy as np
from common.assembly_A_local_global_matrices import *
from common.assembly_K_matrices import *


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
        rs_left = mesh.left_r
        lambda_left = np.arange(
            mesh.NLambdaR + mesh.Nr_y*j, mesh.NLambdaR + mesh.Nr_y*j + mesh.Nr_y)
        for rs, lambda_ in zip(rs_left, lambda_left):
            Brs[lambda_, rs] = 1
        BlambdaR_aux += Brs @ mesh.ARr_array[j*mesh.Nsub_x].T
        BRs_list[j*mesh.Nsub_x] += Brs

    BlambdaR += BlambdaR_aux
    return BlambdaR, BRs_list


def assembly_fR_vector(mesh):
    fR = np.zeros(mesh.NR)
    fs = mesh.fs.copy()
    fr = np.delete(fs, mesh.qs)
    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            start = (i+j*mesh.Nsub_x)*mesh.Nr
            end = start + mesh.Nr
            fR[start:end] = fr
    return fR


def assemby_fP_vector(mesh):
    fP = np.zeros(mesh.NP)
    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            if i > 0:
                # fq = np.zeros(mesh.Nr)
                # qs_loc = [
                #     mesh.Nsub_x*j+ i - 1,
                #     mesh.Nsub_x*j + i,
                #     mesh.Nsub_x*(j + 1) + i - 1,
                #     mesh.Nsub_x*(j + 1) + i
                # ]
                fq = mesh.fs[mesh.qs]
            else:
                fq = mesh.fs[mesh.qs_right]
            fP += mesh.APq_array[i + j*mesh.Nsub_x] @ fq
    return fP


def assemby_fD_vector(mesh):
    fD = np.zeros(mesh.ND)
    for j in range(mesh.Nsub_y):
        fd = mesh.fs[mesh.qs_left_bound]
        fD += mesh.ADq_array[j*mesh.Nsub_x] @ fd
    return fD


class SubdomainsMesh:
    def __init__(self, Nsub_x, Nsub_y, Ks, fs, qs, rs, bounds_r):
        self.Nsub_x = Nsub_x
        self.Nsub_y = Nsub_y

        # Number of primal nodes
        self.Nq = 4  # local
        self.NP = (Nsub_x + 1) * (Nsub_y + 1) - (Nsub_y + 1)  # global
        self.qs = qs  # local indices of primal nodes
        self.qs_right = np.array([qs[1], qs[3]])
        self.qs_left_bound = np.array([qs[0], qs[2]])

        # Number of remaining nodes
        self.Nr = len(fs) - 4  # local
        self.NR = self.Nr * (Nsub_x * Nsub_y)  # global
        self.rs = rs  # local indices of remaining nodes

        # Number of Dirichlet nodes
        self.ND = Nsub_y + 1  # global

        # K matrix
        self.Ks = Ks  # local K matrix

        # f vector
        self.fs = fs  # local f vector

        # Boundaries of subdomain
        self.left_r, self.right_r, self.bottom_r, self.top_r = bounds_r
        assert (len(self.left_r) == len(self.right_r))
        assert (len(self.bottom_r) == len(self.top_r))
        self.Nr_y = len(self.left_r)
        self.Nr_x = len(self.bottom_r)

        # Number of Lagrange multipliers (lambda)
        self.NlambdaR = sum(len(b)
                            for b in bounds_r)  # local number of lambda_r
        self.NLambdaR = Nsub_x * len(self.bottom_r) * (Nsub_y - 1) + \
            Nsub_y * len(self.left_r) * (Nsub_x -
                                         1)  # global number of Lambda_r
        # global number of lambda dirichlet (remaining)
        self.NLambdaR_Dir = len(self.left_r) * Nsub_y
        self.NLambda = self.NLambdaR + self.NLambdaR_Dir  # global number of Lambda

        # Number of nodes
        self.Ntot = self.NP + self.ND + self.NR \
            - len(self.left_r) * (Nsub_x - 1) * Nsub_y \
            - len(self.top_r) * (Nsub_y - 1) * Nsub_x

    def __repr__(self):
        return 'SubdomainsMesh()'

    def __str__(self):
        return 'SubdomainsMesh \n' + \
            'Number of subdomains in x: ' + str(self.Nsub_x) + '\n' + \
            'Number of subdomains in y: ' + str(self.Nsub_y) + '\n' + \
            'Number of nodes r: ' + str(self.Nr) + '\n' + \
            'Number of nodes R: ' + str(self.NR) + '\n'

    def set_APq_matrices(self):
        self.APq_array = create_APq_matrices(self)

    def set_ARr_matrices(self):
        self.ARr_array = create_ARr_matrices_big(self)

    def set_ADq_matrices(self):
        self.ADq_array = create_ADq_matrices(self)

    def set_A_global_local_matrices(self):
        self.set_APq_matrices()
        self.set_ARr_matrices()
        self.set_ADq_matrices()

    def set_K_matrices(self):
        self.KRR = assembly_KRR_matrix(self.Ks, self.ARr_array, self.rs, self)
        self.KPP = assembly_KPP_matrix(
            self.Ks, self.APq_array, self.qs, self.qs_right, self)
        self.KRP, self.Krqs_list = assembly_KRP_matrix(
            self.Ks, self.APq_array, self.ARr_array, self.qs, self.qs_right, self.rs, self)
        self.KPR = self.KRP.T
        self.KPD = assembly_KPD_matrix(
            self.Ks, self.APq_array, self.ADq_array, self.qs_left_bound, self.qs_right, self)
        self.KRD = assembly_KRD_matrix(
            self.Ks, self.ARr_array, self.ADq_array, self.qs_left_bound, self.rs, self)

    def set_B_matrices(self):
        self.BlambdaR, self.Brs_list = assembly_BR_matrix_big(self)
        self.BlambdaR, self.Brs_list = assembly_Dirichlet_BR_matrix_big(self)

    def set_d_vector(self):
        # TODO adapt for Dirichlet conditions distinct to 0
        self.d = np.zeros(self.NLambda)
        # self.d[self.NlambdaR:] = self[np.arange(len(d_dat)) % (self.Nr_y - 1) != 0]

    def set_f_vectors(self):
        self.fR = assembly_fR_vector(self)
        self.fP = assemby_fP_vector(self)
        self.fD = assemby_fD_vector(self)

    def set_uD_vector(self):
        # TODO adapt for Dirichlet conditions distinct to 0
        self.uD = np.zeros(self.ND)

    def get_remaining_numeration(self, rs_array):
        rs_array_renum = np.array([])
        qs = np.sort(self.qs)[::-1]
        assert np.all(np.intersect1d(qs, rs_array).size ==
                      0), "Some remaining nodes have primal numeration"
        assert np.all(rs_array < self.Nr + len(qs)
                      ), "Numeration must be within the number of maximum nodes"
        for r in rs_array:
            assigned = False
            for i, q in enumerate(qs):
                if r > q:
                    r_renum = r - (len(qs) - i)
                    assigned = True
                    break
            if assigned == False:
                r_renum = r
            rs_array_renum = np.append(rs_array_renum, r_renum)
        return rs_array_renum.astype(int)

    def assembly_u_solution_big(self):
        N = self.Ntot
        u = np.array()

    def solve(self):
        KRR_inv = np.linalg.inv(self.KRR)
        # KRR_inv = inv(KRR)

        SPP = self.KPP - self.KPR @ KRR_inv @ self.KRP
        SPP_inv = np.linalg.inv(SPP)
        # SPP_inv = inv(SPP)

        fPH = self.fP - self.KPD @ self.uD
        fRH = self.fR - self.KRD @ self.uD

        IR = np.eye(self.NR)  # RxR identity matrix
        # IR = identity(mesh.NR)

        dH = self.d - self.BlambdaR @ KRR_inv @ (
            (IR + self.KRP @ SPP_inv @ self.KPR @ KRR_inv) @ fRH - self.KRP @ SPP_inv @ fPH)
        F = -self.BlambdaR @ KRR_inv @ (self.KRP @ SPP_inv @
                                        self.KPR @ KRR_inv + IR) @ self.BlambdaR.T

        lambda_ = np.linalg.solve(F, dH)

        # Compute u
        # uP
        self.uP = SPP_inv @ (fPH - self.KPR @ KRR_inv @ fRH +
                             self.KPR @ KRR_inv @ self.BlambdaR.T @ lambda_)

        # uR
        self.uR = KRR_inv @ (IR + self.KRP @ SPP_inv @ self.KPR @ KRR_inv) @ fRH \
            - KRR_inv @ self.KRP @ SPP_inv @ fPH \
            - KRR_inv @ (IR + self.KRP @ SPP_inv @ self.KPR @
                         KRR_inv) @ self.BlambdaR.T @ lambda_

    def assembly_u_vector(self):
        u = np.zeros(self.Ntot)
