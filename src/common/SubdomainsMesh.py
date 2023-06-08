import numpy as np
from scipy.sparse import csr_matrix

from common.mesh_solvers import solve_mesh, solve_parallel_mesh
from common.mesh_utils import get_F_condition_number_mesh, get_remaining_numeration_mesh, plot_u_boundaries_mesh
from parallel.cg_feti import *
from common.utils import *


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
        rs_left = mesh.get_remaining_numeration(mesh.left_r)
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

    @classmethod
    def from_problem(cls, Nsub_x, Nsub_y, problem):
        Ks = problem.generate_stiffness_matrix()
        fs = problem.generate_load_vector()
        qs = problem.get_corners()
        bounds_r = problem.get_boundaries()
        rs = np.setdiff1d(np.arange(problem.nx * problem.ny), qs)
        return cls(Nsub_x, Nsub_y, Ks, fs, qs, rs, bounds_r)

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
        self.Kqrs_list = [Krqs.T for Krqs in self.Krqs_list]
        self.KPD = assembly_KPD_matrix(
            self.Ks, self.APq_array, self.ADq_array, self.qs_left_bound, self.qs_right, self)
        self.KRD = assembly_KRD_matrix(
            self.Ks, self.ARr_array, self.ADq_array, self.qs_left_bound, self.rs, self)

    def set_B_matrices(self):
        self.BlambdaR, self.Brs_list = assembly_BR_matrix_big(self)
        self.BlambdaR, self.Brs_list = assembly_Dirichlet_BR_matrix_big(self)

    def set_d_vector(self):
        # TODO adapt for Dirichlet conditions distinct to 0
        # self.d[self.NlambdaR:] = self[np.arange(len(d_dat)) % (self.Nr_y - 1) != 0]
        self.d = np.zeros(self.NLambda)

    def set_f_vectors(self):
        self.fR = assembly_fR_vector(self)
        self.fP = assemby_fP_vector(self)
        self.fD = assemby_fD_vector(self)

    def set_uD_vector(self):
        # TODO adapt for Dirichlet conditions distinct to 0
        self.uD = np.zeros(self.ND)

    def get_remaining_numeration(self, rs_array):
        return get_remaining_numeration_mesh(self, rs_array)

    def get_F_condition_number(self):
        self.F, self.SPP, self.cond_num = get_F_condition_number_mesh(self)
        return self.cond_num

    def solve(self):
        self.uP, self.uR = solve_mesh(self)

    def solve_parallel(self, comm, size, rank, returns_run_info=False):
        self.uP, self.uR, iterations, time = solve_parallel_mesh(
            self, comm, size, rank, True
        )
        if returns_run_info:
            return [iterations, time]

    def build_and_solve(self):
        self.set_A_global_local_matrices()
        self.set_K_matrices()
        self.set_B_matrices()
        self.set_f_vectors()
        self.set_d_vector()
        self.set_uD_vector()
        self.solve()

    def build_and_solve_parallel(self, comm, size, rank, returns_run_info=False):
        self.set_A_global_local_matrices()
        self.set_K_matrices()
        self.set_B_matrices()
        self.set_f_vectors()
        self.set_d_vector()
        self.set_uD_vector()
        self.get_F_condition_number()
        run_info = self.solve_parallel(comm, size, rank, True)
        if returns_run_info:
            return run_info

    def plot_u_boundaries(self):
        plot_u_boundaries_mesh(self)
