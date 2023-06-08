import numpy as np
from scipy.sparse import csr_matrix

from common.mesh_assembly_A import create_ADq_matrices, create_APq_matrices, create_ARr_matrices_big
from common.mesh_assembly_B import assembly_BR_matrix_big, assembly_Dirichlet_BR_matrix_big
from common.mesh_assembly_f import assembly_fR_vector, assemby_fP_vector, assemby_fD_vector
from common.mesh_assembly_K import assembly_KPD_matrix, assembly_KPP_matrix, assembly_KRR_matrix, assembly_KRP_matrix, assembly_KRD_matrix
from common.mesh_solvers import precompute_matrices_solver_mesh, solve_mesh, solve_parallel_mesh
from common.mesh_utils import get_F_condition_number_mesh, get_remaining_numeration_mesh, plot_u_boundaries_mesh
from parallel.cg_feti import *
from common.utils import *


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
        self.KRR = assembly_KRR_matrix(self)
        self.KPP = assembly_KPP_matrix(self)
        self.KRP, self.Krqs_list = assembly_KRP_matrix(self)
        self.KPR = self.KRP.T
        self.Kqrs_list = [Krqs.T for Krqs in self.Krqs_list]
        self.KPD = assembly_KPD_matrix(self)
        self.KRD = assembly_KRD_matrix(self)

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

    def precompute_matrices_solver(self):
        self.SPP, self.LA_SPP, self.Krrs_inv = \
            precompute_matrices_solver_mesh(self)

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
        self.precompute_matrices_solver()
        self.solve()

    def build_and_solve_parallel(self, comm, size, rank, returns_run_info=False):
        self.set_A_global_local_matrices()
        self.set_K_matrices()
        self.set_B_matrices()
        self.set_f_vectors()
        self.set_d_vector()
        self.set_uD_vector()
        self.precompute_matrices_solver()
        # self.get_F_condition_number()
        run_info = self.solve_parallel(comm, size, rank, True)
        if returns_run_info:
            return run_info

    def plot_u_boundaries(self):
        plot_u_boundaries_mesh(self)
