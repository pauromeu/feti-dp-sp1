# Import modules
import numpy as np
import os
from mpi4py import MPI

# Import functions
from common.assembly_A_local_global_matrices import *
from common.assembly_K_matrices import *
from common.assembly_B_matrices import *
from common.assembly_f_vectors import *
from common.assembly_u_solution import *
from common.RegularSudomainsMesh import RegularSubdomainsMesh
from common.utils import *

from parallel.cg_feti import *

# Get the path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths based on the current file's location
d_path = os.path.join(current_dir, "..", "data", "small", "d.dat")
fP_path = os.path.join(current_dir, "..", "data", "small", "fP.dat")
fr_path = os.path.join(current_dir, "..", "data", "small", "fr.dat")
Ks_path = os.path.join(current_dir, "..", "data", "small", "localK.dat")
sol_path = os.path.join(current_dir, "..", "data", "small", "solution.dat")

# Import the data using the relative paths
d_dat = np.genfromtxt(d_path)
fP_dat = np.genfromtxt(fP_path)
fr_dat = np.genfromtxt(fr_path)
Ks = np.genfromtxt(Ks_path)
solution = np.genfromtxt(sol_path)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(rank)

# Initial data
# Number of subdomains
Nsub_x = 4
Nsub_y = 3

# Number of remaining nodes in each subdomain
Nr_x = 4
Nr_y = 3

# Local remaining and primal indices
rs = np.array([1, 2, 4, 5, 6, 7, 9, 10])
qs = np.array([0, 3, 8, 11])
qs_left_bound = np.array([0, 8])
qs_right = np.array([3, 11])

# Create mesh
mesh = RegularSubdomainsMesh(Nsub_x, Nsub_y, Nr_x, Nr_y)

# Transformation matrices A
# Primal nodes local-global transformation matrices
APq = create_APq_matrices(mesh)

# Remaining nodes local-global transformation matrices
ARr = create_ARr_matrices(mesh)

# Dirichlet nodes local-global transformation matrices
ADd = create_ADq_matrices(mesh)

# Stiffness matrices K
KRR = assembly_KRR_matrix(Ks, ARr, rs, mesh)
KPP = assembly_KPP_matrix(Ks, APq, qs, qs_right, mesh)
KRP, Krqs_list = assembly_KRP_matrix(Ks, APq, ARr, qs, qs_right, rs, mesh)
KPR = KRP.T
Kqrs_list = [Krqs.T for Krqs in Krqs_list]
KPD = assembly_KPD_matrix(Ks, APq, ADd, qs_left_bound, qs_right, mesh)
KRD = assembly_KRD_matrix(Ks, ARr, ADd, qs_left_bound, rs, mesh)

# Assembly B matrices
BlambdaR, Brs_list = assembly_BR_matrix(mesh, ARr)

# Dirichlet boundary conditions
# Left wall remaining
BlambdaR, BRs_list = assembly_Dirichlet_BR_matrix(
    mesh, ARr, BlambdaR, Brs_list)

# Assembly d vector
d = np.zeros(mesh.Nlambda)
d[mesh.NlambdaR:] = d_dat[np.arange(len(d_dat)) % (mesh.Nr_y - 1) != 0]

# Assembly f vectors
# fP
fP, fD = assembly_fP_fD_vectors(mesh, fP_dat)

# fR
fR = assembly_fR_vector(mesh, fr_dat)

# Matrices pre computation
KRR_inv = np.linalg.inv(KRR)

SPP = KPP - KPR @ KRR_inv @ KRP
SPP_inv = np.linalg.inv(SPP)

# Assembly uD vector
uD = d_dat[np.arange(len(d_dat)) % (mesh.Nr_y - 1) == 0]

fPH = fP - KPD @ uD
fRH = fR - KRD @ uD

IR = np.eye(mesh.NR)  # RxR identity matrix

dH = d - BlambdaR @ KRR_inv @ ((IR + KRP @ SPP_inv @
                               KPR @ KRR_inv) @ fRH - KRP @ SPP_inv @ fPH)
F = -BlambdaR @ KRR_inv @ (KRP @ SPP_inv @ KPR @ KRR_inv + IR) @ BlambdaR.T

# Solve lambda linear system
# lambda_ = np.linalg.solve(F, dH)
lambda_ = cg_parallel_feti(comm, size, rank, dH, np.zeros_like(dH), 1e-10,
                           SPP, APq, Ks, rs, Kqrs_list, BRs_list)

# Compute u
# uP
uP = SPP_inv @ (fPH - KPR @ KRR_inv @ fRH + KPR @
                KRR_inv @ BlambdaR.T @ lambda_)

# uR
uR = KRR_inv @ (IR + KRP @ SPP_inv @ KPR @ KRR_inv) @ fRH \
    - KRR_inv @ KRP @ SPP_inv @ fPH \
    - KRR_inv @ (IR + KRP @ SPP_inv @ KPR @ KRR_inv) @ BlambdaR.T @ lambda_

# Assembly u using real global coordinates
u = assembly_u_solution(mesh, uD, uP, uR, BlambdaR)

# Plot solution field as a matrix
u_mat = u.reshape((mesh.Ntot_y, mesh.Ntot_x))
if rank == 0:
    plot_sparse_matrix(u_mat, 'Solution - u field')

# Compare results with actual solution to check
if np.allclose(u, solution):
    print('The solution provided is as expected!')
else:
    print('Solution is not correct')
