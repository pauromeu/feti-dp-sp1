import numpy as np
import os
from common.HeatTransferProblem import HeatTransferProblem
from common.SubdomainsMesh import SubdomainsMesh
from mpi4py import MPI

# # Get the path of the current file
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the relative paths based on the current file's location
# localF_path = os.path.join(current_dir, "..", "data",
#                            "big", "localF_small.dat")
# localK_path = os.path.join(current_dir, "..", "data",
#                            "big", "localK_small.dat")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# # SMALL
# # Import data
# fs = np.genfromtxt(localF_path)
# Ks = np.genfromtxt(localK_path)

# left_r = np.array([4])
# right_r = np.array([7])
# bottom_r = np.array([1, 2])
# top_r = np.array([9, 10])

# bounds_r = [left_r, right_r, bottom_r, top_r]

# Initial data
# Number of subdomains
# Nsub_x = 5
# Nsub_y = 5

# qs_bottom_left = 0
# qs_bottom_right = 3
# qs_top_left = 8
# qs_top_right = 11

# qs = [qs_bottom_left, qs_bottom_right, qs_top_left, qs_top_right]
# rs = np.setdiff1d(np.arange(len(fs)), qs)

# m1 = SubdomainsMesh(Nsub_x, Nsub_y, Ks, fs, qs, rs, bounds_r)

# m1.build_and_solve_parallel(comm, size, rank, True)
# m1.plot_u_boundaries()


# Heat problem data, change as desired
Nsub_x = 4  # Number of subdomains in x
Nsub_y = 4  # Number os subdomains in y

nx = 10  # Nodes in a subdomain in x
ny = 10  # Nodes in a subdomain in y

p = HeatTransferProblem(nx, ny)
m2 = SubdomainsMesh.from_problem(Nsub_x, Nsub_y, p)

m2.build_and_solve_parallel(comm, size, rank, True)

# if rank == 0:
#     m2.plot_u_boundaries()