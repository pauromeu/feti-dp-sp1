import numpy as np
import os
from common.SubdomainsMesh import SubdomainsMesh
from mpi4py import MPI

# Get the path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths based on the current file's location
localF_path = os.path.join(current_dir, "..", "data",
                           "big", "localF_small.dat")
localK_path = os.path.join(current_dir, "..", "data",
                           "big", "localK_small.dat")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# SMALL
# Import data
fs = np.genfromtxt(localF_path)
Ks = np.genfromtxt(localK_path)

left_r = np.array([4])
right_r = np.array([7])
bottom_r = np.array([1, 2])
top_r = np.array([9, 10])

bounds_r = [left_r, right_r, bottom_r, top_r]

# Initial data
# Number of subdomains
Nsub_x = 10
Nsub_y = 10

qs_bottom_left = 0
qs_bottom_right = 3
qs_top_left = 8
qs_top_right = 11

qs = [qs_bottom_left, qs_bottom_right, qs_top_left, qs_top_right]

rs = np.setdiff1d(np.arange(len(fs)), qs)

m1 = SubdomainsMesh(Nsub_x, Nsub_y, Ks, fs, qs, rs, bounds_r)

m1.set_A_global_local_matrices()

m1.APq_array[0]

# m1.set_K_matrices()
# m1.set_B_matrices()
# m1.set_f_vectors()
# m1.set_d_vector()
# m1.set_uD_vector()
# m1.solve()
# m1.solve_parallel(comm, size, rank, returns_run_info=True)

m1.build_and_solve_parallel(comm, size, rank, True)
