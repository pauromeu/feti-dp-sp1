import os
import numpy as np
from mpi4py import MPI

from common.SubdomainsMesh import SubdomainsMesh

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get the path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths based on the current file's location
localF_path = os.path.join(current_dir, "..", "data",
                           "big", "localF.dat")
localK_path = os.path.join(current_dir, "..", "data",
                           "big", "localK.dat")


# Import data
fs = np.genfromtxt(localF_path)
Ks = np.genfromtxt(localK_path)

left_r = np.array([192, 188, 184, 471, 207, 202, 197, 412, 90, 91, 92])
right_r = np.array([268, 267, 266, 377, 28, 33, 38, 372, 15, 19, 23])
bottom_r = np.array([180, 179, 178, 455, 285, 290, 295, 500, 272, 276, 280])
top_r = np.array([104, 100, 96, 428, 119, 114, 109, 356, 9, 10, 11])

bounds_r = [left_r, right_r, bottom_r, top_r]

qs_bottom_left = 460
qs_bottom_right = 489
qs_top_left = 417
qs_top_right = 361

qs = [qs_bottom_left, qs_bottom_right, qs_top_left, qs_top_right]

rs = np.setdiff1d(np.arange(len(fs)), qs)

# Number of subdomains
# Change as desired
Nsub_x = 5
Nsub_y = 3

m = SubdomainsMesh(Nsub_x, Nsub_y, Ks, fs, qs, rs, bounds_r)

m.build_and_solve_parallel(comm, size, rank, True)
m.plot_u_boundaries()
