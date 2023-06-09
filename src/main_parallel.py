from common.HeatTransferProblem import HeatTransferProblem
from common.SubdomainsMesh import SubdomainsMesh
from mpi4py import MPI

# Run this file using the command: mpiexec -n 4 python main_parallel.py
# Change 4 for the desired number of nodes.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
