import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from common.SubdomainsMesh import SubdomainsMesh
from common.HeatTransferProblem import HeatTransferProblem
from common.utils import plot_with_dual_yaxis

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Plot the efect of augmenting the size of the problem while keeping the number of subdomains
Nsub_x = 3
Nsub_y = 3

times = []
iter = []
x = []

for nx in [4, 8, 12, 16, 20]:
    ny = nx
    p = HeatTransferProblem(nx, ny)
    m = SubdomainsMesh.from_problem(Nsub_x, Nsub_y, p)
    print("N:", m.Ntot, "Nr:", m.Nr, "NP:", m.NP)
    info = m.build_and_solve_parallel(comm, 1, 0, returns_run_info=True)

    iter.append(info[0])
    times.append(info[1])
    x.append(nx*ny)

plot_with_dual_yaxis(
    x, times, iter, "Size of subdomain", "Time [s]", "# iterations", save_fig=True, filename='plots/size_subdomain'
)

m.plot_u_boundaries()

# Plot the effect of augmenting the number of subdomains while keeping the problem size constant.
Nsub_x = 1
Nsub_y = 4

ny = 24

times = []
iter = []
x = []

for Nsub_x in [1, 2, 4, 8]:
    nx = int(ny/Nsub_x)
    p = HeatTransferProblem(nx, ny)
    m = SubdomainsMesh.from_problem(Nsub_x, Nsub_y, p)
    print("N:", m.Ntot, "Nr:", m.Nr, "NP:", m.NP)
    info = m.build_and_solve_parallel(comm, 1, 0, returns_run_info=True)

    iter.append(info[0])
    times.append(info[1])
    x.append(Nsub_y*Nsub_x)

plot_with_dual_yaxis(
    x, times, iter, "Number of subdomains (main problem constant size)", "Time [s]", "# iterations", save_fig=True, filename='plots/number_subdomains'
)

m.plot_u_boundaries()
