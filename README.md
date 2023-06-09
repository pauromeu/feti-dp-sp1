# FETI DP method implementation
This repository contains the code and the report of my semester project in the MSc in Computational Science and Engineering at EPFL.

## Code organization

The code is contained within the `src/` folder. Two implementations of the method have been made, one serial and one parallel. The functions used are located in the corresponding folders `src/serial/`, `src/parallel/`, and `src/common/`. The `src/common/` folder contains useful functions for both implementations.

There are 4 main files provided that allow reproducing the figures presented in the report:

- `src/main_serial.ipynb`: Reproduces the solution of a small analytical structural problem. The data for this problem is located in `data/small/`.
- `src/main_h_effect.py`: Reproduces the figures of the subdomain effect from the paper.
- `src/main_parallel.py`: Allows obtaining the execution times presented in the first graph of the paper when run with 1, 2, 3, 4, 6, and 8 nodes. The graph representation is located in `src/analysis`.
- `src/main_parallel_big.py`: Allows obtaining the solution of the large problem proposed to simulate architected metamaterials. The data for this problem is located in `data/big/`.

Additionally, two classes are provided to facilitate the usage of the code:

- `src/common/SubdomainsMesh.py`: Mesh object that contains and calculates all the elements to solve a two-dimensional structural problem using FETI-DP. It accepts Dirichlet conditions on the left nodes. The main input parameters are described in the class constructor. It allows defining the number of subdomains to use in each direction and passes the parameters that define the geometry of one of the subdomains. The following image helps to understand the type of problems to solve.
![SubdomainsMesh](/report/figures/subdomains_mesh.png)
The main method of this class is `SubdomainsMesh.build_and_solve_parallel(...)`, which constructs all the matrices defining the problem and solves the resulting system of linear equations using the FETI-DP method.
Another important method is `SubdomainsMesh.plot_u_boundaries()`. This method plots the solution in the boundaries of all subdomains. An example is shown here.
![SmallMesh](/report/figures/serial_mesh.png)

- `src/common/HeatTransferProblem.py`: Problem object that defines the stiffness matrix and the source vector for a two-dimensional heat conduction problem. The only arguments for the constructor are the number of nodes in x and y dimensions to create. This problem can be solved using the `SubdomainsMesh` class and its second constructor `SubdomainsMesh.from_problem(Nsub_x, Nsub_y, HeatTransferProblem(nx, ny))`. This allows quickly creating problems of different sizes and numbers of subdomains to test the method's efficiency. For example, the following figure shows the solution for a mesh with `Nsub_x = 6` and `Nsub_y = 4` where each subdomain has `nx = 10` x `ny = 8` nodes:
![HeatMesh](/report/figures/heat_mesh.png)

## Run parallel implementation

The main requirements to use MPI and Python are:
- Python 3.6 or higher
- MPI implementation (e.g., OpenMPI or MPICH)
- Python packages: `numpy`, `mpi4py`, etc.

To run a Python script with MPI, use the `mpiexec` command followed by `-n` and the number of processes to use. For example, to use 4 processes, use the following command:

```<bash>
mpiexec -n 4 python script.py
```
