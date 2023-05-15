# FETI DP method implementation
This repository contains the code and the report of my semester project in the MSc in Computational Science and Engineering at EPFL.

## Parallel implementantion

The main requirements to use MPI and Python are:
- Python 3.6 or higher
- MPI implementation (e.g., OpenMPI or MPICH)
- Python packages: `numpy`, `mpi4py`, etc.

To run a Python script with MPI, use the `mpiexec` command followed by `-n` and the number of processes to use. For example, to use 4 processes, use the following command:

```<bash>
mpiexec -n 4 python script.py
```
