from mpi4py import MPI
import numpy as np

from parallel.cg_feti import cg_parallel_feti


def solve_mesh(mesh):
    KRR_inv = np.linalg.inv(mesh.KRR)
    # KRR_inv = inv(KRR)

    SPP = mesh.KPP - mesh.KPR @ KRR_inv @ mesh.KRP
    SPP_inv = np.linalg.inv(SPP)
    # SPP_inv = inv(SPP)

    fPH = mesh.fP - mesh.KPD @ mesh.uD
    fRH = mesh.fR - mesh.KRD @ mesh.uD

    IR = np.eye(mesh.NR)  # RxR identity matrix
    # IR = identity(mesh.NR)

    dH = mesh.d - mesh.BlambdaR @ KRR_inv @ (
        (IR + mesh.KRP @ SPP_inv @ mesh.KPR @ KRR_inv) @ fRH - mesh.KRP @ SPP_inv @ fPH)
    F = -mesh.BlambdaR @ KRR_inv @ (mesh.KRP @ SPP_inv @
                                    mesh.KPR @ KRR_inv + IR) @ mesh.BlambdaR.T

    lambda_ = np.linalg.solve(F, dH)

    # Compute u
    # uP
    uP = SPP_inv @ (fPH - mesh.KPR @ KRR_inv @ fRH +
                    mesh.KPR @ KRR_inv @ mesh.BlambdaR.T @ lambda_)

    # uR
    uR = KRR_inv @ (IR + mesh.KRP @ SPP_inv @ mesh.KPR @ KRR_inv) @ fRH \
        - KRR_inv @ mesh.KRP @ SPP_inv @ fPH \
        - KRR_inv @ (IR + mesh.KRP @ SPP_inv @ mesh.KPR @
                     KRR_inv) @ mesh.BlambdaR.T @ lambda_

    return [uP, uR]


def solve_parallel_mesh(mesh, comm, size, rank, returns_run_info=False):
    KRR_inv = np.linalg.inv(mesh.KRR)
    SPP = mesh.KPP - mesh.KPR @ KRR_inv @ mesh.KRP
    SPP_inv = np.linalg.inv(SPP)

    fPH = mesh.fP - mesh.KPD @ mesh.uD
    fRH = mesh.fR - mesh.KRD @ mesh.uD

    IR = np.eye(mesh.NR)  # RxR identity matrix

    dH = mesh.d - mesh.BlambdaR @ KRR_inv @ ((IR + mesh.KRP @ SPP_inv @
                                              mesh.KPR @ KRR_inv) @ fRH - mesh.KRP @ SPP_inv @ fPH)
    start = MPI.Wtime()

    lambda_, iterations, time = cg_parallel_feti(
        comm, size, rank, mesh, dH, np.zeros_like(dH), 1e-10, True)

    end = MPI.Wtime()
    if rank == 0:
        print("Time to run CG with", size,
              "processors ->", round(end - start, 5), "s")

    # Compute u
    # uP
    uP = SPP_inv @ (fPH - mesh.KPR @ KRR_inv @ fRH +
                    mesh.KPR @ KRR_inv @ mesh.BlambdaR.T @ lambda_)

    # uR
    uR = KRR_inv @ (IR + mesh.KRP @ SPP_inv @ mesh.KPR @ KRR_inv) @ fRH \
        - KRR_inv @ mesh.KRP @ SPP_inv @ fPH \
        - KRR_inv @ (IR + mesh.KRP @ SPP_inv @ mesh.KPR @
                     KRR_inv) @ mesh.BlambdaR.T @ lambda_

    if returns_run_info:
        return [uP, uR, iterations, time]

    else:
        return [uP, uR]


def precompute_matrices_solver_mesh(mesh):
    KRR_inv = np.linalg.inv(mesh.KRR)
    SPP = mesh.KPP - mesh.KPR @ KRR_inv @ mesh.KRP
    Krrs_inv = np.linalg.inv(mesh.Ks[mesh.rs][:, mesh.rs])
    Krrs = mesh.Ks[mesh.rs][:, mesh.rs]
    Krrs_inv = np.linalg.inv(Krrs)
    LA_SPP = np.linalg.cholesky(SPP)

    return SPP, LA_SPP, Krrs_inv
