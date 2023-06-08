import numpy as np
import numpy.linalg as LA
from mpi4py import MPI
from scipy.linalg import solve_triangular


def solve_cholesky(L, b):
    """
    Solves the system of linear equations Ax = b using the Cholesky decomposition,
    where A is a symmetric positive definite matrix and L is its Cholesky factorization.
    This function uses triangular solvers to perform forward and backward substitution.

    Parameters:
    L (ndarray): The lower triangular Cholesky factor of A
    b (ndarray): The right-hand side vector of the equation Ax = b

    Returns:
    x (ndarray): The solution vector x of Ax = b
    """
    # Solve Ly = b using forward substitution
    y = solve_triangular(L, b, lower=True)

    # Solve L.T x = y using backward substitution
    x = solve_triangular(L.T, y, lower=False)

    return x


def subdomains_mat_vec_multiplication(comm, rank, low, high, p, APq, Kqrs_list, BRs_list, Krrs_inv, LA_SPP):
    # print("This is rank ", rank, " with low:", low, " and high: ", high)
    n = BRs_list[0].shape[0]

    x_local = np.zeros(APq[0].shape[0])
    b_local = np.zeros(n)
    for (Apqs, Kqrs, Brs) in zip(APq[low:high], Kqrs_list[low:high], BRs_list[low:high]):
        x_local += Apqs @ Kqrs @ Krrs_inv @ Brs.T @ p
        b_local += Brs @ Krrs_inv @ Brs.T @ p

    x = np.zeros_like(x_local)
    comm.Allreduce(x_local, x, op=MPI.SUM)

    alpha = np.zeros_like(x)
    if rank == 0:
        alpha = solve_cholesky(LA_SPP, x)
    comm.Bcast(alpha, root=0)

    a_local = np.zeros(n)
    for (BRs, Kqrs, Apqs) in zip(BRs_list[low:high], Kqrs_list[low:high], APq[low:high]):
        a_local += BRs @ Krrs_inv @ Kqrs.T @ Apqs.T @ alpha

    a = np.zeros_like(a_local)
    comm.Allreduce(a_local, a, op=MPI.SUM)

    b = np.zeros_like(b_local)
    comm.Allreduce(b_local, b, op=MPI.SUM)

    Fp = -(a + b)
    return Fp


def cg_parallel_feti(comm, size, rank, d, lamb, tol=1e-12, returns_run_info=False, *args):
    SPP, APq, Ks, rs, Kqrs_list, BRs_list = args

    chunk = len(APq) // size
    low = rank * chunk
    high = (rank + 1) * chunk if rank != size - 1 else len(APq)

    Krrs = Ks[rs][:, rs]
    Krrs_inv = LA.inv(Krrs)
    LA_SPP = LA.cholesky(SPP)

    start = MPI.Wtime()

    # r = d - np.dot(F, lamb)
    r = d - \
        subdomains_mat_vec_multiplication(
            comm, rank, low, high, lamb, APq, Kqrs_list, BRs_list, Krrs_inv, LA_SPP
        )
    p = r.copy()
    rsold = np.dot(r.T, r)

    for i in range(len(d)):
        # Fp = np.dot(F, p)
        Fp = subdomains_mat_vec_multiplication(
            comm, rank, low, high, p, APq, Kqrs_list, BRs_list, Krrs_inv, LA_SPP
        )
        alpha = rsold / np.dot(p.T, Fp)
        lamb = lamb + alpha * p
        r = r - alpha * Fp
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    if rank == 0:
        print(f"Number of iterations required: {i + 1}")

    end = MPI.Wtime()

    if returns_run_info:
        return [lamb, i + 1, end - start]
    else:
        return lamb
