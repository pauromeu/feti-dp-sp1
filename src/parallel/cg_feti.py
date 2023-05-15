import numpy as np
import numpy.linalg as LA
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


def subdomains_mat_vec_multiplication(p, APq, Kqrs_list, BRs_list, Krrs_inv, LA_SPP):
    # In parallel
    x = np.zeros(APq[0].shape[0])
    for (Apqs, Kqrs, Brs) in zip(APq, Kqrs_list, BRs_list):
        x += Apqs @ Kqrs @ Krrs_inv @ Brs.T @ p

    alpha = solve_cholesky(LA_SPP, x)

    n = BRs_list[0].shape[0]

    a = np.zeros(n)
    for (BRs, Kqrs, Apqs) in zip(BRs_list, Kqrs_list, APq):
        a += BRs @ Krrs_inv @ Kqrs.T @ Apqs.T @ alpha

    b = np.zeros(n)
    for BRs in BRs_list:
        b += BRs @ Krrs_inv @ BRs.T @ p

    Fp = -(a + b)
    return Fp


def cg_feti(d, lamb, tol=1e-10, *args):
    SPP, APq, Ks, rs, Kqrs_list, BRs_list = args

    Krrs = Ks[rs][:, rs]
    Krrs_inv = LA.inv(Krrs)
    LA_SPP = LA.cholesky(SPP)

    # r = d - np.dot(F, lamb)
    r = d - \
        subdomains_mat_vec_multiplication(
            lamb, APq, Kqrs_list, BRs_list, Krrs_inv, LA_SPP)
    p = r.copy()
    rsold = np.dot(r.T, r)

    for i in range(len(d)):
        # Fp = np.dot(F, p)
        Fp = subdomains_mat_vec_multiplication(
            p, APq, Kqrs_list, BRs_list, Krrs_inv, LA_SPP)
        alpha = rsold / np.dot(p.T, Fp)
        lamb = lamb + alpha * p
        r = r - alpha * Fp
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    print(f"Number of iterations required: {i + 1}")
    return lamb
