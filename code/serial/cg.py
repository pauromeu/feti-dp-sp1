import numpy as np

import numpy as np
from numpy.linalg import inv, cholesky
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


def custom_matrix_vector_multiplication(p, KPR_shape, Ks, rs, qs, APq, Kqrs_list, BRs_list, BlambdaR, KRR_inv, KRP, SPP, Krrs_inv, LA_SPP):
    x = np.zeros(KPR_shape[0])

    print(APq, Kqrs_list, BRs_list)

    # In parallel
    for (Apqs, Kqrs, Brs) in zip(APq, Kqrs_list, BRs_list):
        print(Apqs.shape, Kqrs.shape, Krrs_inv.shape, Brs.T.shape, p.shape)
        x += Apqs @ Kqrs @ Krrs_inv @ Brs.T @ p

    alpha = solve_cholesky(LA_SPP, x)

    a = np.zeros(BlambdaR.shape[0])

    # Sync required!
    for (BRs, Kqrs, Apqs) in zip(BRs_list, Kqrs_list, APq):
        a += BRs @ Krrs_inv @ Kqrs.T @ Apqs.T @ alpha

    b = np.zeros(BlambdaR.shape[0])

    for BRs in BRs_list:
        b += BRs @ Krrs_inv @ BRs.T @ p

    Fp = a + b
    return Fp


def conjgrad(A, b, x, *args):
    Ks, rs, qs, APq, Kqrs_list, BRs_list, BlambdaR, KRR_inv, KRP, SPP = args

    Krrs = Ks[rs][:, rs]
    Krrs_inv = inv(Krrs)
    LA_SPP = cholesky(SPP)

    r = b - custom_matrix_vector_multiplication(x, A.shape, Ks, rs, qs, APq,
                                                Kqrs_list, BRs_list, BlambdaR, KRR_inv, KRP, SPP, Krrs_inv, LA_SPP)
    p = r.copy()
    rsold = np.dot(r.T, r)

    for i in range(len(b)):
        Ap = custom_matrix_vector_multiplication(
            p, A.shape, Ks, rs, qs, APq, Kqrs_list, BRs_list, BlambdaR, KRR_inv, KRP, SPP, Krrs_inv, LA_SPP)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < 1e-10:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    print(f"Number of iterations required: {i + 1}")
    return x


def cg(A, b, x, tol=1e-10):
    r = b - np.dot(A, x)
    p = r.copy()
    rsold = np.dot(r.T, r)

    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    print(f"Number of iterations required: {i + 1}")
    return x
