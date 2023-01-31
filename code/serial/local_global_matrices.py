import numpy as np


def create_APq_matrices(Nsub_x, Nsub_y):
    """Returns an array of matrices used to transform the local coordinates of 
    primal nodes to global coordinates. The output is a matrix with a number 
    of columns equal to the number of local nodes and a number of rows equal to
    the number of global nodes. The matrix has 1's in the row-column combinations
    that show the transformation

    Args:
        Nsub_x (int): Number of subdomains in x direction
        Nsub_y (int): Number of subdomains in y direction

    Returns:
        list of numpy.ndarray: Array of length Nsub_x * Nsub_y of APq matrices used 
        to transform a local node q into global node P for each subdomain
    """

    NP = (Nsub_x + 1)*(Nsub_y + 1)  # Number of total primal nodes P
    Nq = 4  # Number of primal nodes in subdomain q
    APq_array = []

    for j in range(Nsub_y):
        for i in range(Nsub_x):
            P_nodes = [
                j*(Nsub_x + 1) + i,
                j*(Nsub_x + 1) + i + 1,
                (j + 1)*(Nsub_x + 1) + i,
                (j + 1)*(Nsub_x + 1) + i + 1
            ]
            APqs = np.zeros([NP, Nq])
            for col, node in enumerate(P_nodes):
                APqs[node, col] = 1
            APq_array.append(APqs)

    return APq_array


def create_ARr_matrices(Nsub_x, Nsub_y, Nr_x, Nr_y):
    """Returns an array of matrices used to transform the local coordinates of 
    remainig nodes to global coordinates. The output is a matrix with a number 
    of columns equal to the number of local nodes and a number of rows equal to
    the number of global nodes. The matrix has 1's in the row-column combinations
    that show the transformation

    Args:
        Nsub_x (int): Number of subdomains in x direction
        Nsub_y (int): Number of subdomains in y direction
        Nr_x (int): Number of local remaining nodes in each subdomain in x direction
        Nr_y (int): Number of local remaining nodes in each subdomain in y direction

    Returns:
        list of numpy.ndarray: Array of length Nsub_x * Nsub_y of ARr matrices used 
        to transform a local node r into global node R for each subdomain
    """
    ARr_array = []  # Array of ARr matrices for each subdomain
    Nr = Nr_x*Nr_y - 4  # Number of local remaining nodes r
    NR = Nr*(Nsub_x*Nsub_y)  # Number of total remaining nodes R

    for j in range(Nsub_y):
        for i in range(Nsub_x):
            R_nodes = []
            for k in range(Nr_y):
                if (k == 0):
                    R_nodes = np.concatenate(
                        [R_nodes, np.arange(Nr_x - 2) + Nr*Nsub_x*j + (Nr_x - 2)*i])
                elif (k == Nr_y - 1):
                    R_nodes = np.concatenate([R_nodes, np.arange(
                        Nr_x - 2) + Nr*Nsub_x*j + (Nr_x - 2)*i + (Nr_x*k - 2)*Nsub_x])
                else:
                    R_nodes = np.concatenate(
                        [R_nodes, np.arange(Nr_x) + Nr*Nsub_x*j + (Nr_x)*i + (Nr_x*k - 2)*Nsub_x])

            ARrs = np.zeros([NR, Nr])
            for col, node in enumerate(R_nodes):
                ARrs[int(node), col] = 1
            ARr_array.append(ARrs)

    return ARr_array
