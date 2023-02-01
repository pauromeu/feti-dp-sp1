import numpy as np


def assembly_KPP_matrix(Ks, APq, qs):
    """
    Returns the global stiffness matrix KPP for the primal nodes that is assembled 
    using the local stiffness matrix and the local-global transformation matrix.

    Args:
        Ks (numpy.ndarray): 2D local stiffness matrix of all nodes
        APq (list of numpy.ndarray): list of local-global trasformation matrices 
        for primal nodes
        qs (list): list of local nodes that correspond to primal nodes (e.g. [0, 3, 8, 11])

    Returns:
        numpy.ndarray: 2D global stiffness KPP matrix of dimensions P x P where P 
        is the total number of primal nodes
    """
    NP = np.shape(APq[0])[0]
    KPP = np.zeros([NP, NP])
    Kqqs = Ks[qs][:, qs]  # Obtain local Kqqs from local Ks

    for APqs in APq:
        KPP += APqs @ Kqqs @ APqs.T

    return KPP


def assembly_KRR_matrix(Ks, ARr, rs):
    """
    Returns the global stiffness matrix KRR for the remaining nodes that is assembled 
    using the local stiffness matrix and the local-global transformation matrix.

    Args:
        Ks (numpy.ndarray): 2D local stiffness matrix of all nodes
        ARr (list of numpy.ndarray): list of local-global trasformation matrices 
        for remaining nodes
        rs (list): list of local nodes that correspond to remaining nodes

    Returns:
        numpy.ndarray: 2D global stiffness KRR matrix of dimensions R x R where R 
        is the total number of remaining nodes
    """
    NR = np.shape(ARr[0])[0]
    KRR = np.zeros([NR, NR])
    Kqqs = Ks[rs][:, rs]  # Obtain local KRR from local Ks

    for ARrs in ARr:
        KRR += ARrs @ Kqqs @ ARrs.T

    return KRR


def assembly_KPR_matrix(Ks, APq, ARr, qs, rs):
    """Returns the global stiffness matrix KPR for the remaining and primal nodes that is assembled 
    using the local stiffness matrix and the local-global transformation matrix.

    Args:
        Ks (numpy.ndarray): 2D local stiffness matrix of all nodes
        APq (list of numpy.ndarray): list of local-global trasformation matrices 
        for primal nodes
        ARr (list of numpy.ndarray): list of local-global trasformation matrices 
        for remaining nodes
        qs (list): list of local nodes that correspond to primal nodes (e.g. [0, 3, 8, 11])
        rs (list): list of local nodes that correspond to remaining nodes

    Returns:
        numpy.ndarray: 2D global stiffness KPR matrix of dimensions P x R where P 
        and R are the total number of primal and remaining nodes respectively
    """
    NP = np.shape(APq[0])[0]
    NR = np.shape(ARr[0])[0]
    KPR = np.shape([NP, NR])
    Kqrs = Ks[rs][:, qs]

    for APqs, ARrs in zip(APq, ARr):
        KPR += ARrs @ Kqrs @ APqs.T

    return KPR
