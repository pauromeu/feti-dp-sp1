import numpy as np

from common.utils import plot_sparse_matrix


def plot_u_boundaries_mesh(mesh):
    Nbound_x = (mesh.Nsub_x + 1) + mesh.Nsub_x * len(mesh.bottom_r)
    Nbound_y = (mesh.Nsub_y + 1) + mesh.Nsub_y * len(mesh.left_r)
    Nbound_xs = len(mesh.bottom_r) + 2
    Nbound_ys = len(mesh.left_r) + 2

    u = np.zeros(Nbound_x * Nbound_y) - np.max(mesh.uR) * 0.15

    idxs_D_uD = np.arange(
        0,
        Nbound_x*(Nbound_ys - 1)*mesh.Nsub_y + 1,
        Nbound_x*(Nbound_ys - 1)
    )

    u[idxs_D_uD] = mesh.uD

    idxs_P_uD = np.array([])
    idxs_bot_uD = np.array([])
    for j in range(mesh.Nsub_y + 1):
        idxs_P_uD_j = np.arange(
            Nbound_x*(Nbound_ys - 1)*j,
            Nbound_x*(Nbound_ys - 1)*j +
            (Nbound_xs - 1)*mesh.Nsub_x + 1,
            (Nbound_xs - 1)
        )

        idxs_P_uD = np.concatenate((
            idxs_P_uD,
            idxs_P_uD_j
        ))

        idxs_bot_uD_j = np.arange(
            Nbound_x*(Nbound_ys - 1)*j,
            Nbound_x*(Nbound_ys - 1)*j + (Nbound_xs - 1)*mesh.Nsub_x + 1
        )

        idxs_bot_uD_j = np.setdiff1d(idxs_bot_uD_j, idxs_P_uD)
        idxs_bot_uD = np.concatenate((idxs_bot_uD, idxs_bot_uD_j))

    idxs_P_uD = idxs_P_uD.astype(int)
    idxs_P_uD = np.setdiff1d(idxs_P_uD, idxs_D_uD.astype(int))
    idxs_bot_uD = idxs_bot_uD.astype(int)

    bottom_bound_r = mesh.get_remaining_numeration(mesh.bottom_r)
    top_bound_r = mesh.get_remaining_numeration(mesh.top_r)

    idxs_bot_uR = np.array([])
    for s in range(mesh.Nsub_x*mesh.Nsub_y):
        idxs_bot_uR = np.concatenate(
            (idxs_bot_uR, bottom_bound_r + s*mesh.Nr)).astype(int)

    for i in range(mesh.Nsub_x):
        offset = (mesh.Nsub_x*(mesh.Nsub_y - 1)) * mesh.Nr
        idxs_bot_uR = np.concatenate((
            idxs_bot_uR, top_bound_r + i*mesh.Nr + offset)).astype(int)

    idxs_left_uD = []
    for i in range(mesh.Nsub_x + 1):
        idxs_left_uD_j = np.arange(
            i*(Nbound_xs - 1) + 0,
            i*(Nbound_xs - 1) + Nbound_x*Nbound_y,
            Nbound_x
        )
        idxs_left_uD_j = np.setdiff1d(idxs_left_uD_j, idxs_P_uD)
        idxs_left_uD_j = np.setdiff1d(idxs_left_uD_j, idxs_D_uD)
        idxs_left_uD = np.concatenate((idxs_left_uD, idxs_left_uD_j))

    idxs_left_uD = idxs_left_uD.astype(int)

    left_bound_r = mesh.get_remaining_numeration(mesh.left_r)
    right_bound_r = mesh.get_remaining_numeration(mesh.right_r)

    idxs_left_uR = np.array([])
    for i in range(mesh.Nsub_x):
        for j in range(mesh.Nsub_y):
            offset = (j*mesh.Nsub_x + i) * mesh.Nr
            idxs_left_uR = np.concatenate(
                (idxs_left_uR, left_bound_r + offset)
            )

    for j in range(mesh.Nsub_y):
        offset = (mesh.Nsub_x*j + mesh.Nsub_x - 1)*mesh.Nr
        idxs_left_uR = np.concatenate(
            (idxs_left_uR, right_bound_r + offset)
        )

    idxs_left_uR = idxs_left_uR.astype(int)

    u[idxs_P_uD] = mesh.uP
    u[idxs_bot_uD] = mesh.uR[idxs_bot_uR]
    u[idxs_left_uD] = mesh.uR[idxs_left_uR]

    u_mat = u.reshape((Nbound_y, Nbound_x))
    plot_sparse_matrix(u_mat, 'Solution - u field')


def get_remaining_numeration_mesh(mesh, rs_array):
    rs_array_renum = np.array([])
    qs = np.sort(mesh.qs)[::-1]
    assert np.all(np.intersect1d(qs, rs_array).size ==
                  0), "Some remaining nodes have primal numeration"
    assert np.all(rs_array < mesh.Nr + len(qs)
                  ), "Numeration must be within the number of maximum nodes"
    for r in rs_array:
        assigned = False
        for i, q in enumerate(qs):
            if r > q:
                r_renum = r - (len(qs) - i)
                assigned = True
                break
        if assigned == False:
            r_renum = r
        rs_array_renum = np.append(rs_array_renum, r_renum)
    return rs_array_renum.astype(int)


def get_F_condition_number_mesh(mesh):
    KRR_inv = np.linalg.inv(mesh.KRR)
    SPP = mesh.KPP - mesh.KPR @ KRR_inv @ mesh.KRP
    SPP_inv = np.linalg.inv(SPP)
    IR = np.eye(mesh.NR)  # RxR identity matrix
    F = -mesh.BlambdaR @ KRR_inv @ (mesh.KRP @ SPP_inv @
                                    mesh.KPR @ KRR_inv + IR) @ mesh.BlambdaR.T
    cond_num = np.linalg.cond(F)
    return [F, SPP, cond_num]
