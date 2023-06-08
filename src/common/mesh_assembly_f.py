import numpy as np


def assembly_fR_vector(mesh):
    fR = np.zeros(mesh.NR)
    fs = mesh.fs.copy()
    fr = np.delete(fs, mesh.qs)
    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            start = (i+j*mesh.Nsub_x)*mesh.Nr
            end = start + mesh.Nr
            fR[start:end] = fr
    return fR


def assemby_fP_vector(mesh):
    fP = np.zeros(mesh.NP)
    for j in range(mesh.Nsub_y):
        for i in range(mesh.Nsub_x):
            if i > 0:
                fq = mesh.fs[mesh.qs]
            else:
                fq = mesh.fs[mesh.qs_right]
            fP += mesh.APq_array[i + j*mesh.Nsub_x] @ fq
    return fP


def assemby_fD_vector(mesh):
    fD = np.zeros(mesh.ND)
    for j in range(mesh.Nsub_y):
        fd = mesh.fs[mesh.qs_left_bound]
        fD += mesh.ADq_array[j*mesh.Nsub_x] @ fd
    return fD
