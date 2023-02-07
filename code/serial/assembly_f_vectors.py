import numpy as np


def assembly_fR_vector(Nsub_x, Nsub_y, Nr, Nr_x, Nr_y, NR, fr_dat, fR):
    fR = np.zeros(NR)
    for j in range(Nsub_y):
        for i in range(Nsub_x):
            Rs = []
            for sj in range(Nr_y):
                if sj == 0:
                    Rs = np.concatenate([Rs, np.arange(
                        j*(Nsub_x*Nr) + i*(Nr_x - 2), j*(Nsub_x*Nr) + i*(Nr_x - 2) + Nr_x - 2)])
                elif sj == Nr_y - 1:
                    Rs = np.concatenate([Rs, np.arange(j*(Nsub_x*Nr) + i*(Nr_x - 2) + Nsub_x*Nr_x*(Nr_y - 2) + Nsub_x*(Nr_x - 2),
                                                       j*(Nsub_x*Nr) + i*(Nr_x - 2) + Nsub_x*Nr_x*(Nr_y - 2) + Nsub_x*(Nr_x - 2) + Nr_x - 2)])
                else:
                    Rs = np.concatenate([Rs, np.arange(j*(Nsub_x*Nr) + i*(Nr_x) + Nsub_x*Nr_x*(sj - 1) + Nsub_x*(Nr_x - 2),
                                                       j*(Nsub_x*Nr) + i*(Nr_x) + Nsub_x*Nr_x*(sj - 1) + Nsub_x*(Nr_x - 2) + Nr_x)])
            fR[Rs.astype(int)] = fr_dat
    return fR
