import numpy as np


class RegularSubdomainsMesh:
    def __init__(self, Nsub_x, Nsub_y, Nr_x, Nr_y):
        self.Nsub_x = Nsub_x
        self.Nsub_y = Nsub_y
        self.Nr_x = Nr_x
        self.Nr_y = Nr_y

        # Number of primal nodes
        self.Nq = 4  # local
        self.NP = (Nsub_x + 1) * (Nsub_y + 1) - (Nsub_y + 1)  # global

        # Number of remaining nsodes
        self.Nr = Nr_x*Nr_y - 4  # local
        self.NR = self.Nr * (Nsub_x*Nsub_y)  # global

        # Number of Dirichlet nodes
        self.Nd = 2  # local
        self.ND = Nsub_y + 1  # global

        # Number of lambda nodes
        self.NlambdaR = (Nsub_x*(Nr_x - 2)) * (Nsub_y - 1) + \
            (Nsub_y*(Nr_y - 2)) * (Nsub_x - 1)
        self.NlambdaP_Dir = Nsub_y + 1
        self.NlambdaR_Dir = (Nr_y - 2)*Nsub_y
        self.Nlambda = self.NlambdaR + self.NlambdaR_Dir + self.NlambdaP_Dir
