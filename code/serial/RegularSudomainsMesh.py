class RegularSubdomainsMesh:
    def __init__(self, Nsub_x, Nsub_y, Nr_x, Nr_y):
        self.Nsub_x = Nsub_x
        self.Nsub_y = Nsub_y
        self.Nr_x = Nr_x
        self.Nr_y = Nr_y

        # Number of primal nodes
        self.Nq = 4  # local
        self.NP = (Nsub_x + 1) * (Nsub_y + 1)  # global

        # Number of remaining nsodes
        self.Nr = Nr_x*Nr_y - 4  # local
        self.NR = self.Nr * (Nsub_x*Nsub_y)  # global
