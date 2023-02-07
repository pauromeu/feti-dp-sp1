import numpy as np


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
