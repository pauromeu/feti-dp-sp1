import numpy as np

from common.utils import plot_sparse_matrix


def modify_diagonal(matrix):
    # Get the number of rows and columns in the matrix
    rows, cols = matrix.shape
    # Create a copy of the input matrix
    modified_matrix = matrix.copy()

    # Iterate over each row
    for i in range(rows):
        # Calculate the sum of elements in the current row with opposite signs
        opposite_sum = -np.sum(np.delete(matrix[i], i))

        # Set the diagonal element to the opposite sum
        modified_matrix[i, i] = opposite_sum

    return modified_matrix


class HeatTransferProblem:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.N = nx * ny  # Total number of nodes

    def generate_stiffness_matrix(self):
        # Create the main diagonal
        main_diag = 4 * np.ones(self.N)

        # Create the diagonals for the east and west neighbors
        east_west_diag = -1 * np.ones(self.N - 1)
        # Remove the west neighbor for the western boundary nodes
        east_west_diag[np.arange(1, self.ny) * self.nx - 1] = 0

        # Create the diagonals for the north and south neighbors
        north_south_diag = -1 * np.ones(self.N - self.nx)

        # Combine all the diagonals into a matrix
        K = np.diag(main_diag) + np.diag(east_west_diag, k=1) + np.diag(east_west_diag, k=-1) \
            + np.diag(north_south_diag, k=self.nx) + \
            np.diag(north_south_diag, k=-self.nx)

        self.K = modify_diagonal(K)

        return self.K

    def generate_load_vector(self):
        # Create a matrix of zeros
        F = np.zeros((self.ny, self.nx))

        # Set the values for the corner nodes (two neighbors)
        F[0, 0] = F[0, self.nx-1] = F[self.ny -
                                      1, 0] = F[self.ny-1, self.nx-1] = 1.0

        # Set the values for the edge nodes (three neighbors)
        F[1:self.ny-1, 0] = F[1:self.ny-1, self.nx-1] = F[0,
                                                          1:self.nx-1] = F[self.ny-1, 1:self.nx-1] = 2.0

        # Set the values for the internal nodes (four neighbors)
        F[1:self.ny-1, 1:self.nx-1] = 4.0

        # Flatten the matrix into a vector and return it
        self.F = F.ravel()

        return self.F

    def get_corners(self):
        """Returns an array with the indices of the four corners of the grid."""
        return np.array([0, self.nx - 1, self.nx * (self.ny - 1), self.nx * self.ny - 1])

    def get_boundaries(self):
        """Returns four arrays with the indices of the boundary nodes of the grid, excluding corners."""
        bottom = np.arange(1, self.nx - 1)
        top = np.arange(self.nx * (self.ny - 1) + 1, self.nx * self.ny - 1)
        left = np.arange(self.nx, self.nx * (self.ny - 1), self.nx)
        right = np.arange(2 * self.nx - 1, self.nx * self.ny - 1, self.nx)
        return left, right, bottom, top

    def get_cond_num(self):
        return np.linalg.cond(self.K)

    def solve_analytic(self):
        return np.linalg.solve(self.K, self.F)

    def solve(self, x0=None, tol=1e-6, max_iter=10000):
        # Initialize x0 if needed
        if x0 is None:
            x0 = np.zeros(self.F.shape)

        x = x0
        r = self.F - np.dot(self.K, x)
        p = r
        rsold = np.dot(np.transpose(r), r)

        # Compute condition number
        condition_number = np.linalg.cond(self.K)
        print(f"Condition number: {condition_number}")

        for i in range(max_iter):
            Ap = np.dot(self.K, p)
            alpha = rsold / np.dot(np.transpose(p), Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.dot(np.transpose(r), r)
            if np.sqrt(rsnew) < tol:
                print(f"Converged after {i+1} iterations.")
                return x
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x

    def plot_solution(self, mesh):
        u = np.ones(mesh.Ntot)*(-1)

        Ntot_x = mesh.Nsub_x * (self.nx - 1) + 1
        Ntot_y = mesh.Nsub_y * (self.ny - 1) + 1

        nodesD = np.arange(0, (mesh.Nsub_y + 1)*(self.ny - 1)
                           * Ntot_x, (self.ny - 1)*Ntot_x)

        nodesP = np.array([])

        firstNodesR = np.zeros(mesh.Nr)
        firstNodesR = mesh.rs

        botNodesR = np.array([])
        # np.zeros(len(mesh.bottom_r)*mesh.Nsub_x*mesh.Nsub_y)

        for j in range(mesh.Nsub_y + 1):
            nodesPj = np.arange(j*(self.ny - 1)*Ntot_x + self.nx - 1,
                                j*(self.ny - 1)*Ntot_x +
                                (self.nx - 1)*(mesh.Nsub_x + 1),
                                step=self.nx - 1)
            nodesP = np.concatenate((
                nodesP,
                nodesPj
            )).astype(int)
            botNodesRj = np.arange(j*(self.ny - 1)*Ntot_x + 1,
                                   j*(self.ny - 1)*Ntot_x + mesh.Nsub_x*(self.nx - 1) + 1)
            botNodesRj = np.setdiff1d(botNodesRj, nodesPj)
            botNodesR = np.concatenate((
                botNodesR,
                botNodesRj
            )).astype(int)

        bottom_bound_r = mesh.get_remaining_numeration(mesh.bottom_r)
        top_bound_r = mesh.get_remaining_numeration(mesh.top_r)

        bottomNodesUR = np.array([])
        for s in range(mesh.Nsub_x*mesh.Nsub_y):
            bottomNodesUR = np.concatenate(
                (bottomNodesUR, bottom_bound_r + s*mesh.Nr)).astype(int)

        for i in range(mesh.Nsub_x):
            offset = (mesh.Nsub_x*(mesh.Nsub_y - 1)) * mesh.Nr
            bottomNodesUR = np.concatenate((
                bottomNodesUR, top_bound_r + i*mesh.Nr + offset)).astype(int)

        u[nodesD] = mesh.uD
        u[nodesP] = mesh.uP
        u[botNodesR] = mesh.uR[bottomNodesUR]

        u_mat = u.reshape((Ntot_y, Ntot_x))
        plot_sparse_matrix(u_mat, 'Solution - u field')
