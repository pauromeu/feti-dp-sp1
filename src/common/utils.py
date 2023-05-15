import matplotlib.pyplot as plt


def plot_sparse_matrix(A, title=None):
    """Plot a heat map of the matrix A to visualize the non-zeros position

    Args:
        A (numpy.darray): 2D matrix
    """
    plt.imshow(A)
    plt.colorbar()
    if (title):
        plt.title(title)
    plt.show()
