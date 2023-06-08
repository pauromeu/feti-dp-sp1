import numpy as np
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


def plot_with_dual_yaxis(x, y1, y2, xlabel, label1, label2, save_fig=False, filename=None):
    # Create figure and axis objects
    figsize = (4, 4*2/3)

    fig, ax1 = plt.subplots(figsize=figsize)

    color1 = "green"
    marker1 = 's'
    color2 = "purple"
    marker2 = 'o'

    # Plot the first function with the specified color
    ax1.plot(x, y1, color1, label=label1, marker=marker1)
    ax1.set_xlabel(xlabel)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.set_ylabel(label1, color=color1)
    ax1.tick_params('y', colors=color1)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot the second function with the specified color
    ax2.plot(x, y2, color2, label=label2, marker=marker2)
    ax2.set_ylabel(label2, color=color2)
    ax2.tick_params('y', colors=color2)

    # Set x-tick locations at powers of 2
    x_exponent = np.arange(np.floor(np.log2(min(x))),
                           np.ceil(np.log2(max(x))) + 1)
    x_ticks = np.power(2, x_exponent)
    ax1.set_xticks(x_ticks)

    # Set x-tick labels as powers of 2
    xtick_labels = [f"$2^{int(exponent)}$" for exponent in x_exponent]
    ax1.set_xticklabels(xtick_labels)

    # Add legends
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    if save_fig:
        fig.savefig(filename + '.eps', format='eps',
                    dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()
