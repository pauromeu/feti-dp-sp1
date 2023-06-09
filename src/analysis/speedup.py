# import numpy as np
# import matplotlib.pyplot as plt

# execution_times = [0.23173, 0.12999, 0.11024, 0.08501, 0.07586, 0.07129]
# nodes = [1, 2, 3, 4, 6, 8]

# # Calculate the speedup
# speedup = [execution_times[0] / t for t in execution_times]

# # Set the figure size
# plt.figure(figsize=(4, 4*2/3))

# # Plot the speedup
# plt.plot(nodes, speedup, marker='o', color='purple', label='Measured Speedup')

# # Calculate the ideal speedup based on Amdahl's Law
# ideal_speedup = nodes

# # Plot the ideal speedup line
# plt.plot(nodes, ideal_speedup, marker='', linestyle='--',
#          color='red', label='Ideal Speedup')

# # Set the x-axis scale to logarithmic base 2
# plt.xscale('log', basex=2)
# plt.yscale('log', basey=2)

# # Add labels and title
# plt.xlabel('# cores')
# plt.ylabel('Speedup')
# plt.legend()

# # Save the image as speedup.png
# plt.savefig('plots/speedup.eps', dpi=300)

# # Show the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_speedup(dimensions, times, legend_title, filename):
    # Calculate the speedup for each dimension
    speedup = []
    for i, dimension_times in enumerate(times):
        speedup_dimension = [times[i][0] / t for t in dimension_times]
        speedup.append(speedup_dimension)

    # Set the figure size
    plt.figure(figsize=(6*0.85, 4*0.85))

    # Plot the speedup for each dimension
    markers = ['o', 's', 'v', '^']
    colors = ['purple', 'blue', 'green', 'orange']
    nodes = [1, 2, 3, 4, 6, 8]
    for i, dimension in enumerate(dimensions):
        plt.plot(nodes, speedup[i], marker=markers[i],
                 color=colors[i], label=dimension)

    # Calculate the ideal speedup based on Amdahl's Law
    ideal_speedup = nodes

    # Plot the ideal speedup line
    plt.plot(nodes, ideal_speedup, marker='', linestyle='--',
             color='red', label='ideal')

    # Set the x-axis scale to logarithmic base 2
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)

    # Add labels and title
    plt.xlabel('# cores')
    plt.ylabel('Speedup')
    # plt.title('Speedup for Different Problem Dimensions')

    plt.ylim([0.95, 3.5])

    # Add legend with title
    plt.legend(title=legend_title, loc='upper left')

    # Save the image
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


# Example usage
dimensions_nx_ct = ['4x2', '4x4', '4x8', '4x16']
times_nx_ct = [
    [0.01081, 0.00715, 0.00757, 0.00649, 0.00863, 0.00907],
    [0.02434, 0.01505, 0.01508, 0.01247, 0.02033, 0.02205],
    [0.11249, 0.06793, 0.05178, 0.04718, 0.05540, 0.09570],
    [0.58985, 0.34502, 0.36526, 0.32296, 0.39609, 0.39992]
]

dimensions_Nsub_ct = ['4x4', '6x6', '8x8', '10x10']
times_Nsub_ct = [
    [0.01074, 0.00744, 0.00606, 0.00584, 0.00763, 0.00671],
    [0.05524, 0.03061, 0.02919, 0.02758, 0.04025, 0.03191],
    [0.16535, 0.10461, 0.09576, 0.09294, 0.13773, 0.14300],
    [0.31820, 0.25747, 0.23430, 0.20467, 0.23206, 0.24906]
]

plot_speedup(dimensions_nx_ct, times_nx_ct, '# subdomains',
             'plots/speedup_constant_nx.eps')
plot_speedup(dimensions_Nsub_ct, times_Nsub_ct,
             'subdomain size', 'plots/speedup_constant_Nsub.eps')
