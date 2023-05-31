import matplotlib.pyplot as plt
import numpy as np

nodes = [1, 2, 4, 8]
execution_times = [1.122380, 0.619830, 0.479830, 0.433440]

# Calculate the speedup
speedup = [execution_times[0] / t for t in execution_times]

# Create logarithmic base 2 x-axis values
log_nodes = np.log2(nodes)

# Plot the speedup
plt.plot(log_nodes, speedup, marker='o', label='Measured Speedup')

# Calculate the ideal speedup based on Amdahl's Law
ideal_speedup = np.log2(np.array([1, 2, 4, 8])) + 1

# Plot the ideal speedup line
plt.plot(log_nodes, ideal_speedup, marker='',
         linestyle='--', color='red', label='Ideal Speedup')

# Add labels and title
plt.xlabel('log2(Nodes)')
plt.ylabel('Speedup')
plt.title('Speedup vs. log2(Number of Nodes)')
plt.legend()

# Show the plot
plt.show()
