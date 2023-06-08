import numpy as np
import matplotlib.pyplot as plt

execution_times = [0.23173, 0.12999, 0.11024, 0.08501, 0.07586, 0.07129]
nodes = [1, 2, 3, 4, 6, 8]

# Calculate the speedup
speedup = [execution_times[0] / t for t in execution_times]

# Set the figure size
plt.figure(figsize=(4, 4*2/3))

# Plot the speedup
plt.plot(nodes, speedup, marker='o', color='purple', label='Measured Speedup')

# Calculate the ideal speedup based on Amdahl's Law
ideal_speedup = nodes

# Plot the ideal speedup line
plt.plot(nodes, ideal_speedup, marker='', linestyle='--',
         color='red', label='Ideal Speedup')

# Set the x-axis scale to logarithmic base 2
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)

# Add labels and title
plt.xlabel('# cores')
plt.ylabel('Speedup')
plt.legend()

# Save the image as speedup.png
plt.savefig('plots/speedup.eps', dpi=300)

# Show the plot
plt.show()
