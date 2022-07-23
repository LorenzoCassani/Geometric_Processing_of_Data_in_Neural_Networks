# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt

# File path
PATH = "."


num_epochs = 200


t = np.linspace(1, 151, 151)

# Plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Plot: radius of gyration
plt.figure(figsize = (10, 6))
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('radius of gyration', fontsize='xx-large')
plt.grid()
plt.xlim(0, 150)
plt.ylim(0.6, 1.0)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/multiple_runs_average/expected_monotonicity.png', dpi=300)
plt.show()