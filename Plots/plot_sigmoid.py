# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

plt.figure(figsize = (10, 6))
plt.axhline(y=0, color='k', alpha=0.5)
plt.axvline(x=0, color='k', alpha=0.5)
plt.axhline(y=1, color='k', alpha=0.5, linestyle='dashed')
plt.plot(z, sigmoid(z), label=r'σ(z)')
plt.xticks(np.arange(-10, 10, 1))
plt.yticks(np.arange(-10, 10, 0.5))
plt.xlabel('z', fontsize='xx-large')
plt.ylabel(r'σ(z)', fontsize='xx-large')
plt.grid()
plt.xlim(-5, 5)
plt.ylim(-0.1, 1.1)
plt.savefig('../../Desktop/Tesi/Figures/sigmoid.png', dpi=300)
plt.show()