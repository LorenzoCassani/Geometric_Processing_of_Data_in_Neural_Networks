# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt

yHat = np.linspace(0.00001, 1, 1000)

def CrossEntropy(yHat):
    return -np.log(yHat)

# Plot
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

plt.figure(figsize = (10, 6))
plt.axhline(y=0, color='k', alpha=0.5)
plt.axvline(x=0, color='k', alpha=0.5)
plt.plot(yHat, CrossEntropy(yHat))
plt.xticks(np.arange(-1, 2, 0.2))
plt.yticks(np.arange(0, 20, 2))
plt.xlabel('predicted probability for the true label', fontsize='xx-large')
plt.ylabel('cross-entropy loss', fontsize='xx-large')
plt.grid()
plt.xlim(-0.1, 1.1)
plt.ylim(-1, 10)
plt.savefig('../../Desktop/Tesi/Figures/crossEntropy.png', dpi=300)
plt.show()