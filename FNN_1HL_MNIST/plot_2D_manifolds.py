# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt

# File path
PATH = "."

# Choose run whose manifolds will be plotted
run = 222

num_epochs = 150

manifolds_0 = []
manifolds_1 = []
# 2D Manifolds at each training epoch:
for epoch in range (1, num_epochs+1):
    f_manifold_0 = open("{}/Runs/run_{}/data/manifold_0_epoch_{}.dat".format(PATH, run, epoch), "r")
    f_manifold_1 = open("{}/Runs/run_{}/data/manifold_1_epoch_{}.dat".format(PATH, run, epoch), "r")
    manifold_0 = eval(f_manifold_0.read())
    manifold_1 = eval(f_manifold_1.read())
    f_manifold_0.close()
    f_manifold_1.close()
    manifolds_0.append(manifold_0)
    manifolds_1.append(manifold_1)


# Plots
plt.style.use('seaborn-dark')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')


# Plot: 2D manifolds at each training epoch
for epoch in range (0, num_epochs):
    even_x = []
    even_y = []
    odd_x = []
    odd_y = []
    for i in range (len(manifolds_0[0])):
        even_x.append(manifolds_0[epoch][i][0])
        even_y.append(manifolds_0[epoch][i][1])
    for i in range (len(manifolds_1[0])):
        odd_x.append(manifolds_1[epoch][i][0])
        odd_y.append(manifolds_1[epoch][i][1])
    plt.figure(figsize = (10, 10))
    plt.scatter(even_x, even_y, label='"even" manifold')
    plt.scatter(odd_x, odd_y, label='"odd" manifold')
    plt.xticks(np.arange(-1, 1.2, 0.2))
    plt.yticks(np.arange(-1, 1.2, 0.2))
    plt.xlabel('First coordinate', fontsize='xx-large')
    plt.ylabel('Second coordinate', fontsize='xx-large')
    plt.grid()
    plt.legend(fontsize='xx-large', loc='upper center')
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.35)
    plt.show()