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


distance_maxima_indices = []
gyradius_minima_indices = []
rescaled_gyradius_minima_indices = []


# Centre-centre distance:
for i in range(1, 31):
    f_distance = open("{}/Runs/run_{}/cc_distance/cc_distance.dat".format(PATH, i), "r")
    globals()["distance_{}".format(i)] = eval(f_distance.read())
    distance_maxima_indices.append(globals()["distance_{}".format(i)].index(max(globals()["distance_{}".format(i)])))
    f_distance.close()

# Radius of gyration:
for i in range(1, 31):
    f_gyradius = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["gyradius_{}".format(i)] = eval(f_gyradius.read())
    gyradius_minima_indices.append(globals()["gyradius_{}".format(i)].index(min(globals()["gyradius_{}".format(i)])))
    f_gyradius.close()

# Rescaled radius of gyration:
for i in range(1, 31):
    f_rescaled_gyradius = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["rescaled_gyradius_{}".format(i)] = eval(f_rescaled_gyradius.read())
    rescaled_gyradius_minima_indices.append(globals()["rescaled_gyradius_{}".format(i)].index(min(globals()["rescaled_gyradius_{}".format(i)])))
    f_rescaled_gyradius.close()


# Plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Plot: radius of gyration & centre-to-centre distance
plt.figure(figsize = (10, 6))
for i in range(1, 31):
    plt.plot(np.arange(1, num_epochs + 1, 1), globals()["gyradius_{}".format(i)])
    plt.plot(np.arange(1, num_epochs + 1, 1), globals()["distance_{}".format(i)])
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(0.6, 1.7, 0.2))
plt.axvspan(min(min(distance_maxima_indices), min(gyradius_minima_indices)), max(max(distance_maxima_indices), max(gyradius_minima_indices)), alpha=0.1, color='green')
plt.text(30.5, 0.9, 'inversion band', rotation='vertical', fontsize='xx-large', color='green')
plt.text(82, 1.42, 'centre-to-centre distances', fontsize='xx-large')
plt.text(94, 0.67, 'radii of gyration', fontsize='xx-large')
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('geometric observable', fontsize='xx-large')
plt.grid()
plt.xlim(0, 150)
plt.ylim(0.6, 1.6)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/multiple_runs_comparison/radius_of_gyration_and_ctc_distance.png', dpi=300)
plt.show()