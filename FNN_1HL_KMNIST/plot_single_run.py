# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt

# File path
PATH = "."

# Choose run whose observables will be plotted
run = 0

num_epochs = 200


# Training error:
f_training_error = open("{}/Runs/run_{}/errors/training_err.dat".format(PATH, run), "r")
training_error = eval(f_training_error.read())
f_training_error.close()

# Test error:
f_test_error = open("{}/Runs/run_{}/errors/test_err.dat".format(PATH, run), "r")
test_error = eval(f_test_error.read())
f_test_error.close()

# Centre-centre distance:
f_distance = open("{}/Runs/run_{}/cc_distance/cc_distance.dat".format(PATH, run), "r")
distance = eval(f_distance.read())
f_distance.close()

# Radii of gyration:
f_gyradius_0 = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(PATH, run), "r")
f_gyradius_1 = open("{}/Runs/run_{}/gyradius/manifold_1.dat".format(PATH, run), "r")
gyradius_0 = eval(f_gyradius_0.read())
gyradius_1 = eval(f_gyradius_1.read())
f_gyradius_0.close()
f_gyradius_1.close()

# Rescaled radii of gyration:
f_rescaled_gyradius_0 = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(PATH, run), "r")
f_rescaled_gyradius_1 = open("{}/Runs/run_{}/rescaled_gyradius/manifold_1.dat".format(PATH, run), "r")
rescaled_gyradius_0 = eval(f_rescaled_gyradius_0.read())
rescaled_gyradius_1 = eval(f_rescaled_gyradius_1.read())
f_rescaled_gyradius_0.close()
f_rescaled_gyradius_1.close()


# Plots
plt.style.use('seaborn-dark')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Plot: training error & test error
plt.figure(figsize = (16, 9))
plt.plot(np.arange(1, num_epochs + 1, 1), training_error, label='training')
plt.plot(np.arange(1, num_epochs + 1, 1), test_error, label='test')
plt.xticks(np.arange(0, num_epochs + 1, 10))
plt.yticks(np.arange(0, 0.51, 0.05))
plt.xlabel('epoch', fontsize='xx-large')
plt.ylabel('error', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='xx-large', loc='center right')
plt.xlim(0, 150)
plt.ylim(0, 0.50)
plt.show()

# Plot: centre-centre distance
plt.figure(figsize = (16, 9))
plt.plot(np.arange(1, num_epochs + 1, 1), distance)
plt.xticks(np.arange(0, num_epochs + 1, 10))
plt.yticks(np.arange(0.5, 1.4, 0.1))
plt.xlabel('epoch', fontsize='xx-large')
plt.ylabel('centre-centre distance', fontsize='xx-large')
plt.grid()
plt.xlim(0, 150)
plt.ylim(0.5, 1.3)
plt.show()

# Plot: radius of gyration
plt.figure(figsize = (16, 9))
plt.plot(np.arange(1, num_epochs + 1, 1), gyradius_0, label='"even" manifold')
plt.plot(np.arange(1, num_epochs + 1, 1), gyradius_1, label='"odd" manifold')
plt.xticks(np.arange(0, num_epochs + 1, 10))
plt.yticks(np.arange(0.8, 1.1, 0.1))
plt.xlabel('epoch', fontsize='xx-large')
plt.ylabel('radius of gyration', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='xx-large', loc='upper right')
plt.xlim(0, 150)
plt.ylim(0.8, 1.0)
plt.show()

# Plot: radius of gyration rescaled by centre-centre distance
plt.figure(figsize = (16, 9))
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_0, label='"even" manifold')
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_1, label='"odd" manifold')
plt.xticks(np.arange(0, num_epochs + 1, 10))
plt.yticks(np.arange(0.7, 1.5, 0.1))
plt.xlabel('epoch', fontsize='xx-large')
plt.ylabel('radius of gyration / centre-centre distance', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='xx-large', loc='upper right')
plt.xlim(0, 200)
plt.ylim(0.7, 1.5)
plt.show()