# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt

# File path
PATH = "."


num_epochs = 150
test_error_delta_to_flatten = []
test_error_delta = []


# Test errors at epoch 200:
for i in range(1151, 1301):
    f_test_error = open("{}/Runs/run_{}/errors/test_err.dat".format(PATH, i), "r")
    globals()["test_error_{}".format(i)] = eval(f_test_error.read())
    f_test_error.close()
    test_error_delta_to_flatten.append(globals()["test_error_{}".format(i)][149])

# Test errors array flattening
for elem in test_error_delta_to_flatten:
    for item in elem:
        test_error_delta.append(item)


# Plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Plot: test error
plt.figure(figsize = (10, 6))
plt.scatter(np.arange(1, num_epochs + 1, 1), test_error_delta)
plt.xticks(np.arange(0, num_epochs + 1, 10))
plt.yticks(np.arange(0, 0.35, 0.05))
plt.xlabel('epoch *', fontsize='xx-large')
plt.ylabel('test error at the end of training', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='xx-large', loc='upper right')
plt.xlim(0, 155)
plt.ylim(0, 0.25)
plt.show()