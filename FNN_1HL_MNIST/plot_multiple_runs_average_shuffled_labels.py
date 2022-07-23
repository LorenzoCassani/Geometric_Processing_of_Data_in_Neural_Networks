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


# Training error:

for i in range(91, 121):
    f_training_error = open("{}/Runs/run_{}/errors/training_err.dat".format(PATH, i), "r")
    globals()["training_error_{}".format(i - 90)] = eval(f_training_error.read())
    f_training_error.close()

training_error_mean = []
training_error_stdev = []

for i in range(num_epochs):
    
    training_error_mean.append(np.mean(
        [training_error_1[i],
         training_error_2[i],
         training_error_3[i],
         training_error_4[i],
         training_error_5[i],
         training_error_6[i],
         training_error_7[i],
         training_error_8[i],
         training_error_9[i],
         training_error_10[i],
         training_error_11[i],
         training_error_12[i],
         training_error_13[i],
         training_error_14[i],
         training_error_15[i],
         training_error_16[i],
         training_error_17[i],
         training_error_18[i],
         training_error_19[i],
         training_error_20[i],
         training_error_21[i],
         training_error_22[i],
         training_error_23[i],
         training_error_24[i],
         training_error_25[i],
         training_error_26[i],
         training_error_27[i],
         training_error_28[i],
         training_error_29[i],
         training_error_30[i]]
        )
    )
    
    training_error_stdev.append(np.std(
        [training_error_1[i],
         training_error_2[i],
         training_error_3[i],
         training_error_4[i],
         training_error_5[i],
         training_error_6[i],
         training_error_7[i],
         training_error_8[i],
         training_error_9[i],
         training_error_10[i],
         training_error_11[i],
         training_error_12[i],
         training_error_13[i],
         training_error_14[i],
         training_error_15[i],
         training_error_16[i],
         training_error_17[i],
         training_error_18[i],
         training_error_19[i],
         training_error_20[i],
         training_error_21[i],
         training_error_22[i],
         training_error_23[i],
         training_error_24[i],
         training_error_25[i],
         training_error_26[i],
         training_error_27[i],
         training_error_28[i],
         training_error_29[i],
         training_error_30[i]]
        )
    )

training_error_mean = np.array(training_error_mean)
training_error_stdev = np.array(training_error_stdev)

training_error_lower = training_error_mean - training_error_stdev
training_error_upper = training_error_mean + training_error_stdev


# Test error:

for i in range(91, 121):
    f_test_error = open("{}/Runs/run_{}/errors/test_err.dat".format(PATH, i), "r")
    globals()["test_error_{}".format(i - 90)] = eval(f_test_error.read())
    f_test_error.close()

test_error_mean = []
test_error_stdev = []

for i in range(num_epochs):
    
    test_error_mean.append(np.mean(
        [test_error_1[i],
         test_error_2[i],
         test_error_3[i],
         test_error_4[i],
         test_error_5[i],
         test_error_6[i],
         test_error_7[i],
         test_error_8[i],
         test_error_9[i],
         test_error_10[i],
         test_error_11[i],
         test_error_12[i],
         test_error_13[i],
         test_error_14[i],
         test_error_15[i],
         test_error_16[i],
         test_error_17[i],
         test_error_18[i],
         test_error_19[i],
         test_error_20[i],
         test_error_21[i],
         test_error_22[i],
         test_error_23[i],
         test_error_24[i],
         test_error_25[i],
         test_error_26[i],
         test_error_27[i],
         test_error_28[i],
         test_error_29[i],
         test_error_30[i]]
        )
    )
    
    test_error_stdev.append(np.std(
        [test_error_1[i],
         test_error_2[i],
         test_error_3[i],
         test_error_4[i],
         test_error_5[i],
         test_error_6[i],
         test_error_7[i],
         test_error_8[i],
         test_error_9[i],
         test_error_10[i],
         test_error_11[i],
         test_error_12[i],
         test_error_13[i],
         test_error_14[i],
         test_error_15[i],
         test_error_16[i],
         test_error_17[i],
         test_error_18[i],
         test_error_19[i],
         test_error_20[i],
         test_error_21[i],
         test_error_22[i],
         test_error_23[i],
         test_error_24[i],
         test_error_25[i],
         test_error_26[i],
         test_error_27[i],
         test_error_28[i],
         test_error_29[i],
         test_error_30[i]]
        )
    )

test_error_mean = np.array(test_error_mean)
test_error_stdev = np.array(test_error_stdev)

test_error_lower = test_error_mean - test_error_stdev
test_error_upper = test_error_mean + test_error_stdev


# Centre-centre distance:

for i in range(91, 121):
    f_distance = open("{}/Runs/run_{}/cc_distance/cc_distance.dat".format(PATH, i), "r")
    globals()["distance_{}".format(i - 90)] = eval(f_distance.read())
    f_distance.close()

distance_mean = []
distance_stdev = []

for i in range(num_epochs):
    
    distance_mean.append(np.mean(
        [distance_1[i],
         distance_2[i],
         distance_3[i],
         distance_4[i],
         distance_5[i],
         distance_6[i],
         distance_7[i],
         distance_8[i],
         distance_9[i],
         distance_10[i],
         distance_11[i],
         distance_12[i],
         distance_13[i],
         distance_14[i],
         distance_15[i],
         distance_16[i],
         distance_17[i],
         distance_18[i],
         distance_19[i],
         distance_20[i],
         distance_21[i],
         distance_22[i],
         distance_23[i],
         distance_24[i],
         distance_25[i],
         distance_26[i],
         distance_27[i],
         distance_28[i],
         distance_29[i],
         distance_30[i]]
        )
    )
    
    distance_stdev.append(np.std(
        [distance_1[i],
         distance_2[i],
         distance_3[i],
         distance_4[i],
         distance_5[i],
         distance_6[i],
         distance_7[i],
         distance_8[i],
         distance_9[i],
         distance_10[i],
         distance_11[i],
         distance_12[i],
         distance_13[i],
         distance_14[i],
         distance_15[i],
         distance_16[i],
         distance_17[i],
         distance_18[i],
         distance_19[i],
         distance_20[i],
         distance_21[i],
         distance_22[i],
         distance_23[i],
         distance_24[i],
         distance_25[i],
         distance_26[i],
         distance_27[i],
         distance_28[i],
         distance_29[i],
         distance_30[i]]
        )
    )

distance_mean = np.array(distance_mean)
distance_stdev = np.array(distance_stdev)

distance_lower = distance_mean - distance_stdev
distance_upper = distance_mean + distance_stdev


# Radius of gyration:

for i in range(91, 121):
    f_gyradius = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["gyradius_{}".format(i - 90)] = eval(f_gyradius.read())
    f_gyradius.close()

gyradius_mean = []
gyradius_stdev = []

for i in range(num_epochs):
    
    gyradius_mean.append(np.mean(
        [gyradius_1[i],
         gyradius_2[i],
         gyradius_3[i],
         gyradius_4[i],
         gyradius_5[i],
         gyradius_6[i],
         gyradius_7[i],
         gyradius_8[i],
         gyradius_9[i],
         gyradius_10[i],
         gyradius_11[i],
         gyradius_12[i],
         gyradius_13[i],
         gyradius_14[i],
         gyradius_15[i],
         gyradius_16[i],
         gyradius_17[i],
         gyradius_18[i],
         gyradius_19[i],
         gyradius_20[i],
         gyradius_21[i],
         gyradius_22[i],
         gyradius_23[i],
         gyradius_24[i],
         gyradius_25[i],
         gyradius_26[i],
         gyradius_27[i],
         gyradius_28[i],
         gyradius_29[i],
         gyradius_30[i]]
        )
    )
    
    gyradius_stdev.append(np.std(
        [gyradius_1[i],
         gyradius_2[i],
         gyradius_3[i],
         gyradius_4[i],
         gyradius_5[i],
         gyradius_6[i],
         gyradius_7[i],
         gyradius_8[i],
         gyradius_9[i],
         gyradius_10[i],
         gyradius_11[i],
         gyradius_12[i],
         gyradius_13[i],
         gyradius_14[i],
         gyradius_15[i],
         gyradius_16[i],
         gyradius_17[i],
         gyradius_18[i],
         gyradius_19[i],
         gyradius_20[i],
         gyradius_21[i],
         gyradius_22[i],
         gyradius_23[i],
         gyradius_24[i],
         gyradius_25[i],
         gyradius_26[i],
         gyradius_27[i],
         gyradius_28[i],
         gyradius_29[i],
         gyradius_30[i]]
        )
    )

gyradius_mean = np.array(gyradius_mean)
gyradius_stdev = np.array(gyradius_stdev)

gyradius_lower = gyradius_mean - gyradius_stdev
gyradius_upper = gyradius_mean + gyradius_stdev


# Rescaled radius of gyration:

for i in range(91, 121):
    f_rescaled_gyradius = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["rescaled_gyradius_{}".format(i - 90)] = eval(f_rescaled_gyradius.read())
    f_rescaled_gyradius.close()

rescaled_gyradius_mean = []
rescaled_gyradius_stdev = []

for i in range(num_epochs):
    
    rescaled_gyradius_mean.append(np.mean(
        [rescaled_gyradius_1[i],
         rescaled_gyradius_2[i],
         rescaled_gyradius_3[i],
         rescaled_gyradius_4[i],
         rescaled_gyradius_5[i],
         rescaled_gyradius_6[i],
         rescaled_gyradius_7[i],
         rescaled_gyradius_8[i],
         rescaled_gyradius_9[i],
         rescaled_gyradius_10[i],
         rescaled_gyradius_11[i],
         rescaled_gyradius_12[i],
         rescaled_gyradius_13[i],
         rescaled_gyradius_14[i],
         rescaled_gyradius_15[i],
         rescaled_gyradius_16[i],
         rescaled_gyradius_17[i],
         rescaled_gyradius_18[i],
         rescaled_gyradius_19[i],
         rescaled_gyradius_20[i],
         rescaled_gyradius_21[i],
         rescaled_gyradius_22[i],
         rescaled_gyradius_23[i],
         rescaled_gyradius_24[i],
         rescaled_gyradius_25[i],
         rescaled_gyradius_26[i],
         rescaled_gyradius_27[i],
         rescaled_gyradius_28[i],
         rescaled_gyradius_29[i],
         rescaled_gyradius_30[i]]
        )
    )
    
    rescaled_gyradius_stdev.append(np.std(
        [rescaled_gyradius_1[i],
         rescaled_gyradius_2[i],
         rescaled_gyradius_3[i],
         rescaled_gyradius_4[i],
         rescaled_gyradius_5[i],
         rescaled_gyradius_6[i],
         rescaled_gyradius_7[i],
         rescaled_gyradius_8[i],
         rescaled_gyradius_9[i],
         rescaled_gyradius_10[i],
         rescaled_gyradius_11[i],
         rescaled_gyradius_12[i],
         rescaled_gyradius_13[i],
         rescaled_gyradius_14[i],
         rescaled_gyradius_15[i],
         rescaled_gyradius_16[i],
         rescaled_gyradius_17[i],
         rescaled_gyradius_18[i],
         rescaled_gyradius_19[i],
         rescaled_gyradius_20[i],
         rescaled_gyradius_21[i],
         rescaled_gyradius_22[i],
         rescaled_gyradius_23[i],
         rescaled_gyradius_24[i],
         rescaled_gyradius_25[i],
         rescaled_gyradius_26[i],
         rescaled_gyradius_27[i],
         rescaled_gyradius_28[i],
         rescaled_gyradius_29[i],
         rescaled_gyradius_30[i]]
        )
    )

rescaled_gyradius_mean = np.array(rescaled_gyradius_mean)
rescaled_gyradius_stdev = np.array(rescaled_gyradius_stdev)

rescaled_gyradius_lower = rescaled_gyradius_mean - rescaled_gyradius_stdev
rescaled_gyradius_upper = rescaled_gyradius_mean + rescaled_gyradius_stdev


# Plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Plot: training error & test error
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), training_error_mean, label='training')
plt.plot(np.arange(1, num_epochs + 1, 1), test_error_mean, label='test')
plt.fill_between(np.arange(1, num_epochs + 1, 1), training_error_lower, training_error_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), test_error_lower, test_error_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 20))
plt.yticks(np.arange(0.3, 1, 0.05))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('error', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='xx-large', loc='center right')
plt.xlim(0, 200)
plt.ylim(0.3, 0.55)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/shuffled_labels/errors_shuffled_labels.png', dpi=300)
plt.show()

# Plot: radius of gyration
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), gyradius_mean)
plt.fill_between(np.arange(1, num_epochs + 1, 1), gyradius_lower, gyradius_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 20))
plt.yticks(np.arange(0.985, 1.1, 0.005))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('radius of gyration', fontsize='xx-large')
plt.grid()
plt.xlim(0, 200)
plt.ylim(0.985, 1.0)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/shuffled_labels/radius_of_gyration_shuffled_labels.png', dpi=300)
plt.show()

# Plot: centre-to-centre distance
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), distance_mean)
plt.fill_between(np.arange(1, num_epochs + 1, 1), distance_lower, distance_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 20))
plt.yticks(np.arange(0, 0.2, 0.04))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('centre-to-centre distance', fontsize='xx-large')
plt.grid()
plt.xlim(0, 200)
plt.ylim(0, 0.16)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/shuffled_labels/ctc_distance_shuffled_labels.png', dpi=300)
plt.show()

# Plot: radius of gyration rescaled by centre-centre distance
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_mean)
plt.fill_between(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_lower, rescaled_gyradius_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 20))
plt.yticks(np.arange(6, 30, 4))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('dimensionless radius of gyration', fontsize='xx-large')
plt.grid()
plt.xlim(0, 200)
plt.ylim(6, 22)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/shuffled_labels/dimensionless_radius_of_gyration_shuffled_labels.png', dpi=300)
plt.show()