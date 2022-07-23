# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt

# File path
PATH = "."


num_epochs = 1000


# Training error:

for i in range(1, 31):
    f_training_error = open("{}/Runs/run_{}/errors/training_err.dat".format(PATH, i), "r")
    globals()["training_error_{}".format(i)] = eval(f_training_error.read())
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

for i in range(1, 31):
    f_test_error = open("{}/Runs/run_{}/errors/test_err.dat".format(PATH, i), "r")
    globals()["test_error_{}".format(i)] = eval(f_test_error.read())
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


# Radius of gyration (even manifold):

for i in range(1, 31):
    f_gyradius_even = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["gyradius_even_{}".format(i)] = eval(f_gyradius_even.read())
    f_gyradius_even.close()

gyradius_even_mean = []
gyradius_even_stdev = []

for i in range(num_epochs):
    
    gyradius_even_mean.append(np.mean(
        [gyradius_even_1[i],
         gyradius_even_2[i],
         gyradius_even_3[i],
         gyradius_even_4[i],
         gyradius_even_5[i],
         gyradius_even_6[i],
         gyradius_even_7[i],
         gyradius_even_8[i],
         gyradius_even_9[i],
         gyradius_even_10[i],
         gyradius_even_11[i],
         gyradius_even_12[i],
         gyradius_even_13[i],
         gyradius_even_14[i],
         gyradius_even_15[i],
         gyradius_even_16[i],
         gyradius_even_17[i],
         gyradius_even_18[i],
         gyradius_even_19[i],
         gyradius_even_20[i],
         gyradius_even_21[i],
         gyradius_even_22[i],
         gyradius_even_23[i],
         gyradius_even_24[i],
         gyradius_even_25[i],
         gyradius_even_26[i],
         gyradius_even_27[i],
         gyradius_even_28[i],
         gyradius_even_29[i],
         gyradius_even_30[i]]
        )
    )
    
    gyradius_even_stdev.append(np.std(
        [gyradius_even_1[i],
         gyradius_even_2[i],
         gyradius_even_3[i],
         gyradius_even_4[i],
         gyradius_even_5[i],
         gyradius_even_6[i],
         gyradius_even_7[i],
         gyradius_even_8[i],
         gyradius_even_9[i],
         gyradius_even_10[i],
         gyradius_even_11[i],
         gyradius_even_12[i],
         gyradius_even_13[i],
         gyradius_even_14[i],
         gyradius_even_15[i],
         gyradius_even_16[i],
         gyradius_even_17[i],
         gyradius_even_18[i],
         gyradius_even_19[i],
         gyradius_even_20[i],
         gyradius_even_21[i],
         gyradius_even_22[i],
         gyradius_even_23[i],
         gyradius_even_24[i],
         gyradius_even_25[i],
         gyradius_even_26[i],
         gyradius_even_27[i],
         gyradius_even_28[i],
         gyradius_even_29[i],
         gyradius_even_30[i]]
        )
    )

gyradius_even_mean = np.array(gyradius_even_mean)
gyradius_even_stdev = np.array(gyradius_even_stdev)

gyradius_even_lower = gyradius_even_mean - gyradius_even_stdev
gyradius_even_upper = gyradius_even_mean + gyradius_even_stdev


# Radius of gyration (odd manifold):

for i in range(1, 31):
    f_gyradius_odd = open("{}/Runs/run_{}/gyradius/manifold_1.dat".format(PATH, i), "r")
    globals()["gyradius_odd_{}".format(i)] = eval(f_gyradius_odd.read())
    f_gyradius_odd.close()

gyradius_odd_mean = []
gyradius_odd_stdev = []

for i in range(num_epochs):
    
    gyradius_odd_mean.append(np.mean(
        [gyradius_odd_1[i],
         gyradius_odd_2[i],
         gyradius_odd_3[i],
         gyradius_odd_4[i],
         gyradius_odd_5[i],
         gyradius_odd_6[i],
         gyradius_odd_7[i],
         gyradius_odd_8[i],
         gyradius_odd_9[i],
         gyradius_odd_10[i],
         gyradius_odd_11[i],
         gyradius_odd_12[i],
         gyradius_odd_13[i],
         gyradius_odd_14[i],
         gyradius_odd_15[i],
         gyradius_odd_16[i],
         gyradius_odd_17[i],
         gyradius_odd_18[i],
         gyradius_odd_19[i],
         gyradius_odd_20[i],
         gyradius_odd_21[i],
         gyradius_odd_22[i],
         gyradius_odd_23[i],
         gyradius_odd_24[i],
         gyradius_odd_25[i],
         gyradius_odd_26[i],
         gyradius_odd_27[i],
         gyradius_odd_28[i],
         gyradius_odd_29[i],
         gyradius_odd_30[i]]
        )
    )
    
    gyradius_odd_stdev.append(np.std(
        [gyradius_odd_1[i],
         gyradius_odd_2[i],
         gyradius_odd_3[i],
         gyradius_odd_4[i],
         gyradius_odd_5[i],
         gyradius_odd_6[i],
         gyradius_odd_7[i],
         gyradius_odd_8[i],
         gyradius_odd_9[i],
         gyradius_odd_10[i],
         gyradius_odd_11[i],
         gyradius_odd_12[i],
         gyradius_odd_13[i],
         gyradius_odd_14[i],
         gyradius_odd_15[i],
         gyradius_odd_16[i],
         gyradius_odd_17[i],
         gyradius_odd_18[i],
         gyradius_odd_19[i],
         gyradius_odd_20[i],
         gyradius_odd_21[i],
         gyradius_odd_22[i],
         gyradius_odd_23[i],
         gyradius_odd_24[i],
         gyradius_odd_25[i],
         gyradius_odd_26[i],
         gyradius_odd_27[i],
         gyradius_odd_28[i],
         gyradius_odd_29[i],
         gyradius_odd_30[i]]
        )
    )

gyradius_odd_mean = np.array(gyradius_odd_mean)
gyradius_odd_stdev = np.array(gyradius_odd_stdev)

gyradius_odd_lower = gyradius_odd_mean - gyradius_odd_stdev
gyradius_odd_upper = gyradius_odd_mean + gyradius_odd_stdev


# Centre-to-centre distance:

for i in range(1, 31):
    f_distance = open("{}/Runs/run_{}/cc_distance/cc_distance.dat".format(PATH, i), "r")
    globals()["distance_{}".format(i)] = eval(f_distance.read())
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


# dimensionless radius of gyration (even manifold):

for i in range(1, 31):
    f_rescaled_gyradius_even = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["rescaled_gyradius_even_{}".format(i)] = eval(f_rescaled_gyradius_even.read())
    f_rescaled_gyradius_even.close()

rescaled_gyradius_even_mean = []
rescaled_gyradius_even_stdev = []

for i in range(num_epochs):
    
    rescaled_gyradius_even_mean.append(np.mean(
        [rescaled_gyradius_even_1[i],
         rescaled_gyradius_even_2[i],
         rescaled_gyradius_even_3[i],
         rescaled_gyradius_even_4[i],
         rescaled_gyradius_even_5[i],
         rescaled_gyradius_even_6[i],
         rescaled_gyradius_even_7[i],
         rescaled_gyradius_even_8[i],
         rescaled_gyradius_even_9[i],
         rescaled_gyradius_even_10[i],
         rescaled_gyradius_even_11[i],
         rescaled_gyradius_even_12[i],
         rescaled_gyradius_even_13[i],
         rescaled_gyradius_even_14[i],
         rescaled_gyradius_even_15[i],
         rescaled_gyradius_even_16[i],
         rescaled_gyradius_even_17[i],
         rescaled_gyradius_even_18[i],
         rescaled_gyradius_even_19[i],
         rescaled_gyradius_even_20[i],
         rescaled_gyradius_even_21[i],
         rescaled_gyradius_even_22[i],
         rescaled_gyradius_even_23[i],
         rescaled_gyradius_even_24[i],
         rescaled_gyradius_even_25[i],
         rescaled_gyradius_even_26[i],
         rescaled_gyradius_even_27[i],
         rescaled_gyradius_even_28[i],
         rescaled_gyradius_even_29[i],
         rescaled_gyradius_even_30[i]]
        )
    )
    
    rescaled_gyradius_even_stdev.append(np.std(
        [rescaled_gyradius_even_1[i],
         rescaled_gyradius_even_2[i],
         rescaled_gyradius_even_3[i],
         rescaled_gyradius_even_4[i],
         rescaled_gyradius_even_5[i],
         rescaled_gyradius_even_6[i],
         rescaled_gyradius_even_7[i],
         rescaled_gyradius_even_8[i],
         rescaled_gyradius_even_9[i],
         rescaled_gyradius_even_10[i],
         rescaled_gyradius_even_11[i],
         rescaled_gyradius_even_12[i],
         rescaled_gyradius_even_13[i],
         rescaled_gyradius_even_14[i],
         rescaled_gyradius_even_15[i],
         rescaled_gyradius_even_16[i],
         rescaled_gyradius_even_17[i],
         rescaled_gyradius_even_18[i],
         rescaled_gyradius_even_19[i],
         rescaled_gyradius_even_20[i],
         rescaled_gyradius_even_21[i],
         rescaled_gyradius_even_22[i],
         rescaled_gyradius_even_23[i],
         rescaled_gyradius_even_24[i],
         rescaled_gyradius_even_25[i],
         rescaled_gyradius_even_26[i],
         rescaled_gyradius_even_27[i],
         rescaled_gyradius_even_28[i],
         rescaled_gyradius_even_29[i],
         rescaled_gyradius_even_30[i]]
        )
    )

rescaled_gyradius_even_mean = np.array(rescaled_gyradius_even_mean)
rescaled_gyradius_even_stdev = np.array(rescaled_gyradius_even_stdev)

rescaled_gyradius_even_lower = rescaled_gyradius_even_mean - rescaled_gyradius_even_stdev
rescaled_gyradius_even_upper = rescaled_gyradius_even_mean + rescaled_gyradius_even_stdev


# dimensionless radius of gyration (odd manifold):

for i in range(1, 31):
    f_rescaled_gyradius_odd = open("{}/Runs/run_{}/rescaled_gyradius/manifold_1.dat".format(PATH, i), "r")
    globals()["rescaled_gyradius_odd_{}".format(i)] = eval(f_rescaled_gyradius_odd.read())
    f_rescaled_gyradius_odd.close()

rescaled_gyradius_odd_mean = []
rescaled_gyradius_odd_stdev = []

for i in range(num_epochs):
    
    rescaled_gyradius_odd_mean.append(np.mean(
        [rescaled_gyradius_odd_1[i],
         rescaled_gyradius_odd_2[i],
         rescaled_gyradius_odd_3[i],
         rescaled_gyradius_odd_4[i],
         rescaled_gyradius_odd_5[i],
         rescaled_gyradius_odd_6[i],
         rescaled_gyradius_odd_7[i],
         rescaled_gyradius_odd_8[i],
         rescaled_gyradius_odd_9[i],
         rescaled_gyradius_odd_10[i],
         rescaled_gyradius_odd_11[i],
         rescaled_gyradius_odd_12[i],
         rescaled_gyradius_odd_13[i],
         rescaled_gyradius_odd_14[i],
         rescaled_gyradius_odd_15[i],
         rescaled_gyradius_odd_16[i],
         rescaled_gyradius_odd_17[i],
         rescaled_gyradius_odd_18[i],
         rescaled_gyradius_odd_19[i],
         rescaled_gyradius_odd_20[i],
         rescaled_gyradius_odd_21[i],
         rescaled_gyradius_odd_22[i],
         rescaled_gyradius_odd_23[i],
         rescaled_gyradius_odd_24[i],
         rescaled_gyradius_odd_25[i],
         rescaled_gyradius_odd_26[i],
         rescaled_gyradius_odd_27[i],
         rescaled_gyradius_odd_28[i],
         rescaled_gyradius_odd_29[i],
         rescaled_gyradius_odd_30[i]]
        )
    )
    
    rescaled_gyradius_odd_stdev.append(np.std(
        [rescaled_gyradius_odd_1[i],
         rescaled_gyradius_odd_2[i],
         rescaled_gyradius_odd_3[i],
         rescaled_gyradius_odd_4[i],
         rescaled_gyradius_odd_5[i],
         rescaled_gyradius_odd_6[i],
         rescaled_gyradius_odd_7[i],
         rescaled_gyradius_odd_8[i],
         rescaled_gyradius_odd_9[i],
         rescaled_gyradius_odd_10[i],
         rescaled_gyradius_odd_11[i],
         rescaled_gyradius_odd_12[i],
         rescaled_gyradius_odd_13[i],
         rescaled_gyradius_odd_14[i],
         rescaled_gyradius_odd_15[i],
         rescaled_gyradius_odd_16[i],
         rescaled_gyradius_odd_17[i],
         rescaled_gyradius_odd_18[i],
         rescaled_gyradius_odd_19[i],
         rescaled_gyradius_odd_20[i],
         rescaled_gyradius_odd_21[i],
         rescaled_gyradius_odd_22[i],
         rescaled_gyradius_odd_23[i],
         rescaled_gyradius_odd_24[i],
         rescaled_gyradius_odd_25[i],
         rescaled_gyradius_odd_26[i],
         rescaled_gyradius_odd_27[i],
         rescaled_gyradius_odd_28[i],
         rescaled_gyradius_odd_29[i],
         rescaled_gyradius_odd_30[i]]
        )
    )

rescaled_gyradius_odd_mean = np.array(rescaled_gyradius_odd_mean)
rescaled_gyradius_odd_stdev = np.array(rescaled_gyradius_odd_stdev)

rescaled_gyradius_odd_lower = rescaled_gyradius_odd_mean - rescaled_gyradius_odd_stdev
rescaled_gyradius_odd_upper = rescaled_gyradius_odd_mean + rescaled_gyradius_odd_stdev


# Plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Plot: dimensionless radius of gyration
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_even_mean, label = '"even" manifold')
plt.fill_between(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_even_lower, rescaled_gyradius_even_upper, alpha=0.2)
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_odd_mean, label = '"odd" manifold')
plt.fill_between(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_odd_lower, rescaled_gyradius_odd_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 100))
plt.yticks(np.arange(0, 1, 0.03))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('dimensionless radius of gyration', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='xx-large', loc='lower right')
plt.xlim(0, 1000)
plt.ylim(0.21, 0.42)
plt.savefig('../../Desktop/Tesi/Figures/Fashion-MNIST/dimensionless_radius_of_gyration.png', dpi=300)
plt.show()