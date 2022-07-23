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


# Training errors:

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


for i in range(31, 61):
    f_training_error = open("{}/Runs/run_{}/errors/training_err.dat".format(PATH, i), "r")
    globals()["training_error_without_stragglers_{}".format(i - 30)] = eval(f_training_error.read())
    f_training_error.close()

training_error_without_stragglers_mean = []
training_error_without_stragglers_stdev = []

for i in range(num_epochs):
    
    training_error_without_stragglers_mean.append(np.mean(
        [training_error_without_stragglers_1[i],
         training_error_without_stragglers_2[i],
         training_error_without_stragglers_3[i],
         training_error_without_stragglers_4[i],
         training_error_without_stragglers_5[i],
         training_error_without_stragglers_6[i],
         training_error_without_stragglers_7[i],
         training_error_without_stragglers_8[i],
         training_error_without_stragglers_9[i],
         training_error_without_stragglers_10[i],
         training_error_without_stragglers_11[i],
         training_error_without_stragglers_12[i],
         training_error_without_stragglers_13[i],
         training_error_without_stragglers_14[i],
         training_error_without_stragglers_15[i],
         training_error_without_stragglers_16[i],
         training_error_without_stragglers_17[i],
         training_error_without_stragglers_18[i],
         training_error_without_stragglers_19[i],
         training_error_without_stragglers_20[i],
         training_error_without_stragglers_21[i],
         training_error_without_stragglers_22[i],
         training_error_without_stragglers_23[i],
         training_error_without_stragglers_24[i],
         training_error_without_stragglers_25[i],
         training_error_without_stragglers_26[i],
         training_error_without_stragglers_27[i],
         training_error_without_stragglers_28[i],
         training_error_without_stragglers_29[i],
         training_error_without_stragglers_30[i]]
        )
    )
    
    training_error_without_stragglers_stdev.append(np.std(
        [training_error_without_stragglers_1[i],
         training_error_without_stragglers_2[i],
         training_error_without_stragglers_3[i],
         training_error_without_stragglers_4[i],
         training_error_without_stragglers_5[i],
         training_error_without_stragglers_6[i],
         training_error_without_stragglers_7[i],
         training_error_without_stragglers_8[i],
         training_error_without_stragglers_9[i],
         training_error_without_stragglers_10[i],
         training_error_without_stragglers_11[i],
         training_error_without_stragglers_12[i],
         training_error_without_stragglers_13[i],
         training_error_without_stragglers_14[i],
         training_error_without_stragglers_15[i],
         training_error_without_stragglers_16[i],
         training_error_without_stragglers_17[i],
         training_error_without_stragglers_18[i],
         training_error_without_stragglers_19[i],
         training_error_without_stragglers_20[i],
         training_error_without_stragglers_21[i],
         training_error_without_stragglers_22[i],
         training_error_without_stragglers_23[i],
         training_error_without_stragglers_24[i],
         training_error_without_stragglers_25[i],
         training_error_without_stragglers_26[i],
         training_error_without_stragglers_27[i],
         training_error_without_stragglers_28[i],
         training_error_without_stragglers_29[i],
         training_error_without_stragglers_30[i]]
        )
    )

training_error_without_stragglers_mean = np.array(training_error_without_stragglers_mean)
training_error_without_stragglers_stdev = np.array(training_error_without_stragglers_stdev)

training_error_without_stragglers_lower = training_error_without_stragglers_mean - training_error_without_stragglers_stdev
training_error_without_stragglers_upper = training_error_without_stragglers_mean + training_error_without_stragglers_stdev


for i in range(61, 91):
    f_training_error = open("{}/Runs/run_{}/errors/training_err.dat".format(PATH, i), "r")
    globals()["training_error_without_random_data_{}".format(i - 60)] = eval(f_training_error.read())
    f_training_error.close()

training_error_without_random_data_mean = []
training_error_without_random_data_stdev = []

for i in range(num_epochs):
    
    training_error_without_random_data_mean.append(np.mean(
        [training_error_without_random_data_1[i],
         training_error_without_random_data_2[i],
         training_error_without_random_data_3[i],
         training_error_without_random_data_4[i],
         training_error_without_random_data_5[i],
         training_error_without_random_data_6[i],
         training_error_without_random_data_7[i],
         training_error_without_random_data_8[i],
         training_error_without_random_data_9[i],
         training_error_without_random_data_10[i],
         training_error_without_random_data_11[i],
         training_error_without_random_data_12[i],
         training_error_without_random_data_13[i],
         training_error_without_random_data_14[i],
         training_error_without_random_data_15[i],
         training_error_without_random_data_16[i],
         training_error_without_random_data_17[i],
         training_error_without_random_data_18[i],
         training_error_without_random_data_19[i],
         training_error_without_random_data_20[i],
         training_error_without_random_data_21[i],
         training_error_without_random_data_22[i],
         training_error_without_random_data_23[i],
         training_error_without_random_data_24[i],
         training_error_without_random_data_25[i],
         training_error_without_random_data_26[i],
         training_error_without_random_data_27[i],
         training_error_without_random_data_28[i],
         training_error_without_random_data_29[i],
         training_error_without_random_data_30[i]]
        )
    )
    
    training_error_without_random_data_stdev.append(np.std(
        [training_error_without_random_data_1[i],
         training_error_without_random_data_2[i],
         training_error_without_random_data_3[i],
         training_error_without_random_data_4[i],
         training_error_without_random_data_5[i],
         training_error_without_random_data_6[i],
         training_error_without_random_data_7[i],
         training_error_without_random_data_8[i],
         training_error_without_random_data_9[i],
         training_error_without_random_data_10[i],
         training_error_without_random_data_11[i],
         training_error_without_random_data_12[i],
         training_error_without_random_data_13[i],
         training_error_without_random_data_14[i],
         training_error_without_random_data_15[i],
         training_error_without_random_data_16[i],
         training_error_without_random_data_17[i],
         training_error_without_random_data_18[i],
         training_error_without_random_data_19[i],
         training_error_without_random_data_20[i],
         training_error_without_random_data_21[i],
         training_error_without_random_data_22[i],
         training_error_without_random_data_23[i],
         training_error_without_random_data_24[i],
         training_error_without_random_data_25[i],
         training_error_without_random_data_26[i],
         training_error_without_random_data_27[i],
         training_error_without_random_data_28[i],
         training_error_without_random_data_29[i],
         training_error_without_random_data_30[i]]
        )
    )

training_error_without_random_data_mean = np.array(training_error_without_random_data_mean)
training_error_without_random_data_stdev = np.array(training_error_without_random_data_stdev)

training_error_without_random_data_lower = training_error_without_random_data_mean - training_error_without_random_data_stdev
training_error_without_random_data_upper = training_error_without_random_data_mean + training_error_without_random_data_stdev


# Test errors:

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


for i in range(31, 61):
    f_test_error = open("{}/Runs/run_{}/errors/test_err.dat".format(PATH, i), "r")
    globals()["test_error_without_stragglers_{}".format(i - 30)] = eval(f_test_error.read())
    f_test_error.close()

test_error_without_stragglers_mean = []
test_error_without_stragglers_stdev = []

for i in range(num_epochs):
    
    test_error_without_stragglers_mean.append(np.mean(
        [test_error_without_stragglers_1[i],
         test_error_without_stragglers_2[i],
         test_error_without_stragglers_3[i],
         test_error_without_stragglers_4[i],
         test_error_without_stragglers_5[i],
         test_error_without_stragglers_6[i],
         test_error_without_stragglers_7[i],
         test_error_without_stragglers_8[i],
         test_error_without_stragglers_9[i],
         test_error_without_stragglers_10[i],
         test_error_without_stragglers_11[i],
         test_error_without_stragglers_12[i],
         test_error_without_stragglers_13[i],
         test_error_without_stragglers_14[i],
         test_error_without_stragglers_15[i],
         test_error_without_stragglers_16[i],
         test_error_without_stragglers_17[i],
         test_error_without_stragglers_18[i],
         test_error_without_stragglers_19[i],
         test_error_without_stragglers_20[i],
         test_error_without_stragglers_21[i],
         test_error_without_stragglers_22[i],
         test_error_without_stragglers_23[i],
         test_error_without_stragglers_24[i],
         test_error_without_stragglers_25[i],
         test_error_without_stragglers_26[i],
         test_error_without_stragglers_27[i],
         test_error_without_stragglers_28[i],
         test_error_without_stragglers_29[i],
         test_error_without_stragglers_30[i]]
        )
    )
    
    test_error_without_stragglers_stdev.append(np.std(
        [test_error_without_stragglers_1[i],
         test_error_without_stragglers_2[i],
         test_error_without_stragglers_3[i],
         test_error_without_stragglers_4[i],
         test_error_without_stragglers_5[i],
         test_error_without_stragglers_6[i],
         test_error_without_stragglers_7[i],
         test_error_without_stragglers_8[i],
         test_error_without_stragglers_9[i],
         test_error_without_stragglers_10[i],
         test_error_without_stragglers_11[i],
         test_error_without_stragglers_12[i],
         test_error_without_stragglers_13[i],
         test_error_without_stragglers_14[i],
         test_error_without_stragglers_15[i],
         test_error_without_stragglers_16[i],
         test_error_without_stragglers_17[i],
         test_error_without_stragglers_18[i],
         test_error_without_stragglers_19[i],
         test_error_without_stragglers_20[i],
         test_error_without_stragglers_21[i],
         test_error_without_stragglers_22[i],
         test_error_without_stragglers_23[i],
         test_error_without_stragglers_24[i],
         test_error_without_stragglers_25[i],
         test_error_without_stragglers_26[i],
         test_error_without_stragglers_27[i],
         test_error_without_stragglers_28[i],
         test_error_without_stragglers_29[i],
         test_error_without_stragglers_30[i]]
        )
    )

test_error_without_stragglers_mean = np.array(test_error_without_stragglers_mean)
test_error_without_stragglers_stdev = np.array(test_error_without_stragglers_stdev)

test_error_without_stragglers_lower = test_error_without_stragglers_mean - test_error_without_stragglers_stdev
test_error_without_stragglers_upper = test_error_without_stragglers_mean + test_error_without_stragglers_stdev


for i in range(61, 91):
    f_test_error = open("{}/Runs/run_{}/errors/test_err.dat".format(PATH, i), "r")
    globals()["test_error_without_random_data_{}".format(i - 60)] = eval(f_test_error.read())
    f_test_error.close()

test_error_without_random_data_mean = []
test_error_without_random_data_stdev = []

for i in range(num_epochs):
    
    test_error_without_random_data_mean.append(np.mean(
        [test_error_without_random_data_1[i],
         test_error_without_random_data_2[i],
         test_error_without_random_data_3[i],
         test_error_without_random_data_4[i],
         test_error_without_random_data_5[i],
         test_error_without_random_data_6[i],
         test_error_without_random_data_7[i],
         test_error_without_random_data_8[i],
         test_error_without_random_data_9[i],
         test_error_without_random_data_10[i],
         test_error_without_random_data_11[i],
         test_error_without_random_data_12[i],
         test_error_without_random_data_13[i],
         test_error_without_random_data_14[i],
         test_error_without_random_data_15[i],
         test_error_without_random_data_16[i],
         test_error_without_random_data_17[i],
         test_error_without_random_data_18[i],
         test_error_without_random_data_19[i],
         test_error_without_random_data_20[i],
         test_error_without_random_data_21[i],
         test_error_without_random_data_22[i],
         test_error_without_random_data_23[i],
         test_error_without_random_data_24[i],
         test_error_without_random_data_25[i],
         test_error_without_random_data_26[i],
         test_error_without_random_data_27[i],
         test_error_without_random_data_28[i],
         test_error_without_random_data_29[i],
         test_error_without_random_data_30[i]]
        )
    )
    
    test_error_without_random_data_stdev.append(np.std(
        [test_error_without_random_data_1[i],
         test_error_without_random_data_2[i],
         test_error_without_random_data_3[i],
         test_error_without_random_data_4[i],
         test_error_without_random_data_5[i],
         test_error_without_random_data_6[i],
         test_error_without_random_data_7[i],
         test_error_without_random_data_8[i],
         test_error_without_random_data_9[i],
         test_error_without_random_data_10[i],
         test_error_without_random_data_11[i],
         test_error_without_random_data_12[i],
         test_error_without_random_data_13[i],
         test_error_without_random_data_14[i],
         test_error_without_random_data_15[i],
         test_error_without_random_data_16[i],
         test_error_without_random_data_17[i],
         test_error_without_random_data_18[i],
         test_error_without_random_data_19[i],
         test_error_without_random_data_20[i],
         test_error_without_random_data_21[i],
         test_error_without_random_data_22[i],
         test_error_without_random_data_23[i],
         test_error_without_random_data_24[i],
         test_error_without_random_data_25[i],
         test_error_without_random_data_26[i],
         test_error_without_random_data_27[i],
         test_error_without_random_data_28[i],
         test_error_without_random_data_29[i],
         test_error_without_random_data_30[i]]
        )
    )

test_error_without_random_data_mean = np.array(test_error_without_random_data_mean)
test_error_without_random_data_stdev = np.array(test_error_without_random_data_stdev)

test_error_without_random_data_lower = test_error_without_random_data_mean - test_error_without_random_data_stdev
test_error_without_random_data_upper = test_error_without_random_data_mean + test_error_without_random_data_stdev


# Centre-centre distances:

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


for i in range(31, 61):
    f_distance = open("{}/Runs/run_{}/cc_distance/cc_distance.dat".format(PATH, i), "r")
    globals()["distance_without_stragglers_{}".format(i - 30)] = eval(f_distance.read())
    f_distance.close()

distance_without_stragglers_mean = []
distance_without_stragglers_stdev = []

for i in range(num_epochs):
    
    distance_without_stragglers_mean.append(np.mean(
        [distance_without_stragglers_1[i],
         distance_without_stragglers_2[i],
         distance_without_stragglers_3[i],
         distance_without_stragglers_4[i],
         distance_without_stragglers_5[i],
         distance_without_stragglers_6[i],
         distance_without_stragglers_7[i],
         distance_without_stragglers_8[i],
         distance_without_stragglers_9[i],
         distance_without_stragglers_10[i],
         distance_without_stragglers_11[i],
         distance_without_stragglers_12[i],
         distance_without_stragglers_13[i],
         distance_without_stragglers_14[i],
         distance_without_stragglers_15[i],
         distance_without_stragglers_16[i],
         distance_without_stragglers_17[i],
         distance_without_stragglers_18[i],
         distance_without_stragglers_19[i],
         distance_without_stragglers_20[i],
         distance_without_stragglers_21[i],
         distance_without_stragglers_22[i],
         distance_without_stragglers_23[i],
         distance_without_stragglers_24[i],
         distance_without_stragglers_25[i],
         distance_without_stragglers_26[i],
         distance_without_stragglers_27[i],
         distance_without_stragglers_28[i],
         distance_without_stragglers_29[i],
         distance_without_stragglers_30[i]]
        )
    )
    
    distance_without_stragglers_stdev.append(np.std(
        [distance_without_stragglers_1[i],
         distance_without_stragglers_2[i],
         distance_without_stragglers_3[i],
         distance_without_stragglers_4[i],
         distance_without_stragglers_5[i],
         distance_without_stragglers_6[i],
         distance_without_stragglers_7[i],
         distance_without_stragglers_8[i],
         distance_without_stragglers_9[i],
         distance_without_stragglers_10[i],
         distance_without_stragglers_11[i],
         distance_without_stragglers_12[i],
         distance_without_stragglers_13[i],
         distance_without_stragglers_14[i],
         distance_without_stragglers_15[i],
         distance_without_stragglers_16[i],
         distance_without_stragglers_17[i],
         distance_without_stragglers_18[i],
         distance_without_stragglers_19[i],
         distance_without_stragglers_20[i],
         distance_without_stragglers_21[i],
         distance_without_stragglers_22[i],
         distance_without_stragglers_23[i],
         distance_without_stragglers_24[i],
         distance_without_stragglers_25[i],
         distance_without_stragglers_26[i],
         distance_without_stragglers_27[i],
         distance_without_stragglers_28[i],
         distance_without_stragglers_29[i],
         distance_without_stragglers_30[i]]
        )
    )

distance_without_stragglers_mean = np.array(distance_without_stragglers_mean)
distance_without_stragglers_stdev = np.array(distance_without_stragglers_stdev)

distance_without_stragglers_lower = distance_without_stragglers_mean - distance_without_stragglers_stdev
distance_without_stragglers_upper = distance_without_stragglers_mean + distance_without_stragglers_stdev


for i in range(61, 91):
    f_distance = open("{}/Runs/run_{}/cc_distance/cc_distance.dat".format(PATH, i), "r")
    globals()["distance_without_random_data_{}".format(i - 60)] = eval(f_distance.read())
    f_distance.close()

distance_without_random_data_mean = []
distance_without_random_data_stdev = []

for i in range(num_epochs):
    
    distance_without_random_data_mean.append(np.mean(
        [distance_without_random_data_1[i],
         distance_without_random_data_2[i],
         distance_without_random_data_3[i],
         distance_without_random_data_4[i],
         distance_without_random_data_5[i],
         distance_without_random_data_6[i],
         distance_without_random_data_7[i],
         distance_without_random_data_8[i],
         distance_without_random_data_9[i],
         distance_without_random_data_10[i],
         distance_without_random_data_11[i],
         distance_without_random_data_12[i],
         distance_without_random_data_13[i],
         distance_without_random_data_14[i],
         distance_without_random_data_15[i],
         distance_without_random_data_16[i],
         distance_without_random_data_17[i],
         distance_without_random_data_18[i],
         distance_without_random_data_19[i],
         distance_without_random_data_20[i],
         distance_without_random_data_21[i],
         distance_without_random_data_22[i],
         distance_without_random_data_23[i],
         distance_without_random_data_24[i],
         distance_without_random_data_25[i],
         distance_without_random_data_26[i],
         distance_without_random_data_27[i],
         distance_without_random_data_28[i],
         distance_without_random_data_29[i],
         distance_without_random_data_30[i]]
        )
    )
    
    distance_without_random_data_stdev.append(np.std(
        [distance_without_random_data_1[i],
         distance_without_random_data_2[i],
         distance_without_random_data_3[i],
         distance_without_random_data_4[i],
         distance_without_random_data_5[i],
         distance_without_random_data_6[i],
         distance_without_random_data_7[i],
         distance_without_random_data_8[i],
         distance_without_random_data_9[i],
         distance_without_random_data_10[i],
         distance_without_random_data_11[i],
         distance_without_random_data_12[i],
         distance_without_random_data_13[i],
         distance_without_random_data_14[i],
         distance_without_random_data_15[i],
         distance_without_random_data_16[i],
         distance_without_random_data_17[i],
         distance_without_random_data_18[i],
         distance_without_random_data_19[i],
         distance_without_random_data_20[i],
         distance_without_random_data_21[i],
         distance_without_random_data_22[i],
         distance_without_random_data_23[i],
         distance_without_random_data_24[i],
         distance_without_random_data_25[i],
         distance_without_random_data_26[i],
         distance_without_random_data_27[i],
         distance_without_random_data_28[i],
         distance_without_random_data_29[i],
         distance_without_random_data_30[i]]
        )
    )

distance_without_random_data_mean = np.array(distance_without_random_data_mean)
distance_without_random_data_stdev = np.array(distance_without_random_data_stdev)

distance_without_random_data_lower = distance_without_random_data_mean - distance_without_random_data_stdev
distance_without_random_data_upper = distance_without_random_data_mean + distance_without_random_data_stdev


# Radii of gyration:

for i in range(1, 31):
    f_gyradius = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["gyradius_{}".format(i)] = eval(f_gyradius.read())
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


for i in range(31, 61):
    f_gyradius = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["gyradius_without_stragglers_{}".format(i - 30)] = eval(f_gyradius.read())
    f_gyradius.close()

gyradius_without_stragglers_mean = []
gyradius_without_stragglers_stdev = []

for i in range(num_epochs):
    
    gyradius_without_stragglers_mean.append(np.mean(
        [gyradius_without_stragglers_1[i],
         gyradius_without_stragglers_2[i],
         gyradius_without_stragglers_3[i],
         gyradius_without_stragglers_4[i],
         gyradius_without_stragglers_5[i],
         gyradius_without_stragglers_6[i],
         gyradius_without_stragglers_7[i],
         gyradius_without_stragglers_8[i],
         gyradius_without_stragglers_9[i],
         gyradius_without_stragglers_10[i],
         gyradius_without_stragglers_11[i],
         gyradius_without_stragglers_12[i],
         gyradius_without_stragglers_13[i],
         gyradius_without_stragglers_14[i],
         gyradius_without_stragglers_15[i],
         gyradius_without_stragglers_16[i],
         gyradius_without_stragglers_17[i],
         gyradius_without_stragglers_18[i],
         gyradius_without_stragglers_19[i],
         gyradius_without_stragglers_20[i],
         gyradius_without_stragglers_21[i],
         gyradius_without_stragglers_22[i],
         gyradius_without_stragglers_23[i],
         gyradius_without_stragglers_24[i],
         gyradius_without_stragglers_25[i],
         gyradius_without_stragglers_26[i],
         gyradius_without_stragglers_27[i],
         gyradius_without_stragglers_28[i],
         gyradius_without_stragglers_29[i],
         gyradius_without_stragglers_30[i]]
        )
    )
    
    gyradius_without_stragglers_stdev.append(np.std(
        [gyradius_without_stragglers_1[i],
         gyradius_without_stragglers_2[i],
         gyradius_without_stragglers_3[i],
         gyradius_without_stragglers_4[i],
         gyradius_without_stragglers_5[i],
         gyradius_without_stragglers_6[i],
         gyradius_without_stragglers_7[i],
         gyradius_without_stragglers_8[i],
         gyradius_without_stragglers_9[i],
         gyradius_without_stragglers_10[i],
         gyradius_without_stragglers_11[i],
         gyradius_without_stragglers_12[i],
         gyradius_without_stragglers_13[i],
         gyradius_without_stragglers_14[i],
         gyradius_without_stragglers_15[i],
         gyradius_without_stragglers_16[i],
         gyradius_without_stragglers_17[i],
         gyradius_without_stragglers_18[i],
         gyradius_without_stragglers_19[i],
         gyradius_without_stragglers_20[i],
         gyradius_without_stragglers_21[i],
         gyradius_without_stragglers_22[i],
         gyradius_without_stragglers_23[i],
         gyradius_without_stragglers_24[i],
         gyradius_without_stragglers_25[i],
         gyradius_without_stragglers_26[i],
         gyradius_without_stragglers_27[i],
         gyradius_without_stragglers_28[i],
         gyradius_without_stragglers_29[i],
         gyradius_without_stragglers_30[i]]
        )
    )

gyradius_without_stragglers_mean = np.array(gyradius_without_stragglers_mean)
gyradius_without_stragglers_stdev = np.array(gyradius_without_stragglers_stdev)

gyradius_without_stragglers_lower = gyradius_without_stragglers_mean - gyradius_without_stragglers_stdev
gyradius_without_stragglers_upper = gyradius_without_stragglers_mean + gyradius_without_stragglers_stdev


for i in range(61, 91):
    f_gyradius = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["gyradius_without_random_data_{}".format(i - 60)] = eval(f_gyradius.read())
    f_gyradius.close()

gyradius_without_random_data_mean = []
gyradius_without_random_data_stdev = []

for i in range(num_epochs):
    
    gyradius_without_random_data_mean.append(np.mean(
        [gyradius_without_random_data_1[i],
         gyradius_without_random_data_2[i],
         gyradius_without_random_data_3[i],
         gyradius_without_random_data_4[i],
         gyradius_without_random_data_5[i],
         gyradius_without_random_data_6[i],
         gyradius_without_random_data_7[i],
         gyradius_without_random_data_8[i],
         gyradius_without_random_data_9[i],
         gyradius_without_random_data_10[i],
         gyradius_without_random_data_11[i],
         gyradius_without_random_data_12[i],
         gyradius_without_random_data_13[i],
         gyradius_without_random_data_14[i],
         gyradius_without_random_data_15[i],
         gyradius_without_random_data_16[i],
         gyradius_without_random_data_17[i],
         gyradius_without_random_data_18[i],
         gyradius_without_random_data_19[i],
         gyradius_without_random_data_20[i],
         gyradius_without_random_data_21[i],
         gyradius_without_random_data_22[i],
         gyradius_without_random_data_23[i],
         gyradius_without_random_data_24[i],
         gyradius_without_random_data_25[i],
         gyradius_without_random_data_26[i],
         gyradius_without_random_data_27[i],
         gyradius_without_random_data_28[i],
         gyradius_without_random_data_29[i],
         gyradius_without_random_data_30[i]]
        )
    )
    
    gyradius_without_random_data_stdev.append(np.std(
        [gyradius_without_random_data_1[i],
         gyradius_without_random_data_2[i],
         gyradius_without_random_data_3[i],
         gyradius_without_random_data_4[i],
         gyradius_without_random_data_5[i],
         gyradius_without_random_data_6[i],
         gyradius_without_random_data_7[i],
         gyradius_without_random_data_8[i],
         gyradius_without_random_data_9[i],
         gyradius_without_random_data_10[i],
         gyradius_without_random_data_11[i],
         gyradius_without_random_data_12[i],
         gyradius_without_random_data_13[i],
         gyradius_without_random_data_14[i],
         gyradius_without_random_data_15[i],
         gyradius_without_random_data_16[i],
         gyradius_without_random_data_17[i],
         gyradius_without_random_data_18[i],
         gyradius_without_random_data_19[i],
         gyradius_without_random_data_20[i],
         gyradius_without_random_data_21[i],
         gyradius_without_random_data_22[i],
         gyradius_without_random_data_23[i],
         gyradius_without_random_data_24[i],
         gyradius_without_random_data_25[i],
         gyradius_without_random_data_26[i],
         gyradius_without_random_data_27[i],
         gyradius_without_random_data_28[i],
         gyradius_without_random_data_29[i],
         gyradius_without_random_data_30[i]]
        )
    )

gyradius_without_random_data_mean = np.array(gyradius_without_random_data_mean)
gyradius_without_random_data_stdev = np.array(gyradius_without_random_data_stdev)

gyradius_without_random_data_lower = gyradius_without_random_data_mean - gyradius_without_random_data_stdev
gyradius_without_random_data_upper = gyradius_without_random_data_mean + gyradius_without_random_data_stdev


# Rescaled radii of gyration:

for i in range(1, 31):
    f_rescaled_gyradius = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["rescaled_gyradius_{}".format(i)] = eval(f_rescaled_gyradius.read())
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


for i in range(31, 61):
    f_rescaled_gyradius = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["rescaled_gyradius_without_stragglers_{}".format(i - 30)] = eval(f_rescaled_gyradius.read())
    f_rescaled_gyradius.close()

rescaled_gyradius_without_stragglers_mean = []
rescaled_gyradius_without_stragglers_stdev = []

for i in range(num_epochs):
    
    rescaled_gyradius_without_stragglers_mean.append(np.mean(
        [rescaled_gyradius_without_stragglers_1[i],
         rescaled_gyradius_without_stragglers_2[i],
         rescaled_gyradius_without_stragglers_3[i],
         rescaled_gyradius_without_stragglers_4[i],
         rescaled_gyradius_without_stragglers_5[i],
         rescaled_gyradius_without_stragglers_6[i],
         rescaled_gyradius_without_stragglers_7[i],
         rescaled_gyradius_without_stragglers_8[i],
         rescaled_gyradius_without_stragglers_9[i],
         rescaled_gyradius_without_stragglers_10[i],
         rescaled_gyradius_without_stragglers_11[i],
         rescaled_gyradius_without_stragglers_12[i],
         rescaled_gyradius_without_stragglers_13[i],
         rescaled_gyradius_without_stragglers_14[i],
         rescaled_gyradius_without_stragglers_15[i],
         rescaled_gyradius_without_stragglers_16[i],
         rescaled_gyradius_without_stragglers_17[i],
         rescaled_gyradius_without_stragglers_18[i],
         rescaled_gyradius_without_stragglers_19[i],
         rescaled_gyradius_without_stragglers_20[i],
         rescaled_gyradius_without_stragglers_21[i],
         rescaled_gyradius_without_stragglers_22[i],
         rescaled_gyradius_without_stragglers_23[i],
         rescaled_gyradius_without_stragglers_24[i],
         rescaled_gyradius_without_stragglers_25[i],
         rescaled_gyradius_without_stragglers_26[i],
         rescaled_gyradius_without_stragglers_27[i],
         rescaled_gyradius_without_stragglers_28[i],
         rescaled_gyradius_without_stragglers_29[i],
         rescaled_gyradius_without_stragglers_30[i]]
        )
    )
    
    rescaled_gyradius_without_stragglers_stdev.append(np.std(
        [rescaled_gyradius_without_stragglers_1[i],
         rescaled_gyradius_without_stragglers_2[i],
         rescaled_gyradius_without_stragglers_3[i],
         rescaled_gyradius_without_stragglers_4[i],
         rescaled_gyradius_without_stragglers_5[i],
         rescaled_gyradius_without_stragglers_6[i],
         rescaled_gyradius_without_stragglers_7[i],
         rescaled_gyradius_without_stragglers_8[i],
         rescaled_gyradius_without_stragglers_9[i],
         rescaled_gyradius_without_stragglers_10[i],
         rescaled_gyradius_without_stragglers_11[i],
         rescaled_gyradius_without_stragglers_12[i],
         rescaled_gyradius_without_stragglers_13[i],
         rescaled_gyradius_without_stragglers_14[i],
         rescaled_gyradius_without_stragglers_15[i],
         rescaled_gyradius_without_stragglers_16[i],
         rescaled_gyradius_without_stragglers_17[i],
         rescaled_gyradius_without_stragglers_18[i],
         rescaled_gyradius_without_stragglers_19[i],
         rescaled_gyradius_without_stragglers_20[i],
         rescaled_gyradius_without_stragglers_21[i],
         rescaled_gyradius_without_stragglers_22[i],
         rescaled_gyradius_without_stragglers_23[i],
         rescaled_gyradius_without_stragglers_24[i],
         rescaled_gyradius_without_stragglers_25[i],
         rescaled_gyradius_without_stragglers_26[i],
         rescaled_gyradius_without_stragglers_27[i],
         rescaled_gyradius_without_stragglers_28[i],
         rescaled_gyradius_without_stragglers_29[i],
         rescaled_gyradius_without_stragglers_30[i]]
        )
    )

rescaled_gyradius_without_stragglers_mean = np.array(rescaled_gyradius_without_stragglers_mean)
rescaled_gyradius_without_stragglers_stdev = np.array(rescaled_gyradius_without_stragglers_stdev)

rescaled_gyradius_without_stragglers_lower = rescaled_gyradius_without_stragglers_mean - rescaled_gyradius_without_stragglers_stdev
rescaled_gyradius_without_stragglers_upper = rescaled_gyradius_without_stragglers_mean + rescaled_gyradius_without_stragglers_stdev


for i in range(61, 91):
    f_rescaled_gyradius = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(PATH, i), "r")
    globals()["rescaled_gyradius_without_random_data_{}".format(i - 60)] = eval(f_rescaled_gyradius.read())
    f_rescaled_gyradius.close()

rescaled_gyradius_without_random_data_mean = []
rescaled_gyradius_without_random_data_stdev = []

for i in range(num_epochs):
    
    rescaled_gyradius_without_random_data_mean.append(np.mean(
        [rescaled_gyradius_without_random_data_1[i],
         rescaled_gyradius_without_random_data_2[i],
         rescaled_gyradius_without_random_data_3[i],
         rescaled_gyradius_without_random_data_4[i],
         rescaled_gyradius_without_random_data_5[i],
         rescaled_gyradius_without_random_data_6[i],
         rescaled_gyradius_without_random_data_7[i],
         rescaled_gyradius_without_random_data_8[i],
         rescaled_gyradius_without_random_data_9[i],
         rescaled_gyradius_without_random_data_10[i],
         rescaled_gyradius_without_random_data_11[i],
         rescaled_gyradius_without_random_data_12[i],
         rescaled_gyradius_without_random_data_13[i],
         rescaled_gyradius_without_random_data_14[i],
         rescaled_gyradius_without_random_data_15[i],
         rescaled_gyradius_without_random_data_16[i],
         rescaled_gyradius_without_random_data_17[i],
         rescaled_gyradius_without_random_data_18[i],
         rescaled_gyradius_without_random_data_19[i],
         rescaled_gyradius_without_random_data_20[i],
         rescaled_gyradius_without_random_data_21[i],
         rescaled_gyradius_without_random_data_22[i],
         rescaled_gyradius_without_random_data_23[i],
         rescaled_gyradius_without_random_data_24[i],
         rescaled_gyradius_without_random_data_25[i],
         rescaled_gyradius_without_random_data_26[i],
         rescaled_gyradius_without_random_data_27[i],
         rescaled_gyradius_without_random_data_28[i],
         rescaled_gyradius_without_random_data_29[i],
         rescaled_gyradius_without_random_data_30[i]]
        )
    )
    
    rescaled_gyradius_without_random_data_stdev.append(np.std(
        [rescaled_gyradius_without_random_data_1[i],
         rescaled_gyradius_without_random_data_2[i],
         rescaled_gyradius_without_random_data_3[i],
         rescaled_gyradius_without_random_data_4[i],
         rescaled_gyradius_without_random_data_5[i],
         rescaled_gyradius_without_random_data_6[i],
         rescaled_gyradius_without_random_data_7[i],
         rescaled_gyradius_without_random_data_8[i],
         rescaled_gyradius_without_random_data_9[i],
         rescaled_gyradius_without_random_data_10[i],
         rescaled_gyradius_without_random_data_11[i],
         rescaled_gyradius_without_random_data_12[i],
         rescaled_gyradius_without_random_data_13[i],
         rescaled_gyradius_without_random_data_14[i],
         rescaled_gyradius_without_random_data_15[i],
         rescaled_gyradius_without_random_data_16[i],
         rescaled_gyradius_without_random_data_17[i],
         rescaled_gyradius_without_random_data_18[i],
         rescaled_gyradius_without_random_data_19[i],
         rescaled_gyradius_without_random_data_20[i],
         rescaled_gyradius_without_random_data_21[i],
         rescaled_gyradius_without_random_data_22[i],
         rescaled_gyradius_without_random_data_23[i],
         rescaled_gyradius_without_random_data_24[i],
         rescaled_gyradius_without_random_data_25[i],
         rescaled_gyradius_without_random_data_26[i],
         rescaled_gyradius_without_random_data_27[i],
         rescaled_gyradius_without_random_data_28[i],
         rescaled_gyradius_without_random_data_29[i],
         rescaled_gyradius_without_random_data_30[i]]
        )
    )

rescaled_gyradius_without_random_data_mean = np.array(rescaled_gyradius_without_random_data_mean)
rescaled_gyradius_without_random_data_stdev = np.array(rescaled_gyradius_without_random_data_stdev)

rescaled_gyradius_without_random_data_lower = rescaled_gyradius_without_random_data_mean - rescaled_gyradius_without_random_data_stdev
rescaled_gyradius_without_random_data_upper = rescaled_gyradius_without_random_data_mean + rescaled_gyradius_without_random_data_stdev


# Plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Plot: training error
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), training_error_mean, label = 'full training dataset')
plt.plot(np.arange(1, num_epochs + 1, 1), training_error_without_stragglers_mean, label = 'stragglers removed')
plt.plot(np.arange(1, num_epochs + 1, 1), training_error_without_random_data_mean, label = 'random data removed')
plt.fill_between(np.arange(1, num_epochs + 1, 1), training_error_lower, training_error_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), training_error_without_stragglers_lower, training_error_without_stragglers_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), training_error_without_random_data_lower, training_error_without_random_data_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(0, 0.5, 0.05))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('training error', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='x-large', loc='upper center')
plt.xlim(0, 150)
plt.ylim(0, 0.2)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/stragglers/training_error.png', dpi=300)
plt.show()

# Plot: test error
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), test_error_mean, label = 'full training dataset')
plt.plot(np.arange(1, num_epochs + 1, 1), test_error_without_stragglers_mean, label = 'stragglers removed')
plt.plot(np.arange(1, num_epochs + 1, 1), test_error_without_random_data_mean, label = 'random data removed')
plt.fill_between(np.arange(1, num_epochs + 1, 1), test_error_lower, test_error_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), test_error_without_stragglers_lower, test_error_without_stragglers_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), test_error_without_random_data_lower, test_error_without_random_data_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(0, 0.5, 0.05))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('test error', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='x-large', loc='upper center')
plt.xlim(0, 150)
plt.ylim(0, 0.2)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/stragglers/test_error.png', dpi=300)
plt.show()

# Plot: radius of gyration
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), gyradius_mean, label = 'full training dataset')
plt.plot(np.arange(1, num_epochs + 1, 1), gyradius_without_stragglers_mean, label = 'stragglers removed')
plt.plot(np.arange(1, num_epochs + 1, 1), gyradius_without_random_data_mean, label = 'random data removed')
plt.fill_between(np.arange(1, num_epochs + 1, 1), gyradius_lower, gyradius_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), gyradius_without_stragglers_lower, gyradius_without_stragglers_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), gyradius_without_random_data_lower, gyradius_without_random_data_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(0, 2, 0.2))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('radius of gyration', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='x-large', loc='center right')
plt.xlim(0, 150)
plt.ylim(0.2, 1.0)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/stragglers/radius_of_gyration.png', dpi=300)
plt.show()

# Plot: centre-to-centre distance
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), distance_mean, label = 'full training dataset')
plt.plot(np.arange(1, num_epochs + 1, 1), distance_without_stragglers_mean, label = 'stragglers removed')
plt.plot(np.arange(1, num_epochs + 1, 1), distance_without_random_data_mean, label = 'random data removed')
plt.fill_between(np.arange(1, num_epochs + 1, 1), distance_lower, distance_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), distance_without_stragglers_lower, distance_without_stragglers_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), distance_without_random_data_lower, distance_without_random_data_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(1.0, 2.1, 0.2))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('centre-to-centre distance', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='x-large', loc='center right')
plt.xlim(0, 150)
plt.ylim(1.0, 2.0)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/stragglers/ctc_distance.png', dpi=300)
plt.show()

# Plot: dimensionless radius of gyration
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_mean, label = 'full training dataset')
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_without_stragglers_mean, label = 'stragglers removed')
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_without_random_data_mean, label = 'random data removed')
plt.fill_between(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_lower, rescaled_gyradius_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_without_stragglers_lower, rescaled_gyradius_without_stragglers_upper, alpha=0.2)
plt.fill_between(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_without_random_data_lower, rescaled_gyradius_without_random_data_upper, alpha=0.2)
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(0, 2, 0.2))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('dimensionless radius of gyration', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='x-large', loc='upper center')
plt.xlim(0, 150)
plt.ylim(0, 1)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/stragglers/dimensionless_radius_of_gyration.png', dpi=300)
plt.show()