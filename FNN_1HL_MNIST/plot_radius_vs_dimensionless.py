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


# Radius of gyration:

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


# dimensionless radius of gyration:

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


# Plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')


# Plot: radius of gyration & radius of gyration rescaled by centre-centre distance
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, num_epochs + 1, 1), gyradius_mean, label='radius of gyration', color='red')
plt.plot(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_mean, label='dimensionless radius of gyration', color='blue')
plt.fill_between(np.arange(1, num_epochs + 1, 1), gyradius_lower, gyradius_upper, alpha=0.2, color='red')
plt.fill_between(np.arange(1, num_epochs + 1, 1), rescaled_gyradius_lower, rescaled_gyradius_upper, alpha=0.2, color='blue')
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(0.4, 1.1, 0.1))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('geometric observable', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='xx-large', loc='best')
plt.xlim(0, 150)
plt.ylim(0.4, 1)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/multiple_runs_average/radius_vs_dimensionless.png', dpi=300)
plt.show()