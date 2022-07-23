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
num_stragglers_exiting = []
num_stragglers_entering = []


for epoch_star in range (1, num_epochs + 1):
    f1 = open('{}/stragglers_list/stragglers_at_epoch_{}.dat'.format(PATH, epoch_star), 'r')
    f2 = open('{}/stragglers_list/stragglers_at_epoch_{}.dat'.format(PATH, epoch_star + 1), 'r')
    list1 = eval(f1.read())
    list2 = eval(f2.read())
    f1.close()
    f2.close()
    stragglers_exiting = [[sub_x for sub_x in x if sub_x not in y] for x, y in zip(list1, list2)]
    stragglers_entering = [[sub_y for sub_y in y if sub_y not in x] for x, y in zip(list1, list2)]
    num_stragglers_exiting.append(len(stragglers_exiting[0]))
    num_stragglers_entering.append(len(stragglers_entering[0]))


# Plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Plot: number of elements exiting/entering S(t)
plt.figure(figsize = (10, 6))
plt.scatter(np.arange(1, num_epochs + 1, 1), num_stragglers_exiting, label='exiting S(t)')
plt.scatter(np.arange(1, num_epochs + 1, 1), num_stragglers_entering, label='entering S(t)')
plt.xticks(np.arange(0, num_epochs + 1, 15))
plt.yticks(np.arange(0, 200, 20))
plt.xlabel('training epoch', fontsize='xx-large')
plt.ylabel('number of data points', fontsize='xx-large')
plt.grid()
plt.legend(fontsize='xx-large', loc='upper center')
plt.xlim(0, num_epochs)
plt.ylim(0, 80)
plt.savefig('../../Desktop/Tesi/Figures/MNIST/filtration/filtration.png', dpi=300)
plt.show()