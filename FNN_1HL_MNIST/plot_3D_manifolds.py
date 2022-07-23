# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# File path
PATH = "."

# Choose run whose manifolds will be plotted
run = 333

num_epochs = 150

manifolds_0 = []
manifolds_1 = []
# 3D Manifolds at each training epoch:
for epoch in range (1, num_epochs+1):
    f_manifold_0 = open("{}/Runs/run_{}/data/manifold_0_epoch_{}.dat".format(PATH, run, epoch), "r")
    f_manifold_1 = open("{}/Runs/run_{}/data/manifold_1_epoch_{}.dat".format(PATH, run, epoch), "r")
    manifold_0 = eval(f_manifold_0.read())
    manifold_1 = eval(f_manifold_1.read())
    f_manifold_0.close()
    f_manifold_1.close()
    manifolds_0.append(manifold_0)
    manifolds_1.append(manifold_1)


# Plot: 3D manifolds at each training epoch

# Plot style
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rcParams["figure.autolayout"] = True

for epoch in range (0, num_epochs):
    # Define data
    even_x = []
    even_y = []
    even_z = []
    odd_x = []
    odd_y = []
    odd_z = []
    for i in range (len(manifolds_0[0])):
        even_x.append(manifolds_0[epoch][i][0])
        even_y.append(manifolds_0[epoch][i][1])
        even_z.append(manifolds_0[epoch][i][2])
    for i in range (len(manifolds_1[0])):
        odd_x.append(manifolds_1[epoch][i][0])
        odd_y.append(manifolds_1[epoch][i][1])
        odd_z.append(manifolds_1[epoch][i][2])
    # Create figure
    fig = plt.figure(figsize = (6, 6))
    fig.tight_layout()
    ax = plt.axes(projection='3d')
    # Create plot
    ax.scatter3D(even_x, even_y, even_z, label='"even" manifold')
    ax.scatter3D(odd_x, odd_y, odd_z, label='"odd" manifold')
    # Add axes
    ax.set_xlabel('x', labelpad=20, fontsize='xx-large')
    ax.set_ylabel('y', labelpad=20, fontsize='xx-large')
    ax.set_zlabel('z', labelpad=20, fontsize='xx-large')
    # Set ticks labels
    ax.set_xticks(np.arange(-1, 1.5, 0.5))
    ax.set_yticks(np.arange(-1, 1.5, 0.5))
    ax.set_zticks(np.arange(-1, 1.5, 0.5))
    ax.tick_params(axis='z', which='major', pad=10)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('left')
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('right')
    # Modify axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    # Add grid
    ax.grid()
    # Change view angle
    ax.view_init(15, 30)
    # Add legend
    ax.legend(loc='upper center', fontsize='xx-large')
    # Save & show plot
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.savefig("./../../Desktop/Tesi/Figures/MNIST/3D_manifolds/epoch_{}.png".format(epoch + 1), dpi=300)
    #plt.show()