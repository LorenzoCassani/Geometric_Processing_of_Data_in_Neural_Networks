# Standard modules imports
import sys

# Third-party modules imports
import torch

# Local modules imports
import comp_rad


def oth_prop_cc_distance(datapath, run, epoch, num_epochs):
    f0 = open("{}/Runs/run_{}/data/manifold_0_epoch_{}.dat".format(datapath, run, epoch + 1), "r")
    data0 = eval(f0.read())
    data0 = torch.tensor(data0)
    f1 = open("{}/Runs/run_{}/data/manifold_1_epoch_{}.dat".format(datapath, run, epoch + 1), "r")
    data1 = eval(f1.read())
    data1 = torch.tensor(data1)
    f0.close()
    f1.close()
    
    if epoch == 0:
        f_dist = open("{}/Runs/run_{}/cc_distance/cc_distance.dat".format(datapath, run), "w+")
        f_dist.write("[")
    else:
        f_dist = open("{}/Runs/run_{}/cc_distance/cc_distance.dat".format(datapath, run), "a+")
    
    dist = comp_rad.cc_distance(data0, data1)
    
    f_dist.write("{},".format(dist) + "\n")
    
    if epoch == num_epochs - 1:
        f_dist.write("]")
    
    f_dist.close()
    
    return dist


def oth_prop_gyradius(datapath, run, epoch, num_epochs):
    f0 = open("{}/Runs/run_{}/data/manifold_0_epoch_{}.dat".format(datapath, run, epoch + 1), "r")
    data0 = eval(f0.read())
    data0 = torch.tensor(data0)
    f1 = open("{}/Runs/run_{}/data/manifold_1_epoch_{}.dat".format(datapath, run, epoch + 1), "r")
    data1 = eval(f1.read())
    data1 = torch.tensor(data1)
    f0.close()
    f1.close()
    
    if epoch == 0:
        f_r0 = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(datapath, run), "w+")
        f_r1 = open("{}/Runs/run_{}/gyradius/manifold_1.dat".format(datapath, run), "w+")
        f_r0.write("[")
        f_r1.write("[")
    else:
        f_r0 = open("{}/Runs/run_{}/gyradius/manifold_0.dat".format(datapath, run), "a+")
        f_r1 = open("{}/Runs/run_{}/gyradius/manifold_1.dat".format(datapath, run), "a+")
    
    rad0 = comp_rad.gyradius(data0)
    rad1 = comp_rad.gyradius(data1)
    
    f_r0.write("{},".format(rad0) + "\n")
    f_r1.write("{},".format(rad1) + "\n")
    
    if epoch == num_epochs - 1:
        f_r0.write("]")
        f_r1.write("]")
    
    f_r0.close()
    f_r1.close()
    
    return rad0, rad1


def oth_prop_gyradius_dimensionless(datapath, run, epoch, num_epochs):
    f0 = open("{}/Runs/run_{}/data/manifold_0_epoch_{}.dat".format(datapath, run, epoch + 1), "r")
    data0 = eval(f0.read())
    data0 = torch.tensor(data0)
    f1 = open("{}/Runs/run_{}/data/manifold_1_epoch_{}.dat".format(datapath, run, epoch + 1), "r")
    data1 = eval(f1.read())
    data1 = torch.tensor(data1)
    f0.close()
    f1.close()
    
    if epoch == 0:
        f_r0 = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(datapath, run), "w+")
        f_r1 = open("{}/Runs/run_{}/rescaled_gyradius/manifold_1.dat".format(datapath, run), "w+")
        f_r0.write("[")
        f_r1.write("[")
    else:
        f_r0 = open("{}/Runs/run_{}/rescaled_gyradius/manifold_0.dat".format(datapath, run), "a+")
        f_r1 = open("{}/Runs/run_{}/rescaled_gyradius/manifold_1.dat".format(datapath, run), "a+")
    
    rad0 = comp_rad.gyradius(data0) / comp_rad.cc_distance(data0, data1)
    rad1 = comp_rad.gyradius(data1) / comp_rad.cc_distance(data0, data1)
    
    f_r0.write("{},".format(rad0) + "\n")
    f_r1.write("{},".format(rad1) + "\n")
    
    if epoch == num_epochs - 1:
        f_r0.write("]")
        f_r1.write("]")
    
    f_r0.close()
    f_r1.close()
    
    return rad0, rad1