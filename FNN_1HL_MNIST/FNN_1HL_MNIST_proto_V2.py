# Standard modules imports
import os
import sys
from time import time

# Third-party modules imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Local modules imports
sys.path.insert(1, "../Libraries/")  # Local modules path
import datasets
import manif_oth_prop
#import join_stragglers

# File output path
PATH = "."


# Rightness function
def rightness(output, labels, batch_size):
    rights = 0
    for j in range(batch_size):  # j: images / output / labels within the mini-batch
        out = output[j].cpu().detach().numpy()
        ground_truth = labels[j].cpu().detach().numpy()
        if out[0] >= out[1]:
            prediction = 0
        else:
            prediction = 1
        if prediction == ground_truth:
            rights += 1
    return rights

# Manifolds dictionaring function:
# stores data inner representations (at hidden layer) and their respective labels in a container.
def dictionaring(dictionary, images, labels, num_classes):
    for i in range(len(images)):
        dictionary[labels[i].item()] += [images[i]]
    return dictionary


# Manifolds writing function:
# writes data inner representations (at hidden layer) in a file.
def save_data(dictionary_hidden, run):
    for i in range(len(dictionary_hidden)):
        filename = "{}/Runs/run_{}/data/manifold_{}.dat".format(PATH, run, i)
        f2 = open(filename, "w+")
        f2.write("[\n")
        for k in range(len(dictionary_hidden[i])):
            f2.write(str(dictionary_hidden[i][k].tolist()) + ",\n")
        f2.write("]")
        f2.close()


# Directories creation
def create_directories(run):
    os.makedirs("{}/Runs/run_{}".format(PATH, run), exist_ok=True)
    os.makedirs("{}/Runs/run_{}/errors".format(PATH, run), exist_ok=True)
    os.makedirs("{}/Runs/run_{}/data".format(PATH, run), exist_ok=True)
    os.makedirs("{}/Runs/run_{}/cc_distance".format(PATH, run), exist_ok=True)
    os.makedirs("{}/Runs/run_{}/gyradius".format(PATH, run), exist_ok=True)
    os.makedirs("{}/Runs/run_{}/rescaled_gyradius".format(PATH, run), exist_ok=True)


# Neural network
class FCNN(nn.Module):
    
    # NN initializer
    # Note: each image is 28 x 28, and is being stored as a flattened row of length 784 (=28*28).
    def __init__(self, epoch_star, hidden_size, input_size=28*28*1, output_size=2):
        super().__init__()
        
        # NN structure
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Tanh())
        
        self.epoch_star = epoch_star   # Epoch at which we identify stragglers
        
        # NN Hyperparameters
        self.num_data = 10000          # Training set size (Max 60000 for MNIST dataset)
        self.batch_size = 10000        # Mini-batch size
        self.learning_rate = 0.5       # Optimizer learning rate
        self.momentum = 0.             # Optimizer momentum
        
        # NN Variables
        self.loss = 0.                 # Average loss (at a certain epoch)
        self.training_error = 0.       # Average training error (at a certain epoch)
        self.test_error = 0.           # Average test error (at a certain epoch)
        
        # Loss function: Cross Entropy Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer: Mini-batch Gradient Descent
        # Note: it is erroneously named SGD (Stochastic Gradient Descent) in PyTorch
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        # Dataset
        self.dataset = datasets.Custom_MNIST(num_data=self.num_data, batch_size=self.batch_size)
        
        # Computing Device Choice
        self.device = torch.device("cpu")
        '''
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("\nUsing GPU")
        else:
            selfdevice = torch.device("cpu")
            print("\nUsing CPU")
        '''
        
        print("\nNumber of training data: {}".format(self.num_data))
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    # Returns a torch.tensor tuple composed of data at output and data at hidden layer
    def forward_p(self, x):
        y = self.fc1(x)  # Copies data at hidden layer
        z = self.fc2(y)
        return z, y
    
    # Finds stragglers indices in the unshuffled "base_dataset" (useful for comparing stragglers from different runs)
    def stragglers(self, dataset, run):
        stragglers_list = []
        stragglers_shuffled = []
        
        images_list = self.dataset.trainloader_untouched[0]
        labels_list = self.dataset.trainloader_untouched[1]
        
        for i in range(len(labels_list)):
            labels = labels_list[i].to(self.device)
            images = images_list[i].to(self.device)
            output, hidden = self.forward_p(images)  # Computes output (and copies data at hidden layer)
            
            for k in range(len(labels_list[i])):
                out = output[k].cpu().detach().numpy()
                ground_truth = labels[k].cpu().detach().numpy()
                if np.amax(out) == out[0]:
                    prediction = 0
                elif np.amax(out) == out[1]:
                    prediction = 1
                if prediction != ground_truth:   # If not correctly classified,
                    stragglers_list += [(i, k)]  # saves index in the stragglers indices list
    
    
    # ========== Training session ==========
    
    def training_session(self, run, num_epochs, innerdata=False):
        
        # Removes stragglers from dataset
        f1 = open('{}/stragglers_list/stragglers_at_epoch_{}.dat'.format(PATH, self.epoch_star), 'r')
        f2 = open('{}/stragglers_list/stragglers_at_epoch_{}.dat'.format(PATH, self.epoch_star + 1), 'r')
        list1 = eval(f1.read())
        list2 = eval(f2.read())
        f1.close()
        f2.close()
        
        stragglers_delta = [[sub_x for sub_x in x if sub_x not in y] for x, y in zip(list1, list2)]

        self.dataset.stragglers = stragglers_delta
        self.dataset.epure()
        
        
        f_training_err = open("{}/Runs/run_{}/errors/training_err.dat".format(PATH,run), "w+")
        f_training_err.write("[")
        f_test_err = open("{}/Runs/run_{}/errors/test_err.dat".format(PATH,run), "w+")
        f_test_err.write("[")
        
        errors = []               # List for recording training and test errors
        cc_distance = []          # List for recording centre-centre distances
        gyradii = []              # List for recording gyration radii
        rescaled_gyradii = []     # List for recording dimensionless gyration radii
        
        time0 = time()
        
        for epoch in range(num_epochs):
            # Identifies stragglers
            '''
            # Finds stragglers at epoch = epoch_star
            if (epoch == self.epoch_star):
                stragglers_list  = self.stragglers(self.dataset, run)
                self.dataset.stragglers += [stragglers_list]
            '''
            # Creates the container for the inner representations at a certain epoch
            if innerdata:
                dictionary_hidden = {}
                for i in range(2):
                    dictionary_hidden[i] = []
            
            images_list = self.dataset.trainloader[0]
            labels_list = self.dataset.trainloader[1]
            
            num_data = 0
            total_loss = 0.
            training_rights = 0
            
            num_minibatches = len(labels_list)
            for minib in range(num_minibatches):  # minib: minibatch index
                images = images_list[minib].to(self.device)  # Unpack mini-batch: images
                labels = labels_list[minib].to(self.device)  # Unpack mini-batch: labels
                batch_size = len(labels)  # Mini-batch Size
                
                self.optimizer.zero_grad()  # Set all the partial derivatives to zero
                output, hidden = self.forward_p(images)  # Computes output (and copies data at hidden layer)
                
                # Computes mini-batch training rights
                training_rights += rightness(output, labels, batch_size)
                    
                # Saves inner representations at hidden layer
                if innerdata:
                    dictionary_hidden = dictionaring(dictionary_hidden, hidden, labels, 2)  # Sorts all data in a dictionary (by their label)
                
                loss = self.criterion(output, labels)  # Computes average mini-batch loss
                
                # Parameters optimization
                loss.backward()  # Computes partial derivatives
                self.optimizer.step()  # NN parameters optimization
                
                total_loss += loss*batch_size  # Total loss: average mini-batch loss multiplied by mini-batch size
                num_data += batch_size
            
            # Inner representations geometry at a certain epoch
            if innerdata:
                # Writes inner representations on a file (one for each label, for each epoch)
                save_data(dictionary_hidden, run)
                # Computes inner representations geometric observables
                cc_distance.append(manif_oth_prop.oth_prop_cc_distance(PATH, run, epoch, num_epochs))
                gyradii.append(manif_oth_prop.oth_prop_gyradius(PATH, run, epoch, num_epochs))
                rescaled_gyradii.append(manif_oth_prop.oth_prop_gyradius_dimensionless(PATH, run, epoch, num_epochs))
            
            self.loss = total_loss.item() / num_data
            self.training_error = 1 - float(training_rights / num_data)
            f_training_err.write("[" + str(self.training_error) + "]," + "\n")
            
            print('Epoch [{:3}/{}], Loss: {:.4f} | Training error rate: {:6.2f} % '.format(
                epoch + 1, num_epochs, self.loss, 100*self.training_error), end = "")
            
            
            # ========== Test session ==========
            
            test_images_list = self.dataset.testloader[0]
            test_labels_list = self.dataset.testloader[1]
            
            test_rights = 0
            num_test_minibatches = len(test_labels_list)
            
            for minib in range(num_test_minibatches):
                test_images = test_images_list[minib].to(self.device)
                test_labels = test_labels_list[minib].to(self.device)
                
                test_output = self.forward(test_images)
                
                test_batch_size = len(test_output)
                
                # Computes mini-batch test rights
                test_rights += rightness(test_output, test_labels, test_batch_size)
            
            self.test_error = 1 - float(test_rights / len(self.dataset.testset))
            f_test_err.write("[" + str(self.test_error) + "]," + "\n")
            
            print('| Test error rate: {:6.2f} %'.format(100*self.test_error))
            
            # "errors" records (training error rate, test error rate) at each epoch
            errors.append((self.training_error, self.test_error))
        
        print("\nTraining Time (in minutes):", (time()-time0) / 60)

        f_training_err.write("]")
        f_training_err.close()
        f_test_err.write("]")
        f_test_err.close()


# ========== Main ==========

def main():
    run = 0
    epochs = 200
    create_directories(run)
    net = FCNN(30, 10)
    print("\nNet created:\n")
    print(net)
    print("")
    net.to(net.device)
    net.device
    net.training_session(run, epochs, innerdata=False)


if __name__ == '__main__':
    main()

'''
def main():
    run = 1151
    epoch_star = 1
    while run <= 1300:
        epochs = 200
        create_directories(run)
        net = FCNN(epoch_star, 10)
        print("\nNet created:\n")
        print(net)
        print("")
        net.to(net.device)
        net.device
        net.training_session(run, epochs, innerdata=False)
        run += 1
        epoch_star += 1


if __name__ == '__main__':
    main()
'''

# Dictionary:
# "FCNN" stands for "Fully Connected Neural Network" (a.k.a. "Multilayer Perceptron")
# FCNN == MLP
# f.c. stands for "fully connected"