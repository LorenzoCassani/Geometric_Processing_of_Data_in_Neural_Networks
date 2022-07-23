# Standard modules imports
import random
import time
import copy
import ssl

# Third-party modules imports
import numpy as np
import sklearn.preprocessing as preprocessing
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# MNIST data preparation
def dataloader_MNIST(dataset, num_data, batch_size):
    start_ind = 0
    
    images = []
    labels = []
    
    for i in range(start_ind, start_ind + num_data):
        image = dataset.data[i] / 255	    # Rescales values in [0,1]
        image = image.view(-1)			    # Flattens images in a vector
        images += [image.squeeze_()]		# Collects images in a list
        labels += [dataset.targets[i] % 2]  # Collects labels in a list (even / odd classification: 0 == even / 1 == odd)
    
    with torch.no_grad():
        images = torch.stack(images)  # Transform the list of tensors in a unique tensor
        labels = torch.stack(labels)  # Transform the list of tensors in a unique tensor
    
    images = preprocessing.scale(images)  # Data standardization (stdev = 1 for each pixel)
    images = torch.from_numpy(images)
    images = images.type(torch.FloatTensor)  # Conversion to the right format
    
    # Mini-batches creation
    images = list(torch.split(images, batch_size))  # Splitting of images in mini-batches
    labels = list(torch.split(labels, batch_size))  # Splitting of labels in mini-batches
    
    dataloader = [images, labels]  # Collects images and labels in a dataloader
    
    return dataloader


# MNIST TCC (Two-Classes Classification) data preparation
def dataloader_MNIST_TCC(dataset, num_data, batch_size):
    start_ind = 0
    
    images = []
    labels = []
    
    #  Each training and test example is assigned to one of the following labels:
    #  0 T-shirt/top
    #  1 Trouser
    #  2 Pullover
    #  3 Dress
    #  4 Coat
    #  5 Sandal
    #  6 Shirt
    #  7 Sneaker
    #  8 Bag
    #  9 Ankle boot
    
    i = start_ind
    j = 0
    while j < num_data:
        if dataset.targets[i] == 0 or dataset.targets[i] == 6:
            image = dataset.data[i] / 255	    # Rescales values in [0,1]
            image = image.view(-1)			    # Flattens images in a vector
            images += [image.squeeze_()]		# Collects images in a list
            if dataset.targets[i] == 0:
                labels += [torch.tensor(0)]
                j += 1
            elif dataset.targets[i] == 6:
                labels += [torch.tensor(1)]
                j += 1
        i += 1
    
    with torch.no_grad():
        images = torch.stack(images)  # Transform the list of tensors in a unique tensor
        labels = torch.stack(labels)  # Transform the list of tensors in a unique tensor
    
    images = preprocessing.scale(images)  # Data standardization (stdev = 1 for each pixel)
    images = torch.from_numpy(images)
    images = images.type(torch.FloatTensor)  # Conversion to the right format
    
    # Mini-batches creation
    images = list(torch.split(images, batch_size))  # Splitting of images in mini-batches
    labels = list(torch.split(labels, batch_size))  # Splitting of labels in mini-batches
    
    dataloader = [images, labels]  # Collects images and labels in a dataloader
    
    return dataloader


# Custom MNIST dataset
class Custom_MNIST():
    
    def __init__(self, num_data, batch_size):
        
        self.num_data = num_data
        self.batch_size = batch_size
        
        self.stragglers = []     # Stragglers container
        self.num_epurations = 0  # Number of epurations
        self.num_shuffle = 0     # Number of shuffles
        
        # Full MNIST dataset
        # Transformations (the effective stanrdardization happens in "dataloader_MNIST")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        
        # Full MNIST training set
        self.trainset = datasets.MNIST(root='../Datasets/', train=True, download=True, transform=self.transform)
        # Full MNIST test set
        self.testset = datasets.MNIST(root='../Datasets/', train=False, download=True, transform=self.transform)
        
        # Subsets of MNIST effectively utilized
        # Training dataset that remains untouched by the stragglers removal
        self.trainloader_untouched = dataloader_MNIST(self.trainset, self.num_data, self.batch_size)
        # Training dataset that may experience stragglers removal
        self.trainloader = dataloader_MNIST(self.trainset, self.num_data, self.batch_size)
        # Testing dataset
        self.testloader = dataloader_MNIST(self.testset, len(self.testset), self.batch_size)
    
    # Shuffles (images, labels) pairs
    def shuffle(self):
        random.seed(time.time())                # Sets the seed
        random.shuffle(self.trainloader[0][0])  # Shuffles images
        random.seed(time.time())                # Resets the seed (such that images and labels are shuffled in pairs)
        random.shuffle(self.trainloader[1][0])  # Shuffle labels
        self.num_shuffle += 1
    
    # Shuffles labels only
    def shuffle_labels(self):
        # Train Set labels shuffling
        random.seed(time.time())                # Sets the seed
        random.shuffle(self.trainloader[1][0])  # Shuffles labels
        # Test set labels shuffling
        random.seed(time.time())                # Sets the seed
        random.shuffle(self.testloader[1][0])   # Shuffles labels
        
    # Epuration of stragglers
    # Note 1: the stragglers indices (computed in "Net.stragglers" function)
    # are referred to the stragglers positions in base_dataset (sorted and not epured)
    def epure(self):
        # Copy of the base dataset
        images = copy.deepcopy(self.trainloader_untouched[0])
        labels = copy.deepcopy(self.trainloader_untouched[1])
        
        # Concatenates images and labels in one tensor
        with torch.no_grad():
            images = torch.cat(images)
            labels = torch.cat(labels)
        
        images = images.tolist()
        labels = labels.tolist()
        
        # Stragglers removal
        # Note 2: this works thanks to Note 1 and thanks to dataset.stragglers being sorted
        k = 0
        for indices in self.stragglers[0]:
            del(images[indices[0]*self.batch_size + indices[1] - k])
            del(labels[indices[0]*self.batch_size + indices[1] - k])
            k += 1
        
        labels = torch.tensor(labels)			
        images = torch.tensor(images)
        images = preprocessing.scale(images)  # Standardization of the epured dataset
        images = torch.from_numpy(images)
        images = images.type(torch.FloatTensor)  # Conversion to the right format
        
        # Creates mini-batches
        images = list(torch.split(images, self.batch_size))
        labels = list(torch.split(labels, self.batch_size))
        
        self.trainloader = [images, labels]
        self.num_epurations += 1


# Custom EMNIST dataset
class Custom_EMNIST():
    
    def __init__(self, num_data, batch_size):
        
        self.num_data = num_data
        self.batch_size = batch_size
        
        self.stragglers = []     # Stragglers container
        self.num_epurations = 0  # Number of epurations
        self.num_shuffle = 0     # Number of shuffles
        
        # Full EMNIST dataset
        # Transformations (the effective stanrdardization happens in "dataloader_MNIST")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        
        # Full EMNIST training set
        self.trainset = datasets.EMNIST(root='../Datasets/', split='letters', train=True, download=True, transform=self.transform)
        # Full EMNIST test set
        self.testset = datasets.EMNIST(root='../Datasets/', split='letters', train=False, download=True, transform=self.transform)
        
        # Subsets of EMNIST effectively utilized
        # Training dataset that remains untouched by the stragglers removal
        self.trainloader_untouched = dataloader_MNIST(self.trainset, self.num_data, self.batch_size)
        # Training dataset that may experience stragglers removal
        self.trainloader = dataloader_MNIST(self.trainset, self.num_data, self.batch_size)
        # Testing dataset
        self.testloader = dataloader_MNIST(self.testset, len(self.testset), self.batch_size)


# Custom KMNIST dataset
class Custom_KMNIST():
    
    def __init__(self, num_data, batch_size):
        
        self.num_data = num_data
        self.batch_size = batch_size
        
        self.stragglers = []     # Stragglers container
        self.num_epurations = 0  # Number of epurations
        self.num_shuffle = 0     # Number of shuffles
        
        # Full KMNIST dataset
        # Transformations (the effective stanrdardization happens in "dataloader_MNIST")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        
        # Full KMNIST training set
        self.trainset = datasets.KMNIST(root='../Datasets/', train=True, download=True, transform=self.transform)
        # Full KMNIST test set
        self.testset = datasets.KMNIST(root='../Datasets/', train=False, download=True, transform=self.transform)
        
        # Subsets of KMNIST effectively utilized
        # Training dataset that remains untouched by the stragglers removal
        self.trainloader_untouched = dataloader_MNIST(self.trainset, self.num_data, self.batch_size)
        # Training dataset that may experience stragglers removal
        self.trainloader = dataloader_MNIST(self.trainset, self.num_data, self.batch_size)
        # Testing dataset
        self.testloader = dataloader_MNIST(self.testset, len(self.testset), self.batch_size)


# Custom Fashion-MNIST dataset
class Custom_FashionMNIST():
    
    def __init__(self, num_data, batch_size):
        
        self.num_data = num_data
        self.batch_size = batch_size
        
        self.stragglers = []     # Stragglers container
        self.num_epurations = 0  # Number of epurations
        self.num_shuffle = 0     # Number of shuffles
        
        # Full Fashion-MNIST dataset
        # Transformations (the effective stanrdardization happens in "dataloader_MNIST")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        
        # Full Fashion-MNIST training set
        self.trainset = datasets.FashionMNIST(root='../Datasets/', train=True, download=True, transform=self.transform)
        # Full Fashion-MNIST test set
        self.testset = datasets.FashionMNIST(root='../Datasets/', train=False, download=True, transform=self.transform)
        
        # Subsets of Fashion-MNIST effectively utilized
        # Training dataset that remains untouched by the stragglers removal
        self.trainloader_untouched = dataloader_MNIST(self.trainset, self.num_data, self.batch_size)
        # Training dataset that may experience stragglers removal
        self.trainloader = dataloader_MNIST(self.trainset, self.num_data, self.batch_size)
        # Testing dataset
        self.testloader = dataloader_MNIST(self.testset, len(self.testset), self.batch_size)


# Custom Fashion-MNIST TCC (Two-Classes Classification) dataset
class Custom_FashionMNIST_TCC():
    
    def __init__(self, num_data, batch_size):
        
        self.num_data = num_data
        self.batch_size = batch_size
        
        self.stragglers = []     # Stragglers container
        self.num_epurations = 0  # Number of epurations
        self.num_shuffle = 0     # Number of shuffles
        
        # Full MNIST dataset
        # Transformations (the effective stanrdardization happens in "dataloader_MNIST")
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        
        # Full MNIST training set
        self.trainset = datasets.FashionMNIST(root = '../Datasets/', train=True, download=True, transform=self.transform)
        # Full MNIST test set
        self.testset = datasets.FashionMNIST(root = '../Datasets/', train=False, download=True, transform=self.transform)
        
        # Subsets of MNIST effectively utilized
        # Training dataset that remains untouched by the stragglers removal
        self.trainloader_untouched = dataloader_MNIST_TCC(self.trainset, self.num_data, self.batch_size)
        # Training dataset that may experience stragglers removal
        self.trainloader = dataloader_MNIST_TCC(self.trainset, self.num_data, self.batch_size)
        # Testing dataset
        self.testloader = dataloader_MNIST_TCC(self.testset, 1000, self.batch_size)