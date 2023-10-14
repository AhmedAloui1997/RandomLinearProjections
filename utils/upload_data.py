import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def LoadDataset(DatasetName):
    
    if DatasetName == "CaliforniaHousing":
        # load California Housing
        california = fetch_california_housing()
        output1, output2 = california.data, np.expand_dims(california.target,1)
        
    elif DatasetName == "WineQuality":
        # Fetch the "Elevators" dataset from OpenML
        wine = fetch_openml(name='wine_quality', version=1)
        output1, output2 = wine.data.to_numpy(), np.expand_dims(wine.target.to_numpy(),1)
        
    elif DatasetName == "Linear":
        # MULTIDIMENSIONAL DATASET
        rng = np.random.RandomState(0)
        X = rng.rand(6000, 5) # Same dataset will now be used across different trials
        y = np.expand_dims((0.5*X[:,0] + 1.5*X[:,1] + 2.5*X[:,2] + 3.5*X[:,3] + 4.5*X[:,4]), 1)
        output1, output2 = X, y
    
    elif DatasetName == "Nonlinear": 
        # MULTIDIMENSIONAL DATASET
        rng = np.random.RandomState(1)
        X = rng.rand(6000, 7) # Same dataset will now be used across different trials
        y = np.expand_dims((X[:,0] + X[:,1]**2 + X[:,2]**3 + X[:,3]**4 + X[:,4]**5 + np.exp(X[:,5]) + np.sin(X[:,6])), 1)
        output1, output2 = X, y
        
    elif DatasetName == "TrigonometricFunctions":
        pass # create dataset
    # image datasets
    elif DatasetName == "MNIST":
        transform = transforms.ToTensor()
        output1 = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
        output2 = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
    elif DatasetName == "CIFAR10":
        transform = transforms.ToTensor()
        output1 = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        output2 = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    else:
        print("Error: Dataset name is undefined")
        exit(1)
  
    return output1, output2