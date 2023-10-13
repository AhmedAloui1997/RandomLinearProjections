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
    elif DatasetName == "1DPolynomial":
        pass # create dataset
    elif DatasetName == "5DPolynomial": 
        pass # create dataset
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