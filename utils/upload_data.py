import numpy as np
from sklearn.datasets import fetch_california_housing

def LoadDataset(DatasetName):
    
    if DatasetName == "CaliforniaHousing":
        # load California Housing
        california = fetch_california_housing()
        X, y = california.data, np.expand_dims(california.target,1)
    elif DatasetName == "Diabetes":
        pass # load Diabetes dataset
    elif DatasetName == "MNIST":
        pass # load MNIST
    elif DatasetName == "CIFAR10":
        pass # load CIFAR10
    elif DatasetName == "1DPolynomial":
        pass # create dataset
    elif DatasetName == "5DPolynomial":
        pass # create dataset
    elif DatasetName == "TrigonometricFunctions":
        pass # create dataset
    else:
        print("Error: Dataset name is undefined")
        
    return X, y