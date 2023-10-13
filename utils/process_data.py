import torch
import numpy as np
from sklearn.model_selection import train_test_split

def balanced_batch_generator_reg(data, labels, N, K):
    """
    Generates K batches of data and associated labels, each of size N, ensuring 
    that each data point appears in at least one batch.

    Parameters:
    - data (numpy.array): The dataset.
    - labels (numpy.array): Corresponding labels for the dataset.
    - N (int): Batch size.
    - K (int): Number of batches.

    Yields:
    - tuple: Batches of data and corresponding labels.
    """

    # Determine the total number of samples in the dataset
    num_samples = len(data)

    # Use a set to keep track of batches to ensure uniqueness
    selected_batches = set()

    # All available indices from 0 to num_samples
    all_indices = np.arange(num_samples)

    # Shuffle the indices to ensure randomness
    np.random.shuffle(all_indices)

    # Assign data points to batches, ensuring each data point is in at least one batch
    for i in range(0, num_samples, N):
        selected_batches.add(tuple(sorted(all_indices[i:i+N])))

    # Continuously sample random unique batches until we have a total of K batches
    while len(selected_batches) < K:
        # Randomly select N indices without replacement
        batch_indices = tuple(sorted(np.random.choice(num_samples, N, replace=False)))

        # Ensure uniqueness of the batch
        if batch_indices not in selected_batches:
            selected_batches.add(batch_indices)

    # Convert the set back to a list for easier indexing
    selected_batches = list(selected_batches)

    # Yield batches of data and their corresponding labels one by one
    for indices in selected_batches:
        yield data[np.array(indices)], labels[np.array(indices)]


def balanced_batch_generator_auto(data, M, K):
    """
    Generates K batches of data, each of size M, ensuring that each data point 
    appears in at least one batch.

    Parameters:
    - data (torch.Tensor): The dataset.
    - M (int): Batch size.
    - K (int): Number of batches.

    Yields:
    - torch.Tensor: Batches of data.
    """

    # Determine the total number of samples in the dataset
    num_samples = len(data)

    # Use a set to keep track of batches to ensure uniqueness
    selected_batches = set()

    # All available indices from 0 to num_samples
    all_indices = np.arange(num_samples)

    # Shuffle the indices to ensure randomness
    np.random.shuffle(all_indices)

    # Assign data points to batches, ensuring each data point is in at least one batch
    for i in range(0, num_samples, M):
        selected_batches.add(tuple(sorted(all_indices[i:i+M])))

    # Continuously sample random unique batches until we have a total of K batches
    while len(selected_batches) < K:
        # Randomly select N indices without replacement
        batch_indices = tuple(sorted(np.random.choice(num_samples, M, replace=False)))

        # Ensure uniqueness of the batch
        if batch_indices not in selected_batches:
            selected_batches.add(batch_indices)

    # Convert the set back to a list for easier indexing
    selected_batches = list(selected_batches)

    # Yield batches one by one
    for indices in selected_batches:
        yield data[torch.tensor(indices)]  # Convert indices to torch tensor for indexing


def SplitDataset(X, y, shift, train_size):
    """
    Split the dataset based on the specified shift criterion for regression tasks.

    Parameters:
    - X (numpy array): Input features.
    - y (numpy array): Target variable.
    - shift (float): Determines the split criterion. If -1, a fixed split is applied.
                     If between 0 and 1, splits based on deviation from mean criterion.
    - train_size (float): Fraction of data to be used for training in fixed split.

    Returns:
    - tuple: X_train, X_test, y_train, y_test arrays.
    """
    
    # If shift is -1, use a fixed size split
    if shift == -1:
        X_train, X_test, y_train, y_test = train_test_split_fixed(X, y, train_size)
        
    # If shift is between 0 and 1, split based on deviation from the mean criterion
    elif shift > 0 and shift < 1:
        # Determine number of samples in the dataset
        num_samples = X.shape[0]
        
        # Compute the mean and standard deviation for the features
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        
        # Define the condition for deviation from the mean
        condition = np.abs(X - mean_X) < 0.5 * std_X
        
        # Initialize list to store training data indices
        train_indices = []
        
        # For each sample, check the condition and decide if it goes into the training set
        for i in range(num_samples):
            # If all features of a sample satisfy the condition
            if condition[i].all():
                # Sample the index with probability 'shift'
                if np.random.rand() < shift:
                    train_indices.append(i)
            else:
                # For samples not meeting the condition, sample with probability '1-shift'
                if np.random.rand() < 1 - shift:
                    train_indices.append(i)

        # Indices not in train_indices form the test set
        test_indices = [i for i in range(num_samples) if i not in train_indices]
        
        # Split the data into training and test sets based on selected indices
        X_train = X[train_indices]
        y_train = y[train_indices]
        
        X_test = X[test_indices]
        y_test = y[test_indices]

    # If shift is not between 0 and 1 or -1, raise an error
    else:
        print('Error: Shift must be between 0 and 1')
        exit(1)
        
    return X_train, X_test, y_train, y_test

        
def train_test_split_fixed(X, y, train_size, shuffle=True, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters:
    - X, y: Arrays or matrices.
    - train_size: int, size of training dataset.
    - shuffle: Whether or not to shuffle the data before splitting.
    - random_state: Seed for reproducibility.
    
    Returns:
    - Split data into X_train, X_test, y_train, y_test.
    """
    
    # Ensure X and y have the same number of samples
    assert X.shape[0] == y.shape[0], "Inconsistent number of samples between X and y."
    
    # Ensure train_size is valid
    if not (0 <= train_size < X.shape[0]):
        print("Error: TrainSize must be between 0 and the total number of samples.")
        exit(1)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(X.shape[0])
    else:
        indices = np.arange(X.shape[0])
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def add_gaussian_noise(X_train, alpha):
    """
    Add standard normal noise with specified scaling factor (alpha) to X_train.
    
    Parameters:
    - X_train (numpy array): The input data.
    - alpha (float): The scaling factor.
    
    Returns:
    - numpy array: The noisy data.
    """
    
    # Generate Gaussian noise of the same shape as X_train
    noise = np.random.normal(0, 1, X_train.shape)
    
    # Add the noise to X_train
    X_train = X_train + noise * alpha
    
    return X_train