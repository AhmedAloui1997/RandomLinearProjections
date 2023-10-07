import torch
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split

def balanced_batch_generator_reg(data, labels, N, K):
    num_samples = len(data)
    selected_batches = set()
    all_indices = np.arange(num_samples)

    # Make sure each data point is in at least one batch
    np.random.shuffle(all_indices)
    for i in range(0, num_samples, N):
        selected_batches.add(tuple(sorted(all_indices[i:i+N])))

    # Add additional random unique batches until we have K batches
    while len(selected_batches) < K:
        batch_indices = tuple(sorted(np.random.choice(num_samples, N, replace=False)))
        if batch_indices not in selected_batches:
            selected_batches.add(batch_indices)

    selected_batches = list(selected_batches)  # convert back to list for indexing
    for indices in selected_batches:
        yield data[np.array(indices)], labels[np.array(indices)]

def balanced_batch_generator_auto(data, N, K):
    num_samples = len(data)
    selected_batches = set()
    all_indices = np.arange(num_samples)

    # Make sure each data point is in at least one batch
    np.random.shuffle(all_indices)
    for i in range(0, num_samples, N):
        selected_batches.add(tuple(sorted(all_indices[i:i+N])))

    # Add additional random unique batches until we have K batches
    while len(selected_batches) < K:
        batch_indices = tuple(sorted(np.random.choice(num_samples, N, replace=False)))
        if batch_indices not in selected_batches:
            selected_batches.add(batch_indices)

    selected_batches = list(selected_batches)  # convert back to list for indexing
    for indices in selected_batches:
        yield data[torch.tensor(indices)]  # use torch tensors for indexing

def SplitDataset(X, y, shift, train_size):
    
    if shift == -1:
        X_train, X_test, y_train, y_test = train_test_split_fixed(X, y, train_size)
        
    elif shift >= 0 and shift <= 1:
        # Assuming X and y are already defined and have the same number of samples
        num_samples = X.shape[0]
        
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        
        # Compute the condition for each sample in X
        condition = np.abs(X - mean_X) < 0.5 * std_X
        
        # Based on the condition, sample the indices for the training set
        train_indices = []
        
        for i in range(num_samples):
            if condition[i].all():
                # Sample with probability p
                if np.random.rand() < shift:
                    train_indices.append(i)
            else:
                # Sample with probability 1-p
                if np.random.rand() < 1 - shift:
                    train_indices.append(i)

        # All other indices are for the test set
        test_indices = [i for i in range(num_samples) if i not in train_indices]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        
        X_test = X[test_indices]
        y_test = y[test_indices]

    else:
        print('Error: Shift must be between 0 and 1')
        
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
        raise ValueError("train_size must be between 0 and the total number of samples.")
    
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