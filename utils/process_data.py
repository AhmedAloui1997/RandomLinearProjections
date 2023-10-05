import torch
from itertools import combinations
import numpy as np

def balanced_batch_generator(data, N, K):
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
