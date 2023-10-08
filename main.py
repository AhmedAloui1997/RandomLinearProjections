import argparse
import sys
import numpy as np
import torch
import os

sys.path.append('./RandomLinearProjections')
from utils.upload_data import LoadDataset
from utils.process_data import SplitDataset
from models.train_regression import train
from models.train_autoencoder import train_encoder_results
# to run file: python3 main.py --Dataset "DatasetName" (optional: --Shift "Shift", etc.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations.')
    parser.add_argument('--Dataset', type=str, help='Dataset to upload', required=True)
    parser.add_argument('--Task', type=str, default='Regression', help='Task (Regression or Autoencoder)')
    parser.add_argument('--LossFunction', type=str, default='MSE', help='Loss function (MSE, MSEL2, or RLP)')
    parser.add_argument('--TrainSize', type=float, default=10000, help="Training dataset size", required=False)
    parser.add_argument('--BatchSize', type=int, default=100, help='Size of each training batch')
    parser.add_argument('--NumBatches', type=int, default=1000, help='Number of RLP Loss Training Batches')
    parser.add_argument('--Epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--Iterations', type=int, default=30, help='Number of iterations to repeat task')
    parser.add_argument('--Shift', type=float, default=-1, help="Dataset shift", required=False)
    args = parser.parse_args()
    
    DatasetName = args.Dataset
    task = args.Task
    loss_function = args.LossFunction
    train_size = args.TrainSize
    batch_size = args.BatchSize
    num_batches = args.NumBatches
    epochs = args.Epochs
    iterations = args.Iterations
    shift = args.Shift

    # This function calls the dataset we need for the experiment
    # distinguish between regression and autoencoder
    if task == 'Regression':
        X, y = LoadDataset(DatasetName)
        train_losses, test_losses = train(X, y, shift, train_size, task, iterations, epochs, batch_size, num_batches, loss_function)
    elif task == 'Autoencoder':
        X_train, X_test = LoadDataset(DatasetName)
        train_losses, test_losses = train_encoder_results(X_train, X_test, shift, train_size, task, iterations, epochs, batch_size, num_batches, loss_function)
    elif task == 'VAE':
        pass
    else:
        print('Error: Task must be Regression or Autoencoder')

    
    
    # Path for the directory
    results_dir = 'Results'
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Path for the CSV files
    train_losses_path = f'Results/TRAIN_{DatasetName}_{task}_{loss_function}_TrainSize={train_size}_BatchSize={batch_size}_NumBatches={num_batches}_Epochs={epochs}_Iterations={iterations}_Shift={shift}.csv'
    test_losses_path = f'Results/TEST_{DatasetName}_{task}_{loss_function}_TrainSize={train_size}_BatchSize={batch_size}_NumBatches={num_batches}_Epochs={epochs}_Iterations={iterations}_Shift={shift}.csv'
    
    # Save matrices to CSV
    np.savetxt(train_losses_path, train_losses, delimiter=',')
    np.savetxt(test_losses_path, test_losses, delimiter=',')
