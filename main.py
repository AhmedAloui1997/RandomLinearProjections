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
    parser.add_argument('--TrainSize', type=int, default=10000, help="Training dataset size", required=False)
    parser.add_argument('--BatchSize', type=int, default=100, help='Size of each training batch')
    parser.add_argument('--NumBatches', type=int, default=1000, help='Number of RLP Loss Training Batches')
    parser.add_argument('--Epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--Iterations', type=int, default=30, help='Number of iterations to repeat task')
    parser.add_argument('--Shift', type=float, default=-1, help="Dataset shift", required=False)
    parser.add_argument('--Noise', type=float, default=-1, help="Noise Scaling Factor", required=False)
    parser.add_argument('--EvalMetric', type=str, default='MSE', help='Evaluation Metric (MSE or RLP)', required=False)
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
    noise = args.Noise
    eval_metric = args.EvalMetric

    # This function calls the dataset we need for the experiment
    # Distinguish between regression and autoencoder
    if task == 'Regression':
        X, y = LoadDataset(DatasetName)
        train_losses, test_losses = train(X, y, eval_metric, noise, shift, train_size, task, iterations, epochs, batch_size, num_batches, loss_function)
    elif task == 'Autoencoder':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, X_test = LoadDataset(DatasetName)
        train_losses, test_losses = train_encoder_results(X_train, X_test, shift, train_size, task, iterations, epochs, batch_size, num_batches, loss_function, device)
    elif task == 'VAE':
        pass
    else:
        print('Error: Task must be Regression or Autoencoder')
        exit(1)
    
    
    # Path for the directory
    results_dir = 'Results'
    MSE_dir = 'Results/MSE'
    RLP_dir = 'Results/RLP'
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Check if the directory exists, and if not, create it
    if not os.path.exists(MSE_dir):
        os.makedirs(MSE_dir)
    # Check if the directory exists, and if not, create it
    if not os.path.exists(RLP_dir):
        os.makedirs(RLP_dir)
        
    # Path for the CSV files
    if eval_metric == 'MSE': 
        train_losses_path = f'Results/MSE/TRAIN_{DatasetName}_{task}_{loss_function}_TrainSize={train_size}_BatchSize={batch_size}_NumBatches={num_batches}_Epochs={epochs}_Iterations={iterations}_Shift={shift}_Noise={noise}.csv'
        test_losses_path = f'Results/MSE/TEST_{DatasetName}_{task}_{loss_function}_TrainSize={train_size}_BatchSize={batch_size}_NumBatches={num_batches}_Epochs={epochs}_Iterations={iterations}_Shift={shift}_Noise={noise}.csv'
    elif eval_metric == 'RLP':
        train_losses_path = f'Results/RLP/TRAIN_{DatasetName}_{task}_{loss_function}_TrainSize={train_size}_BatchSize={batch_size}_NumBatches={num_batches}_Epochs={epochs}_Iterations={iterations}_Shift={shift}_Noise={noise}.csv'
        test_losses_path = f'Results/RLP/TEST_{DatasetName}_{task}_{loss_function}_TrainSize={train_size}_BatchSize={batch_size}_NumBatches={num_batches}_Epochs={epochs}_Iterations={iterations}_Shift={shift}_Noise={noise}.csv'
    else:
        print('Error: Evaluation metric must be MSE or RLP')
        exit(1)
    
    # Save matrices to CSV
    np.savetxt(train_losses_path, train_losses, delimiter=',')
    np.savetxt(test_losses_path, test_losses, delimiter=',')
