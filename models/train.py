import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.classes import RegressionModel, Autoencoder
from utils.process_data import SplitDataset
from utils.process_data import balanced_batch_generator_reg, balanced_batch_generator_auto

def train(X, y, shift, train_size, task, iterations, epochs, batch_size, num_batches, loss_function):
    train_losses = np.zeros((iterations, epochs))
    test_losses = np.zeros((iterations, epochs))
    
    for i in range(iterations):
        # Divide the dataset into train and test
        X_train, X_test, y_train, y_test = SplitDataset(X, y, shift, train_size)
        
        if task == 'Regression':
            # Convert numpy arrays to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            
            # Instantiating Model
            input_size = X_train.shape[1]
            hidden_size = 32
            output_size = 1
            model = RegressionModel(input_size, hidden_size, output_size)
            criterion = nn.MSELoss()
            
            if loss_function == 'MSE':  
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                loss_train, loss_test = train_mse_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion)
            
            elif loss_function == 'MSEL2':  
                optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
                loss_train, loss_test = train_mse_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion)
  
            elif loss_function == 'RLP':
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                loss_train, loss_test = train_rlp_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, num_batches, model, optimizer, criterion)
                
    
        elif task == 'Autoencoder':
            pass
        
        else:
            print('Error: Task must be regression or autoencoder')
            
        train_losses[i,:], test_losses[i,:] = loss_train, loss_test
    
    return train_losses, test_losses
        
def train_mse_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion):
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    
    # Instantiate train and test loss
    loss_train = np.zeros(epochs)
    loss_test = np.zeros(epochs)

    for epoch in range(epochs):
        epoch_loss = 0
    
        for batch_X, batch_y in train_dataloader:
            model.train()
    
            # Zero the gradients
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) # Simple MSE Loss
    
            # Backward pass
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        model.eval()
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).detach().numpy()
    
        epoch_loss /= len(train_dataloader)
        loss_train[epoch] = epoch_loss
        loss_test[epoch] = test_loss
        print(f'Iteration [{i + 1}/{iterations}],    Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {test_loss:.4f}')
        
    return loss_train, loss_test

def train_rlp_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, num_batches, model, optimizer, criterion):    
    # Randomly select N unique batches to use for each epoch
    unique_batches = list(balanced_batch_generator_reg(X_train_tensor, y_train_tensor, batch_size, num_batches))
    
    # Instantiate train and test loss
    loss_train = np.zeros(epochs)
    loss_test = np.zeros(epochs)

    for epoch in range(epochs):
        epoch_loss = 0
    
        for batch_X, batch_y in unique_batches:
            model.train()
    
            # Zero the gradients
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            c = torch.linalg.lstsq(batch_X, batch_y).solution
            c_pred = torch.linalg.lstsq(batch_X, outputs).solution
            loss = criterion(batch_X @ c_pred, batch_X @ c) # RLP Loss
    
            # Backward pass
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        model.eval()
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).detach().numpy()
    
        epoch_loss /= num_batches
        loss_train[epoch] = epoch_loss
        loss_test[epoch] = test_loss
        print(f'Iteration [{i + 1}/{iterations}],    Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {test_loss:.4f}')
        
    return loss_train, loss_test