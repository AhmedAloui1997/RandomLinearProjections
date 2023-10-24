import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.classes import RegressionModel
from utils.process_data import SplitDataset, add_gaussian_noise
from utils.process_data import balanced_batch_generator_reg

def train(X, y, eval_metric, noise, shift, train_size, task, iterations, epochs, batch_size, num_batches, loss_function):
    train_losses = np.zeros((iterations, epochs))
    test_losses = np.zeros((iterations, epochs))
    
    for i in range(iterations):     
        # Divide the dataset into train and test
        X_train, X_test, y_train, y_test = SplitDataset(X, y, shift, train_size)
        
        if noise >= 0:
            # If noise mean > 0, add gaussian noise
            X_train = add_gaussian_noise(X_train, noise)
        
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
        learning_rate = 0.0001
        
        if loss_function == 'MSE':  
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            loss_train, loss_test = train_mse_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion, eval_metric)
        
        elif loss_function == 'MSEL2':  
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            loss_train, loss_test = train_mse_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion, eval_metric)

        elif loss_function == 'RLP':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            loss_train, loss_test = train_rlp_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, num_batches, model, optimizer, criterion, eval_metric)
        
        elif loss_function == 'MIXUP':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            loss_train, loss_test = train_mixup_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion, eval_metric)
                      
        elif loss_function == 'RLPMIX':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            loss_train, loss_test = train_rlpmix_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, num_batches, model, optimizer, criterion, eval_metric)
        else:
            print('Error: Loss function not recognized.')
            exit(1)
            
        train_losses[i,:], test_losses[i,:] = loss_train, loss_test

    return train_losses, test_losses

        
# Training Regression Neural Network with MSE loss
def train_mse_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion, eval_metric):
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    
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
        
        # Using MSE to measure test performance
        if eval_metric == 'MSE':
            test_loss = criterion(test_outputs, y_test_tensor).detach().numpy()
        # Using RLP to measure test performance
        elif eval_metric == 'RLP':
            # c = torch.linalg.pinv((X_test_tensor.T @ X_test_tensor)) @ (X_test_tensor.T @ y_test_tensor)
            # c_pred = torch.linalg.pinv((X_test_tensor.T @ X_test_tensor)) @ (X_test_tensor.T @ test_outputs)
            c = torch.linalg.lstsq(X_test_tensor, y_test_tensor).solution
            c_pred = torch.linalg.lstsq(X_test_tensor, test_outputs).solution
            test_loss = criterion(X_test_tensor @ c_pred, X_test_tensor @ c).detach().numpy()
        else:
            print('Error: Evaluation metric must be MSE or RLP')
            exit(1)
    
        epoch_loss /= len(train_dataloader)
        loss_train[epoch] = epoch_loss
        loss_test[epoch] = test_loss
        print(f'Iteration [{i + 1}/{iterations}],    Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {test_loss:.4f}')
        
    return loss_train, loss_test


# Training Regression Neural Network with RLP loss
def train_rlp_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, num_batches, model, optimizer, criterion, eval_metric):    
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
            
            # c = torch.linalg.pinv((batch_X.T @ batch_X)) @ (batch_X.T @ batch_y)
            # c = torch.linalg.pinv((batch_X.T @ batch_X)) @ (batch_X.T @ outputs)
            c = torch.linalg.lstsq(batch_X, batch_y).solution
            c_pred = torch.linalg.lstsq(batch_X, outputs).solution
            loss = criterion(batch_X @ c_pred, batch_X @ c) # RLP Loss
    
            # Backward pass
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        model.eval()
        test_outputs = model(X_test_tensor)
        
        # Using MSE to measure test performance
        if eval_metric == 'MSE':
            test_loss = criterion(test_outputs, y_test_tensor).detach().numpy()
        # Using RLP to measure test performance
        elif eval_metric == 'RLP':
            # c = torch.linalg.pinv((X_test_tensor.T @ X_test_tensor)) @ (X_test_tensor.T @ y_test_tensor)
            # c_pred = torch.linalg.pinv((X_test_tensor.T @ X_test_tensor)) @ (X_test_tensor.T @ test_outputs)
            c = torch.linalg.lstsq(X_test_tensor, y_test_tensor).solution
            c_pred = torch.linalg.lstsq(X_test_tensor, test_outputs).solution
            test_loss = criterion(X_test_tensor @ c_pred, X_test_tensor @ c).detach().numpy()
        else:
            print('Error: Evaluation metric must be MSE or RLP')
            exit(1)
    
        epoch_loss /= num_batches
        loss_train[epoch] = epoch_loss
        loss_test[epoch] = test_loss
        print(f'Iteration [{i + 1}/{iterations}],    Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {test_loss:.4f}')
        
    return loss_train, loss_test


# Training Regression Neural Network with mixup-augmented MSE
def train_mixup_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, model, optimizer, criterion, eval_metric):
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    train_dataloader_2 = DataLoader(train_dataset, batch_size, shuffle=True)
    
    # Instantiate train and test loss
    loss_train = np.zeros(epochs)
    loss_test = np.zeros(epochs)

    for epoch in range(epochs):
        epoch_loss = 0
        
        # y1, y2 should be one-hot vectors
        for (x1, y1), (x2, y2) in zip(train_dataloader, train_dataloader_2):
            if x1.size() != x2.size(): continue         
            model.train()
            
            alpha = 0.25
            lam = np.random.beta(alpha, alpha)
            x = lam * x1 + (1. - lam) * x2
            y = lam * y1 + (1. - lam) * y2
            
            # Zero the gradients
            optimizer.zero_grad()  
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        model.eval()
        test_outputs = model(X_test_tensor)
        
        # Using MSE to measure test performance
        if eval_metric == 'MSE':
            test_loss = criterion(test_outputs, y_test_tensor).detach().numpy()
        # Using RLP to measure test performance
        elif eval_metric == 'RLP':
            # c = torch.linalg.pinv((X_test_tensor.T @ X_test_tensor)) @ (X_test_tensor.T @ y_test_tensor)
            # c_pred = torch.linalg.pinv((X_test_tensor.T @ X_test_tensor)) @ (X_test_tensor.T @ test_outputs)
            c = torch.linalg.lstsq(X_test_tensor, y_test_tensor).solution
            c_pred = torch.linalg.lstsq(X_test_tensor, test_outputs).solution
            test_loss = criterion(X_test_tensor @ c_pred, X_test_tensor @ c).detach().numpy()
        else:
            print('Error: Evaluation metric must be MSE or RLP')
            exit(1)
    
        epoch_loss /= len(list(zip(train_dataloader, train_dataloader)))
        loss_train[epoch] = epoch_loss
        loss_test[epoch] = test_loss
        print(f'Iteration [{i + 1}/{iterations}],    Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {test_loss:.4f}')
        
    return loss_train, loss_test


# Training Regression Neural Network with mixup-augmented RLP
def train_rlpmix_reg(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterations, i, epochs, batch_size, num_batches, model, optimizer, criterion, eval_metric):    
    # Randomly select N unique batches to use for each epoch
    unique_batches = list(balanced_batch_generator_reg(X_train_tensor, y_train_tensor, batch_size, num_batches))
    unique_batches_2 = list(balanced_batch_generator_reg(X_train_tensor, y_train_tensor, batch_size, num_batches))
    
    # Instantiate train and test loss
    loss_train = np.zeros(epochs)
    loss_test = np.zeros(epochs)

    for epoch in range(epochs):
        epoch_loss = 0
        
        # y1, y2 should be one-hot vectors
        for (x1, y1), (x2, y2) in zip(unique_batches, unique_batches_2):
            # Adapting Balanced Batch Generator for mixup
            model.train()
            
            alpha = 0.25
            lam = np.random.beta(alpha, alpha)
            x = lam * x1 + (1. - lam) * x2
            y = lam * y1 + (1. - lam) * y2
            
            # Zero the gradients
            optimizer.zero_grad()  
            outputs = model(x)
            
            c = torch.linalg.lstsq(x, y).solution
            c_pred = torch.linalg.lstsq(x, outputs).solution
            loss = criterion(x @ c_pred, x @ c) # RLP Loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        model.eval()
        test_outputs = model(X_test_tensor)
        
        # Using MSE to measure test performance
        if eval_metric == 'MSE':
            test_loss = criterion(test_outputs, y_test_tensor).detach().numpy()
            
        # Using RLP to measure test performance
        elif eval_metric == 'RLP':
            # c = torch.linalg.pinv((X_test_tensor.T @ X_test_tensor)) @ (X_test_tensor.T @ y_test_tensor)
            # c_pred = torch.linalg.pinv((X_test_tensor.T @ X_test_tensor)) @ (X_test_tensor.T @ test_outputs)
            c = torch.linalg.lstsq(X_test_tensor, y_test_tensor).solution
            c_pred = torch.linalg.lstsq(X_test_tensor, test_outputs).solution
            test_loss = criterion(X_test_tensor @ c_pred, X_test_tensor @ c).detach().numpy()
        else:
            print('Error: Evaluation metric must be MSE or RLP')
            exit(1)
    
        epoch_loss /= num_batches
        loss_train[epoch] = epoch_loss
        loss_test[epoch] = test_loss
        print(f'Iteration [{i + 1}/{iterations}],    Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {test_loss:.4f}')
        
    return loss_train, loss_test