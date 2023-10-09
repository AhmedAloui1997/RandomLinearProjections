import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.classes import Autoencoder, VAE  # If these classes are in this module, adjust the import accordingly
from utils.process_data import balanced_batch_generator_auto
import numpy as np
from utils.measure_performance import calculate_fretchet
import torch.nn as nn
from ignite.metrics import FID
import os
import torchvision.utils as vutils

def save_images(images, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    # make a grid of images and save
    grid = vutils.make_grid(images, nrow=5, padding=2, normalize=True)
    vutils.save_image(grid, os.path.join(path, 'generated_images.png'))

def train_autoencoder(data, test,epochs, batch_size, model_type, loss_type, input_channels=1, image_size=28,device='cpu'):
    # Define model
    if model_type == 'Autoencoder':
        #model = Autoencoder(input_channels=input_channels, image_size=image_size)
        model = Autoencoder()
    elif model_type == 'VAE':
        model = VAE()  # Placeholder for VAE
    else:
        print('Error: Model type not recognized.')
        return

    if loss_type == 'MSE' or loss_type == 'MSEL2':
        # Define loss and optimizer
        if loss_type == 'MSE':
            distance = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        elif loss_type == 'MSEL2':
            distance = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        else:
            print('Error: Loss type not recognized.')
            return
        model = model.to(device)

        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:
                img, _ = batch  # For autoencoder, we don't use labels. The target is the input itself.
                img = img.to(device)
                img = Variable(img)
                output = model(img)
                loss = distance(output, img)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # You can print or log the epoch loss here if needed
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    else:
        model = model.to(device)
        # Get the total number of samples
        #X_train_tensor = torch.tensor(data, dtype=torch.float32)
        #num_samples = len(X_train_tensor)
        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        X_train_tensor = data.data.to(torch.float32)
        X_train_tensor = X_train_tensor.view(-1, 28*28) / 255

        X_test_tensor = test.data.to(torch.float32)
        X_test_tensor = X_test_tensor.view(-1, 28*28) / 255 
        N = 10 #28*28+1  # Define the batch size
        K = 100 # Define the number of batches to use per epoch

        # Randomly select N unique batches to use for each epoch
        unique_batches = list(balanced_batch_generator_auto(X_train_tensor, N, K))

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0

            for batch_X in unique_batches:
                # Forward pass: Compute predicted X by passing X to the model
                model.train()
                batch_X = batch_X.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)

                c = torch.linalg.lstsq(batch_X, batch_X).solution#torch.matmul(XTX_inv, XTy)
                #XTy_hat = torch.matmul(batch_X, outputs)#torch.linalg.lstsq(batch_X, outputs).solution
                c_pred = torch.linalg.lstsq(batch_X, outputs).solution #torch.matmul(XTX_inv, XTy_hat)
                #print(c_pred.shape)
        

                # You could use a custom loss here
                loss = distance(batch_X @ c_pred, batch_X @ c)
                #print(loss)

                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()

            model.eval()
            X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
            print(X_train_tensor.shape)
            prediction_train = model(X_train_tensor.to(device))
            prediction_test = model(X_test_tensor.to(device))
            
            train_perf = calculate_fretchet(prediction_train, X_train_tensor)
            #print(train_perf)
            test_perf = calculate_fretchet(prediction_test, X_test_tensor)
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {loss:.4f}, Train Performance: {train_perf:.4f}, Test Performance: {test_perf:.4f}')

            #print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {loss:.4f}')
    
    sample_images = prediction_test[:25].unsqueeze(1) / 255.0
    save_images(sample_images, 'generated images')
    
    return train_perf,test_perf  

# a function that will return results accross multiple iterations
def train_encoder_results(X_train, X_test, shift, train_size, task, iterations, epochs, batch_size, num_batches, loss_function,device):
    train_losses = np.zeros((iterations, epochs))
    test_losses = np.zeros((iterations, epochs)) 
    print(f"Training {task} model with {loss_function} loss")
    for i in range(iterations):
        print(f'Iteration [{i + 1}/{iterations}]:')
        train_perf,test_perf = train_autoencoder(X_train, X_test, epochs, batch_size, 'Autoencoder', 'RLP', input_channels=1, image_size=28,device=device)  
        train_losses[i,:] = train_perf
        test_losses[i,:] = test_perf
    return train_losses, test_losses