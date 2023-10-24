import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.classes import Autoencoder
from utils.process_data import balanced_batch_generator_auto
import numpy as np
import torch.nn as nn
import os
import torchvision.utils as vutils

# function to save images
def save_images(images, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    # make a grid of images and save
    grid = vutils.make_grid(images, nrow=5, padding=2, normalize=True)
    vutils.save_image(grid, os.path.join(path, 'generated_images.png'))

def train_autoencoder(dataset_name,train_size,data, test,epochs,batch_size, num_batches, model_type, loss_type,device='cpu'):
    # Define model
    print(f"Training {model_type} model with {loss_type} loss")
    train_perf = np.zeros(epochs)
    test_perf = np.zeros(epochs)
    if model_type == 'Autoencoder':
        #model = Autoencoder(input_channels=input_channels, image_size=image_size)
        if dataset_name == 'MNIST':
            model = Autoencoder(28*28)
        elif dataset_name == 'CIFAR10':
            model = Autoencoder(32*32*3)
    else:
        print('Error: Model type not recognized.')
        return

    if loss_type == 'MSE' or loss_type == 'MSEL2':
        print(f"Training {model_type} model with {loss_type} loss")
        # Define loss and optimizer
        if loss_type == 'MSE':
            distance = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        elif loss_type == 'MSEL2':
            distance = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        else:
            print('Error: Loss function not recognized.')
            return
        model = model.to(device)

        try: 
            X_train_tensor = data.data.to(torch.float32)
            X_train_tensor = X_train_tensor.view(-1, 28*28) / 255
        except:
            # upload cifar 10 to change
            input_channels=3
            X_train_tensor = data.data.to(torch.float32)
            X_train_tensor = X_train_tensor.view(-1, 32*32 * input_channels) / 255.0

        X_train_tensor = X_train_tensor[:train_size]

        X_test_tensor = test.data.to(torch.float32)
        try: 
            X_test_tensor = X_test_tensor.view(-1, 28*28) / 255.0
        except:
            # upload cifar 10 to change
            input_channels=3
            X_test_tensor = X_test_tensor.view(-1, 32*32*input_channels) / 255.0
        X_test_tensor = X_test_tensor[:1000]
        dataloader = DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                img = batch  # For autoencoder, we don't use labels. The target is the input itself.
                img = img.to(device)
                #img = Variable(img)
                output = model(img)
                loss = distance(output, img)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataloader)

            # You can print or log the epoch loss here if needed
            #print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
            model.eval()
            X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
            #prediction_train = model(X_train_tensor.to(device))
            prediction_test = model(X_test_tensor.to(device))
 
            train_perf[epoch] = epoch_loss
            test_perf_ = distance(prediction_test, X_test_tensor.to(device))
            test_perf[epoch] = test_perf_
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss}, Test Performance: {test_perf_}')

    else:
        model = model.to(device)

        distance = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=.99)
        X_train_tensor = data.data.to(torch.float32)
        try: 
            X_train_tensor = data.data.to(torch.float32)
            X_train_tensor = X_train_tensor.view(-1, 28*28) / 255
        except:
            # upload cifar 10 to change
            X_train_tensor = data.data.to(torch.float32)
            X_train_tensor = X_train_tensor.view(-1, 32*32*3) / 255

        #X_train_tensor = X_train_tensor[:5000]

        X_test_tensor = test.data.to(torch.float32)
        try: 
            X_test_tensor = X_test_tensor.view(-1, 28*28) / 255
        except:
            # upload cifar 10 to change
            X_test_tensor = X_test_tensor.view(-1, 32*32*3) / 255 
        #X_train_tensor = X_train_tensor.view(-1, 28*28) / 255
        # use just train_size samples for training
        X_train_tensor = X_train_tensor[:train_size]
        X_test_tensor = X_test_tensor[:1000]


        N = batch_size #28*28+1  # Define the batch size
        K = num_batches # Define the number of batches to use per epoch

        unique_batches = list(balanced_batch_generator_auto(X_train_tensor, N, K))
        print("Unique batches generated")

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0

            for batch_X in unique_batches:
                # Forward pass: Compute predicted X by passing X to the model
                model.train()
                batch_X = batch_X.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                c = torch.eye(batch_X.shape[1], device=device)
                c_pred =  torch.linalg.pinv(batch_X.T @ batch_X) @ batch_X.T @ outputs
                
                loss = distance(batch_X @ c_pred, batch_X @ c)

                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            epoch_loss /= len(unique_batches)

            model.eval()
            X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
            prediction_train = model(X_train_tensor.to(device))
            prediction_test = model(X_test_tensor.to(device))
            
            train_perf[epoch] = epoch_loss
            test_perf_ = distance(prediction_test, X_test_tensor.to(device))
            test_perf[epoch] = test_perf_
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss}, Test Performance: {test_perf_}')

    return train_perf,test_perf  

# A function that will return results accross multiple iterations
def train_encoder_results(dataset_name,train_size,X_train, X_test, task, iterations, epochs, batch_size, num_batches, loss_function,device):
    train_losses = np.zeros((iterations, epochs))
    test_losses = np.zeros((iterations, epochs)) 
    print(f"Training {task} model with {loss_function} loss")
    for i in range(iterations):
        print(f'Iteration [{i + 1}/{iterations}]:')
        train_perf,test_perf = train_autoencoder(dataset_name, train_size, X_train, X_test, epochs, num_batches,batch_size, 'Autoencoder', loss_function,device=device)  
        train_losses[i,:] = train_perf
        test_losses[i,:] = test_perf
    return train_losses, test_losses
