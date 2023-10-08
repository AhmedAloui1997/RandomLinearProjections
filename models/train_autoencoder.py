import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.classes import Autoencoder, VAE  # If these classes are in this module, adjust the import accordingly
from utils.process_data import balanced_batch_generator_auto
import numpy as np
from utils import measure_performance
import torch.nn as nn

def train_autoencoder(data, test,epochs, batch_size, model_type, loss_type, input_channels=1, image_size=28):
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

        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:
                img, _ = batch  # For autoencoder, we don't use labels. The target is the input itself.
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
        # Get the total number of samples
        #X_train_tensor = torch.tensor(data, dtype=torch.float32)
        #num_samples = len(X_train_tensor)
        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        X_train_tensor = data.data.to(torch.float32)
        X_train_tensor = X_train_tensor.view(-1, 28*28) / 255

        X_test_tensor = test.data.to(torch.float32)
        X_test_tensor = X_test_tensor.view(-1, 28*28) / 255 
        N = 28*28+1  # Define the batch size
        K = 10000  # Define the number of batches to use per epoch

        # Randomly select N unique batches to use for each epoch
        unique_batches = list(balanced_batch_generator_auto(X_train_tensor, N, K))

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0

            for batch_X in unique_batches:
                # Forward pass: Compute predicted X by passing X to the model
                model.train()
                
                optimizer.zero_grad()
                outputs = model(batch_X)

                #c = torch.linalg.lstsq(batch_X, batch_X).solution
                #XTX_inv = torch.inverse(torch.matmul(batch_X, batch_X.t()))
                #XTy = torch.matmul(batch_X, batch_X.t())#torch.linalg.lstsq(batch_X, outputs).solution
                c = torch.linalg.lstsq(batch_X, batch_X).solution#torch.matmul(XTX_inv, XTy)
                #XTy_hat = torch.matmul(batch_X, outputs)#torch.linalg.lstsq(batch_X, outputs).solution
                c_pred = torch.linalg.lstsq(batch_X, outputs).solution #torch.matmul(XTX_inv, XTy_hat)
                #print(c_pred.shape)
        

                # You could use a custom loss here
                loss = distance(batch_X @ c_pred, batch_X @ c)
                print(loss)

                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()

            model.eval()
            prediction_train = model(X_train_tensor)
            prediction_test = model(X_test_tensor)
            train_perf = measure_performance(model,prediction_train, data)
            test_perf = measure_performance(model,prediction_test, test)
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {loss:.4f}, Train Performance: {train_perf:.4f}, Test Performance: {test_perf:.4f}')

            #print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f},    Test Loss: {loss:.4f}')
    
    
    return train_perf  

# a function that will return results accross multiple iterations
def train_encoder_results(X_train, X_test, shift, train_size, task, iterations, epochs, batch_size, num_batches, loss_function):
    train_losses = np.zeros((iterations, epochs))
    test_losses = np.zeros((iterations, epochs)) 
    for i in range(1):
        perf = train_autoencoder(X_train, X_test, epochs, batch_size, 'Autoencoder', 'RLP', input_channels=1, image_size=28)  
    return 0.0, 0.0