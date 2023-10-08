import torch
import torch.nn as nn

# Regression Neural Network
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,encoding_dim),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 28*28),
            nn.Sigmoid(),  # To get a range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
      
# class Autoencoder(nn.Module):
#     def __init__(self, input_channels=1, image_size=28):
#         super(Autoencoder, self).__init__()

#         # Compute intermediate sizes
#         size_after_conv1 = image_size - 4  # considering kernel=5 and stride=1
#         size_after_conv2 = size_after_conv1 - 4
#         flattened_size = size_after_conv2 * size_after_conv2 * 8
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_channels, 4, kernel_size=5),
#             nn.ReLU(True),
#             nn.Conv2d(4, 8, kernel_size=5),
#             nn.ReLU(True),
#             nn.Flatten(),
#             nn.Linear(flattened_size, 10),
#             nn.Softmax()
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(10, 400),
#             nn.ReLU(True),
#             nn.Linear(400, flattened_size),
#             nn.ReLU(True),
#             nn.Unflatten(1, (8, size_after_conv2, size_after_conv2)),
#             nn.ConvTranspose2d(8, 4, kernel_size=5),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(4, input_channels, kernel_size=5),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         enc = self.encoder(x)
#         dec = self.decoder(enc)
#         return dec




class VAE(nn.Module):
    def __init__(self, input_channels=1, image_size=28):
        super(Autoencoder, self).__init__()

        # Compute intermediate sizes
        size_after_conv1 = image_size - 4  # considering kernel=5 and stride=1
        size_after_conv2 = size_after_conv1 - 4
        flattened_size = size_after_conv2 * size_after_conv2 * 8
    
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(flattened_size, 10),
            nn.Softmax()
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(True),
            nn.Linear(400, flattened_size),
            nn.ReLU(True),
            nn.Unflatten(1, (8, size_after_conv2, size_after_conv2)),
            nn.ConvTranspose2d(8, 4, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, input_channels, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
