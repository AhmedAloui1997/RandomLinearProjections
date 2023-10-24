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
    def __init__(self, dim,encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim,encoding_dim),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, dim),
            nn.Sigmoid(),  # To get a range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


