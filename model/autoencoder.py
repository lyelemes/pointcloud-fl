import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudAutoencoder(nn.Module):
    def __init__(self, point_size=600, latent_size=1024):  # Paper uses 1024
        super(PointCloudAutoencoder, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        # ENCODER 
        self.conv1 = nn.Conv1d(11, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, latent_size, 1)

        # Activations
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act4 = nn.PReLU()

        # DECODER
        self.fc1 = nn.Linear(self.latent_size, 512)       # Use self.latent_size!
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, point_size * 3)
       
    def encoder(self, x): 
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.conv5(x)
        x = torch.max(x, 2, keepdim=True)[0]  
        x = x.view(-1, self.latent_size)                  # Use self.latent_size!
        return x
    
    def decoder(self, x):
        # MLP with BatchNorm like paper
        x = F.relu(self.fc1(x))    # BatchNorm + ReLU
        x = F.relu(self.fc2(x))     # BatchNorm + ReLU
        x = self.fc3(x)                       # No activation on final layer
        return x
    
    def forward(self, x):
        """
        Args:
            x: [batch, 600, 11] - 600 points with 11 features each
        Returns:
            reconstructed: [batch, 600, 3] - XYZ coordinates only
            latent: [batch, 1024] - latent vector (fixed 1024)
        """
        # Permute to [batch, 11, 600] for Conv1d
        x = x.permute(0, 2, 1)
        
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.view(-1, self.point_size, 3)
        
        return reconstructed, latent.view(-1, 1, 32, 32)