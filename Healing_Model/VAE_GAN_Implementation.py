import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(hidden_channels, z_dim*2, kernel_size=3, stride=1, padding=1) # z_dim*2 for mean and logvar

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        mean, logvar = torch.chunk(x, 2, dim=1) # Split into mean and logvar
        return mean, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_channels, hidden_channels, z_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose3d(z_dim, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose3d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return torch.sigmoid(x)  # Use sigmoid to constrain output to [0, 1]

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(hidden_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x)
        return torch.sigmoid(x)  # Use sigmoid to constrain output to [0, 1]

# VAE-GAN Model
class VAEGAN(nn.Module):
    def __init__(self, input_channels, hidden_channels, z_dim):
        super(VAEGAN, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels, z_dim)
        self.decoder = Decoder(input_channels, hidden_channels, z_dim)
        self.discriminator = Discriminator(input_channels, hidden_channels)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

# Loss Functions
def vae_loss(x, x_recon, mean, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def gan_loss(real_output, fake_output):
    real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss

# Instantiate the model
input_channels = 1  # Assuming grayscale brain images
hidden_channels = 32
z_dim = 64
model = VAEGAN(input_channels, hidden_channels, z_dim)

# Optimizers
vae_optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-4)
discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)

# Training Loop (Example)
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:  # Assuming dataloader is defined elsewhere
        # Get input data
        x = data.to(device)
        
        # VAE-GAN forward pass
        x_recon, mean, logvar = model(x)
        
        # VAE loss
        recon_loss = vae_loss(x, x_recon, mean, logvar)
        
        # Discriminator forward pass
        real_output = model.discriminator(x)
        fake_output = model.discriminator(x_recon.detach())
        
        # GAN loss
        d_loss = gan_loss(real_output, fake_output)
        
        # Update VAE
        vae_optimizer.zero_grad()
        recon_loss.backward()
        vae_optimizer.step()
        
        # Update Discriminator
        discriminator_optimizer.zero_grad()
        d_loss.backward()
        discriminator_optimizer.step()

    print(f"Epoch {epoch + 1}, VAE Loss: {recon_loss.item()}, GAN Loss: {d_loss.item()}")

