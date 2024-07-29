import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Vector Quantizer Module
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, x):
        # Flatten input
        flat_input = x.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(hidden_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_channels, hidden_channels, embedding_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose3d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1)
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
        return torch.sigmoid(x)

# VQ-VAE-GAN Model
class VQVAEGAN(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAEGAN, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(input_channels, hidden_channels, embedding_dim)
        self.discriminator = Discriminator(input_channels, hidden_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, quantization_loss, perplexity = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, quantization_loss, perplexity

# Loss Functions
def vae_loss(x, x_recon, quantization_loss):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    return recon_loss + quantization_loss

def gan_loss(real_output, fake_output):
    real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss

# Training function
def train_vqvaegan(model, dataset, num_epochs=100, batch_size=4, lr=1e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae_optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.quantizer.parameters()), lr=lr)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_vae_loss = 0
        epoch_gan_loss = 0
        for batch_data in data_loader:
            images = batch_data['ct'].to(device).float()

            # VQ-VAE-GAN forward pass
            x_recon, quantization_loss, _ = model(images)
            
            # VAE loss
            recon_loss = vae_loss(images, x_recon, quantization_loss)
            
            # Discriminator forward pass
            real_output = model.discriminator(images)
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

            epoch_vae_loss += recon_loss.item()
            epoch_gan_loss += d_loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, VAE Loss: {epoch_vae_loss/len(data_loader)}, GAN Loss: {epoch_gan_loss/len(data_loader)}")

    return model

# Save model function
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model function
def load_model(model, path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")