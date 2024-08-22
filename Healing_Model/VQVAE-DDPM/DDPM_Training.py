import torch
from torch.utils.data import DataLoader
from HealthyData import HealthyDataset
from VQ_VAE_Implementation import VQVAE
from DDPM_Implementation import DDPM, UNet3D  # Use the new 3D UNet
import torch.nn.functional as F
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained VQ-VAE model (for encoding the images)
vqvae_model_path = '/home/agoyal19/My_Work/Healing_Model/Healthy_Models/healthy_vqvae.pth'
vqvae_model = VQVAE(input_channels=1, hidden_channels=64, embedding_dim=64, num_embeddings=256, commitment_cost=0.25)
vqvae_model.load_state_dict(torch.load(vqvae_model_path, map_location=device))
vqvae_model.to(device)
vqvae_model.eval()

# Initialize DDPM model
ddpm_model = DDPM(
    betas=np.linspace(1e-4, 0.02, 1000),
    num_timesteps=1000,
    model=UNet3D(in_channels=64, out_channels=64),  # Use the 3D UNet here
    device=device
)
ddpm_model.to(device)

# Prepare dataset of latent representations from healthy data
dataset_directory = '/home/agoyal19/Dataset/control_data'
dataset = HealthyDataset(directory=dataset_directory)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(ddpm_model.parameters(), lr=1e-4)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_data in data_loader:
        images = batch_data['ct'].to(device).float()
        
        # Encode images to latent space using the VQ-VAE encoder
        z = vqvae_model.encoder(images)

        # Sample random timesteps
        t = torch.randint(0, ddpm_model.num_timesteps, (z.size(0),), device=device).long()
        
        # Generate noisy versions of z
        z_noisy = ddpm_model.q_sample(z, t)
        
        # Predict the noise added
        predicted_noise = ddpm_model.model(z_noisy)
        
        # Compute the loss (how close the predicted noise is to the actual noise)
        loss = F.mse_loss(predicted_noise, z_noisy - z)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(data_loader)}')

# Save the trained DDPM model
ddpm_save_path = "/home/agoyal19/My_Work/Healing_Model/DDPM_Models/ddpm_model.pth"
torch.save(ddpm_model.state_dict(), ddpm_save_path)
print(f"DDPM model saved to {ddpm_save_path}")