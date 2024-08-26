import torch
from torch.utils.data import DataLoader
from LesionedData import LesionedDataset
from VQ_VAE_Implementation import VQVAE
from DDPM_Implementation import DDPM, UNet3D
import torch.nn.functional as F
import os
import nibabel as nib
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained VQ-VAE model
vqvae_model_path = '/home/agoyal19/My_Work/Healing_Model/Healthy_Models/healthy_vqvae.pth'
vqvae_model = VQVAE(input_channels=1, hidden_channels=64, embedding_dim=64, num_embeddings=256, commitment_cost=0.25)
vqvae_model.load_state_dict(torch.load(vqvae_model_path, map_location=device))
vqvae_model.to(device)
vqvae_model.eval()

# Load the DDPM model
ddpm_model = DDPM(
    betas=np.linspace(1e-4, 0.02, 1000),
    num_timesteps=1000,
    model=UNet3D(in_channels=64, out_channels=64),  # Use the 3D UNet
    device=device
)
ddpm_model.to(device)

# Prepare your dataset
dataset_directory = '/home/agoyal19/Dataset/Dataset_Full/images'
dataset = LesionedDataset(directory=dataset_directory)
data_loader = DataLoader(dataset, batch_size=2, shuffle=False)  # Reduced batch size to 1 to mitigate OOM

# Directory to save the reconstructed images in NIfTI format
save_directory = '/home/agoyal19/Dataset/Reconstructions/reconstructed_images_full_ddpm'
os.makedirs(save_directory, exist_ok=True)

# Validation and saving reconstructed images
total_loss = 0
with torch.no_grad():
    for batch_data in data_loader:
        images = batch_data['ct'].to(device).float()
        filenames = batch_data['filename']

        # Forward pass through the VQ-VAE encoder
        z = vqvae_model.encoder(images).float()

        # Forward diffusion process through DDPM
        z_noisy = ddpm_model.q_sample(z, t=torch.randint(0, ddpm_model.num_timesteps, (z.size(0),), device=device).long())

        # Reverse process through DDPM to denoise
        z_healed = ddpm_model.sample(z_noisy.shape)

        # Reconstruct the healed image using the VQ-VAE decoder
        reconstructed_images = vqvae_model.decoder(z_healed).float()

        # Calculate the reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed_images, images)
        total_loss += reconstruction_loss.item()

        # Save reconstructed and denoised images as .nii files
        for j in range(images.size(0)):
            img_data = reconstructed_images[j, 0].cpu().numpy()
            img_nii = nib.Nifti1Image(img_data, affine=np.eye(4))  # Assuming no need for specific affine
            original_name = filenames[j].replace('.nii.gz', '')  # Remove .nii.gz extension for new filename
            img_path = os.path.join(save_directory, f'{original_name}_reconstructed.nii.gz')
            nib.save(img_nii, img_path)

        # Clear cache after each batch to manage memory
        torch.cuda.empty_cache()

# Compute average loss
average_loss = total_loss / len(data_loader)
print(f'Average Validation Loss: {average_loss:.4f}')
