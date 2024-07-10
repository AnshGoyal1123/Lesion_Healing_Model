import torch
from torch.utils.data import DataLoader
from LesionedData import LesionedDataset
from VAE_GAN_Implementation import VAEGAN, load_model
import torch.nn.functional as F
import os
import nibabel as nib
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model_path = '/home/agoyal19/My_Work/Healing_Model/Healthy_Models/healthy_vaegan.pth'
input_channels = 1  # Assuming grayscale brain images
hidden_channels = 32
z_dim = 64
model = VAEGAN(input_channels, hidden_channels, z_dim)
model = load_model(model, model_path, device=device)
model.eval()

# Prepare your dataset
dataset_directory = '/home/agoyal19/Dataset/Dataset_Full/images'
dataset = LesionedDataset(directory=dataset_directory)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Directory to save the reconstructed images in NIfTI format
save_directory = '/home/agoyal19/Dataset/Reconstructions/reconstructed_images_full'
os.makedirs(save_directory, exist_ok=True)

# Validation and saving reconstructed images
total_loss = 0
with torch.no_grad():
    for batch_data in data_loader:
        images = batch_data['ct'].to(device).float()
        filenames = batch_data['filename']

        # Forward pass through the model
        x_recon, mean, logvar = model(images)

        # Calculate the reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(x_recon, images)
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = reconstruction_loss + kld_loss
        total_loss += loss.item()

        # Save reconstructed images as .nii files
        for j in range(images.size(0)):
            img_data = x_recon[j, 0].cpu().numpy()
            img_nii = nib.Nifti1Image(img_data, affine=np.eye(4))  # Assuming no need for specific affine
            original_name = filenames[j].replace('.nii.gz', '')  # Remove .nii.gz extension for new filename
            img_path = os.path.join(save_directory, f'{original_name}_reconstructed.nii.gz')
            nib.save(img_nii, img_path)

# Compute average loss
average_loss = total_loss / len(data_loader)
print(f'Average Validation Loss: {average_loss:.4f}')