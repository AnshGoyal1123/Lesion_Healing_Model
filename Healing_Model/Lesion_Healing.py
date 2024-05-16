import torch
from torch.utils.data import DataLoader
from LesionedData import LesionedDataset
from VQ_VAE_Implementation import VQVAE
import torch.nn.functional as F
import os
import nibabel as nib
import numpy as np
import bm3d

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model_path = '/home/agoyal19/My_Work/Lesion_Healing_Model/Healing_Model/Healthy_Models/healthy_vqvae.pth'
model = VQVAE(input_channels=1, hidden_channels=64, embedding_dim=64, num_embeddings=256, commitment_cost=0.25)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Prepare your dataset
dataset_directory = '/home/agoyal19/Dataset/data/images/'
dataset = LesionedDataset(directory=dataset_directory)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Directory to save the reconstructed images in NIfTI format
save_directory = '/home/agoyal19/Dataset/reconstructed_images_bm3d/'
os.makedirs(save_directory, exist_ok=True)

# Function to apply BM3D denoising
def apply_bm3d_to_tensor(tensor, sigma_psd=30):
    tensor_np = tensor.cpu().numpy()
    denoised_tensor_np = bm3d.bm3d(tensor_np, sigma_psd=sigma_psd)
    return torch.from_numpy(denoised_tensor_np).to(tensor.device)

# Validation and saving reconstructed images
total_loss = 0
with torch.no_grad():
    for batch_data in data_loader:
        images = batch_data['ct'].to(device).float()
        filenames = batch_data['filename']

        # Forward pass through the model
        reconstructed_images, quantization_loss, _ = model(images)

        # Denoise using BM3D
        reconstructed_images_denoised = apply_bm3d_to_tensor(reconstructed_images)

        # Calculate the reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed_images_denoised, images)
        loss = reconstruction_loss + quantization_loss
        total_loss += loss.item()

        # Save reconstructed and denoised images as .nii files
        for j in range(images.size(0)):
            img_data = reconstructed_images_denoised[j, 0].cpu().numpy()
            img_nii = nib.Nifti1Image(img_data, affine=np.eye(4))  # Assuming no need for specific affine
            original_name = filenames[j].replace('.nii.gz', '')  # Remove .nii.gz extension for new filename
            img_path = os.path.join(save_directory, f'{original_name}_reconstructed_denoised.nii.gz')
            nib.save(img_nii, img_path)

# Compute average loss
average_loss = total_loss / len(data_loader)
print(f'Average Validation Loss: {average_loss:.4f}')
