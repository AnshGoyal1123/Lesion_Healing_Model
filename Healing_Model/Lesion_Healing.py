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
model_path = '/home/agoyal19/My_Work/Healing_Model/Healthy_Models/healthy_vqvae.pth'
model = VQVAE(input_channels=1, hidden_channels=64, embedding_dim=64, num_embeddings=256, commitment_cost=0.25)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Prepare your dataset
#TODO: Update dataset size based on threshold........|
dataset_directory = '/home/agoyal19/Dataset/Dataset_Full/images'
dataset = LesionedDataset(directory=dataset_directory)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Directory to save the reconstructed images in NIfTI format
#TODO: Update dataset size based on threshold..................................|
save_directory = '/home/agoyal19/Dataset/Reconstructions/reconstructed_images_full'
os.makedirs(save_directory, exist_ok=True)

# Validation and saving reconstructed images
total_loss = 0
with torch.no_grad():
    for batch_data in data_loader:
        images = batch_data['ct'].to(device).float()
        filenames = batch_data['filename']

        # Forward pass through the model
        reconstructed_images, quantization_loss, _ = model(images)

        print(f'Reconstructed images shape: {reconstructed_images.shape}')  # Debug statement

        # Calculate the reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed_images, images)
        loss = reconstruction_loss + quantization_loss
        total_loss += loss.item()

        # Save reconstructed and denoised images as .nii files
        for j in range(images.size(0)):
            img_data = reconstructed_images[j, 0].cpu().numpy()
            img_nii = nib.Nifti1Image(img_data, affine=np.eye(4))  # Assuming no need for specific affine
            original_name = filenames[j].replace('.nii.gz', '')  # Remove .nii.gz extension for new filename
            img_path = os.path.join(save_directory, f'{original_name}_reconstructed.nii.gz')
            nib.save(img_nii, img_path)

# Compute average loss
average_loss = total_loss / len(data_loader)
print(f'Average Validation Loss: {average_loss:.4f}')
