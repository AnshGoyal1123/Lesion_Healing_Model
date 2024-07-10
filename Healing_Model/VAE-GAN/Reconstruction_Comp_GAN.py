import torch
import nibabel as nib
import numpy as np
import os
from VAE_GAN_Implementation import VAEGAN, load_model
from torch.utils.data import DataLoader
from LesionedData import LesionedDataset
import torch.nn.functional as F

def load_nifti(file_path):
    """Load a NIfTI file and return the data array."""
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine

def compare_voxels(img1, img2):
    """Compare two voxel arrays and return a difference map."""
    return np.abs(img1 - img2)

def threshold_difference(diff_map, threshold_multiplier=4):
    """Apply threshold to difference map to find significant differences."""
    threshold = np.mean(diff_map) + threshold_multiplier * np.std(diff_map)
    return diff_map > threshold

def save_difference_map(diff_map, affine, output_path):
    """Save the difference map as a NIfTI file."""
    new_image = nib.Nifti1Image(diff_map.astype(np.float32), affine)
    nib.save(new_image, output_path)

# Paths to the folders
lesioned_folder = '/home/agoyal19/Dataset/Dataset_5/images'
reconstructed_folder = '/home/agoyal19/Dataset/Reconstructions/reconstructed_images_5'
output_folder = '/home/agoyal19/Dataset/Difference_Maps_5'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of files in the lesioned folder
lesioned_files = [f for f in os.listdir(lesioned_folder) if f.endswith('.nii.gz')]

# Load the pre-trained model
model_path = '/home/agoyal19/My_Work/Healing_Model/Healthy_Models/healthy_vaegan.pth'
input_channels = 1  # Assuming grayscale brain images
hidden_channels = 32
z_dim = 64
model = VAEGAN(input_channels, hidden_channels, z_dim)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(model, model_path, device=device)
model.eval()

# Prepare your dataset
dataset = LesionedDataset(directory=lesioned_folder)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Processing and saving difference maps
for batch_data in data_loader:
    images = batch_data['ct'].to(device).float()
    filenames = batch_data['filename']

    with torch.no_grad():
        # Forward pass through the model
        x_recon, mean, logvar = model(images)

        # Save the reconstructed images
        for j in range(images.size(0)):
            reconstructed_img_data = x_recon[j, 0].cpu().numpy()
            original_name = filenames[j].replace('.nii.gz', '')  # Remove .nii.gz extension for new filename
            reconstructed_path = os.path.join(reconstructed_folder, f'{original_name}_reconstructed.nii.gz')
            img_nii = nib.Nifti1Image(reconstructed_img_data, affine=np.eye(4))  # Assuming no need for specific affine
            nib.save(img_nii, reconstructed_path)

            # Load the image data from files
            lesioned_scan, affine = load_nifti(os.path.join(lesioned_folder, filenames[j]))
            reconstructed_scan, _ = load_nifti(reconstructed_path)

            # Generate a difference map
            difference_map = compare_voxels(lesioned_scan, reconstructed_scan)

            # Threshold the difference map to find significant differences
            significant_diff = threshold_difference(difference_map)

            # Save the significant difference map as a NIfTI file
            output_path = os.path.join(output_folder, filenames[j])
            save_difference_map(significant_diff, affine, output_path)

print("Processing complete.")