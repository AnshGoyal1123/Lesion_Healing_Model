import wandb
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize wandb
wandb.init(project='lesion_detection')

def load_nifti(file_path):
    """Load a NIfTI file and return the data array."""
    nii = nib.load(file_path)
    return nii.get_fdata()

def compare_voxels(img1, img2):
    """Compare two voxel arrays and return a difference map."""
    return np.abs(img1 - img2)

def plot_difference(diff_map, filename):
    """Plot the difference map."""
    plt.figure(figsize=(12, 6))
    plt.imshow(np.max(diff_map, axis=0), cmap='hot')
    plt.colorbar()
    plt.title(f'Difference Map - {filename} - Max Intensity Projection')
    plt.show()
    
    # Log the plot to W&B
    wandb.log({f"difference_map_{filename}": wandb.Image(plt)})
    plt.close()

# Paths to the folders
healthy_folder = '/home/agoyal19/Dataset/data/images'
reconstructed_folder = '/home/agoyal19/Dataset/reconstructed_images'

# Get the list of files in the healthy folder
healthy_files = [f for f in os.listdir(healthy_folder) if f.endswith('.nii.gz')]

# Loop through each file in the healthy folder
for file in healthy_files:
    healthy_path = os.path.join(healthy_folder, file)
    # Construct the path for the corresponding reconstructed file
    reconstructed_file = file.replace('.nii.gz', '_reconstructed.nii.gz')
    reconstructed_path = os.path.join(reconstructed_folder, reconstructed_file)

    # Check if the reconstructed file exists
    if os.path.exists(reconstructed_path):
        # Load the image data from files
        normal_scan = load_nifti(healthy_path)
        lesion_scan = load_nifti(reconstructed_path)

        # Generate a difference map
        difference_map = compare_voxels(normal_scan, lesion_scan)

        # Plot and log the difference map
        plot_difference(difference_map, file[:-7])  # Remove extension for naming
    else:
        print(f"No corresponding file found for {file}")
