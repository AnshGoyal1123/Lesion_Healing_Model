import nibabel as nib
import numpy as np
import os
from scipy.ndimage import label

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
#TODO: Update dataset size based on threshold
lesioned_folder = '/home/agoyal19/Dataset/Dataset_5/images'
#TODO: Update dataset size based on threshold
reconstructed_folder = '/home/agoyal19/Dataset/Reconstructions/reconstructed_images_5'
#TODO: Update dataset size based on threshold
output_folder = '/home/agoyal19/Dataset/Difference_Maps_5'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of files in the lesioned folder
lesioned_files = [f for f in os.listdir(lesioned_folder) if f.endswith('.nii.gz')]

# Loop through each file in the lesioned folder
for file in lesioned_files:
    lesioned_path = os.path.join(lesioned_folder, file)
    # Construct the path for the corresponding reconstructed file
    reconstructed_file = file.replace('.nii.gz', '_reconstructed.nii.gz')
    reconstructed_path = os.path.join(reconstructed_folder, reconstructed_file)

    # Check if the reconstructed file exists
    if os.path.exists(reconstructed_path):
        # Load the image data from files
        lesioned_scan, affine = load_nifti(lesioned_path)
        reconstructed_scan, _ = load_nifti(reconstructed_path)

        # Generate a difference map
        difference_map = compare_voxels(lesioned_scan, reconstructed_scan)

        # Threshold the difference map to find significant differences
        significant_diff = threshold_difference(difference_map)

        # Save the significant difference map as a NIfTI file
        output_path = os.path.join(output_folder, file)
        save_difference_map(significant_diff, affine, output_path)
    else:
        print(f"No corresponding file found for {file}")

print("Processing complete.")
