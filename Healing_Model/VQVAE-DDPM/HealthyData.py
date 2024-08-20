import os
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset

class HealthyDataset(Dataset):
    def __init__(self, directory):
        """
        Args:
            directory (string): Directory with all the .nii files.
        """
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        # Load the .nii file and get its data as a numpy array
        image = nib.load(file_path).get_fdata()
        
        # Normalize the image data
        image = (image - np.mean(image)) / np.std(image)
        
        # Convert the numpy array to a PyTorch tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Adding channel dimension
        
        return {'ct': image_tensor}