import os
import torch
import nibabel as nib
from torch.utils.data import Dataset

class LesionedDataset(Dataset):
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
        
        # Convert the numpy array to a PyTorch tensor
        # Adding a new axis at the beginning for the channel dimension
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        return {'ct': image_tensor}
