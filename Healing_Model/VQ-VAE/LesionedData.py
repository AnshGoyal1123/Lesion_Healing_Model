import os
import torch
import nibabel as nib
from torch.utils.data import Dataset

class LesionedDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
        print(f"Loaded {len(self.filenames)} files from {directory}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.directory, filename)
        image = nib.load(file_path).get_fdata()
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Adding channel dimension
        return {'ct': image_tensor, 'filename': filename}
