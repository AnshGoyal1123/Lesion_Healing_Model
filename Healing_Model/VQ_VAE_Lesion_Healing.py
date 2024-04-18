import torch
from torch.utils.data import DataLoader
from LesionedData import LesionedDataset  # Modify with your actual dataset import
from VQ_VAE_Implementation import VQVAE  # Modify with your actual model import
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model_path = '/home/agoyal19/My_Work/Lesion_Healing_Model/Healing_Model/Healthy_Model/vqvae_trained_on_healthy_images.pth'  # Update this path
model = VQVAE(input_channels=1, hidden_channels=64, embedding_dim=64, num_embeddings=256, commitment_cost=0.25)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Prepare your dataset
dataset_directory = '/home/agoyal19/Dataset/data/images/'  # Update this path
dataset = LesionedDataset(directory=dataset_directory)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)  # Batch size can be adjusted

# Validation phase
total_loss = 0
with torch.no_grad():
    for batch_data in data_loader:
        images = batch_data['ct'].to(device).float()

        # Forward pass through the model
        reconstructed_images, quantization_loss, _ = model(images)
        
        # Calculate the reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed_images, images)
        loss = reconstruction_loss + quantization_loss
        total_loss += loss.item()

        # Optionally save or display images for visual comparison
        # This would be a good place to implement visualizations or save output

# Compute average loss
average_loss = total_loss / len(data_loader)
print(f'Average Validation Loss: {average_loss:.4f}')
