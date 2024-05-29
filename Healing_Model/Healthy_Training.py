from VQ_VAE_Implementation import VQVAE
import torch
from torch.utils.data import DataLoader
from HealthyData import HealthyDataset
from torch.nn import functional as F

dataset_directory = '/home/agoyal19/Dataset/control_data'  # Update this to your directory
dataset = HealthyDataset(directory=dataset_directory)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model Initialization
healthy_model = VQVAE(input_channels=1, hidden_channels=64, embedding_dim=64, num_embeddings=256, commitment_cost=0.25)

# Choose a device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
healthy_model.to(device)

# Optimizer
optimizer = torch.optim.Adam(healthy_model.parameters(), lr=1e-4)

# Training loop
for epoch in range(500):
    epoch_loss = 0
    for batch_data in data_loader:
        images = batch_data['ct'].to(device).float()  # Ensure your dataset outputs the correct key
        
        optimizer.zero_grad()
        
        reconstructed_images, quantization_loss, _ = healthy_model(images)
        reconstruction_loss = torch.mean((reconstructed_images - images) ** 2)
        loss = reconstruction_loss + quantization_loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{500}, Loss: {epoch_loss/len(data_loader)}")

# Saving the trained model
model_save_path = "/home/agoyal19/My_Work/Healing_Model/Healthy_Models/healthy_vqvae.pth"
torch.save(healthy_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
