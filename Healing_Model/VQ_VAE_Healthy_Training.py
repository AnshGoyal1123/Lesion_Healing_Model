from VQ_VAE_Implementation import VQVAE
import torch
from torch.utils.data import DataLoader
from CTDataset import StrokeAI  # Ensure this is the correct path to your StrokeAI dataset class

# Initialize dataset with only healthy images for training
dataset = StrokeAI(
    CT_root="../Dataset/data/images/",
    DWI_root="/scratch4/rsteven1/StrokeAI/CTMRI_coreistration",
    ADC_root="/scratch4/rsteven1/StrokeAI/",
    label_root="/scratch4/rsteven1/StrokeAI/CTMRI_coreistration",
    MRI_type='DWI',
    mode='train',  # Ensure this filters for healthy images only
    bounding_box=True,
    instance_normalize=True,
    padding=False,
    crop=False,  # Adjust based on your requirement
    RotatingResize=False
)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model Initialization
healthy_model = VQVAE(input_channels=1, hidden_channels=64, embedding_dim=64, num_embeddings=256, commitment_cost=0.25)

# Choose a device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
healthy_model.to(device)

# Optimizer
optimizer = torch.optim.Adam(healthy_model.parameters(), lr=1e-4)

# Training loop
num_epochs = 100  # Set the number of epochs
for epoch in range(num_epochs):
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
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}")

# Saving the trained model
model_save_path = "vqvae_trained_on_healthy_images.pth"
torch.save(healthy_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
