import torch
from HealthyData import HealthyDataset
from VAE_GAN_Implementation import VAEGAN, train_vaegan, save_model

# Dataset Directory
dataset_directory = '/home/agoyal19/Dataset/control_data'  # Update this to your directory
dataset = HealthyDataset(directory=dataset_directory)

# Model Initialization
input_channels = 1  # Assuming grayscale brain images
hidden_channels = 32
z_dim = 64
model = VAEGAN(input_channels, hidden_channels, z_dim)

# Train the model
trained_model = train_vaegan(model, dataset, num_epochs=500, batch_size=4, lr=1e-4)

# Save the trained model
model_save_path = "/home/agoyal19/My_Work/Healing_Model/Healthy_Models/healthy_vaegan.pth"
save_model(trained_model, model_save_path)