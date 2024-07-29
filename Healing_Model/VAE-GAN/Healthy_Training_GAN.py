import torch
from HealthyData import HealthyDataset
from VAE_GAN_Implementation import VQVAEGAN, train_vqvaegan, save_model

# Dataset Directory
dataset_directory = '/home/agoyal19/Dataset/control_data'  # Update this to your directory
dataset = HealthyDataset(directory=dataset_directory)

# Model Initialization
input_channels = 1  # Assuming single-channel (grayscale) 3D brain images
hidden_channels = 32
embedding_dim = 64
num_embeddings = 512  # Number of discrete embeddings
commitment_cost = 0.25  # Weisght of the commitment loss

model = VQVAEGAN(input_channels, hidden_channels, embedding_dim, num_embeddings, commitment_cost)

# Train the model
num_epochs = 500
batch_size = 4
learning_rate = 1e-4
trained_model = train_vqvaegan(model, dataset, num_epochs=num_epochs, batch_size=batch_size, lr=learning_rate)

# Save the trained model
model_save_path = "/home/agoyal19/My_Work/Healing_Model/Healthy_Models/healthy_vqvaegan.pth"
save_model(trained_model, model_save_path)