#!/bin/bash

#SBATCH --account=rsteven1_gpu
#SBATCH --job-name=healing_pipeline
#SBATCH --nodes=2
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --time=50:00:00
#SBATCH --mem=100G
#SBATCH --export=ALL

# Step 1: Train the VQ-VAE model on healthy brain scans
#echo "Starting VQ-VAE training on healthy data..."
#python /home/agoyal19/My_Work/Healing_Model/VQVAE-DDPM/Healthy_Training_VAE.py
#echo "VQ-VAE training completed."

# Step 2: Train the DDPM model on latent representations of healthy brain scans
echo "Starting DDPM training on latent representations from VQ-VAE..."
python /home/agoyal19/My_Work/Healing_Model/VQVAE-DDPM/DDPM_Training.py
echo "DDPM training completed."

# Step 3: Apply the trained VQ-VAE and DDPM models to reconstruct lesioned brain scans
echo "Starting lesion healing and reconstruction with VQ-VAE and DDPM..."
python /home/agoyal19/My_Work/Healing_Model/VQVAE-DDPM/Lesion_Healing_VAE_with_DDPM.py
echo "Lesion healing and reconstruction completed."

# Step 4: Compare the original lesioned images with the reconstructed images
echo "Starting comparison of original and reconstructed images..."
python /home/agoyal19/My_Work/Healing_Model/VQVAE-DDPM/Reconstruction_Comp_VAE_with_DDPM.py
echo "Comparison completed."

echo "All steps have been successfully completed."