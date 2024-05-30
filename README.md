# Lesion_Healing_Model
A spin-off project from 3D-Lesion-Segmentation, dedicated to creating a model that can heal lesioned CT images and compare them against the original to locate lesions.

# Project Introduction

This project is under Dr. Robert Stevens at the Laboratory for Computational Intensive Care Medicine at Johns Hopkins University.

The purpose behind the project is to create a deep learning model that can reliably detect lesions in CT scans of acute ischemic stroke patients. Small lesions are notoriously difficult to find in CT scans, and usually require an MRI scan to see, a process which takes time that patients may not have. Our project aims to find a way to locate lesions without the need for MRIs

This repository represents an offshoot of the original project which takes a new approach. Rather than directly scanning each voxel for lesions and comparing to a ground truth label, this approach, inspired the paper "Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models" aims to create a VQ-VAE model that can take an input of lesioned CT scans and output a "healed" image with no lesions, which can then be compared to the original image to detect the exact locations of the lesion.

This approach is being tested by undergraduate researcher Ansh Goyal under the guidance of graduate researchers Yanlin Wu and Xinyuan Fang alongside Dr. Robert Stevens.

Note: Training and testing data is not stored in this repository.