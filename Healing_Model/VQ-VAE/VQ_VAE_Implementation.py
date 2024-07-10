import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, x):
        # Flatten input
        flat_input = x.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(hidden_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_channels, hidden_channels, embedding_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose3d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose3d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(input_channels, hidden_channels, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, quantization_loss, perplexity = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, quantization_loss, perplexity